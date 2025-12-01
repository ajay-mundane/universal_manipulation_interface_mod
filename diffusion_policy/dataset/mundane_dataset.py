import copy
from typing import Dict, Optional

import os
from datetime import datetime
import pathlib
import numpy as np
import torch
import zarr
from threadpoolctl import threadpool_limits
from tqdm import trange, tqdm
from filelock import FileLock
import shutil

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.normalize_util import (
    array_to_stats, concatenate_normalizer, get_identity_normalizer_from_stat,
    get_image_identity_normalizer, get_range_normalizer_from_stat)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

register_codecs()

class MundaneDataset(BaseDataset):
    def __init__(self,
        shape_meta: dict,
        dataset_path: str,
        cache_dir: Optional[str]=None,
        pose_repr: dict={},
        action_padding: bool=False,
        temporally_independent_normalization: bool=False,
        repeat_frame_prob: float=0.0,
        seed: int=42,
        val_ratio: float=0.0,
        max_duration: Optional[float]=None
    ):
        # Pose representation: 'absolute' or 'delta'
        self.action_pose_repr = pose_repr.get('action_pose_repr', 'absolute')
        self.obs_pose_repr = pose_repr.get('obs_pose_repr', 'absolute')
        
        if cache_dir is None:
            # load into memory store
            # with zarr.ZipStore(dataset_path, mode='r') as zip_store:
            #     replay_buffer = ReplayBuffer.copy_from_store(
            #         src_store=zip_store, 
            #         store=zarr.MemoryStore()
            #     )
            src_group = zarr.open_group(dataset_path, mode='r')

            # 2. Pass the opened group's store to your copy function
            replay_buffer = ReplayBuffer.copy_from_store(
                src_store=src_group.store, 
                store=zarr.MemoryStore()
            )
        else:
            # determine path name
            mod_time = os.path.getmtime(dataset_path)
            stamp = datetime.fromtimestamp(mod_time).isoformat()
            stem_name = os.path.basename(dataset_path).split('.')[0]
            cache_name = '_'.join([stem_name, stamp])
            cache_dir = pathlib.Path(os.path.expanduser(cache_dir))
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir.joinpath(cache_name + '.zarr.mdb')
            lock_path = cache_dir.joinpath(cache_name + '.lock')
            
            # load cached file
            print('Acquiring lock on cache.')
            with FileLock(lock_path):
                # cache does not exist
                if not cache_path.exists():
                    try:
                        with zarr.LMDBStore(str(cache_path),     
                            writemap=True, metasync=False, sync=False, map_async=True, lock=False
                            ) as lmdb_store:
                            # Handle both zarr directories and zip files
                            if dataset_path.endswith('.zip'):
                                with zarr.ZipStore(dataset_path, mode='r') as zip_store:
                                    print(f"Copying data from zip to {str(cache_path)}")
                                    ReplayBuffer.copy_from_store(
                                        src_store=zip_store,
                                        store=lmdb_store
                                    )
                            else:
                                # Handle zarr directories
                                src_group = zarr.open_group(dataset_path, mode='r')
                                print(f"Copying data from directory to {str(cache_path)}")
                                ReplayBuffer.copy_from_store(
                                    src_store=src_group.store,
                                    store=lmdb_store
                                )
                        print("Cache written to disk!")
                    except Exception as e:
                        shutil.rmtree(cache_path)
                        raise e
            
            # open read-only lmdb store
            store = zarr.LMDBStore(str(cache_path), readonly=True, lock=False)
            replay_buffer = ReplayBuffer.create_from_group(
                group=zarr.group(store)
            )
        
        # Extract keys from shape_meta
        rgb_keys = list()
        lowdim_keys = list()
        key_horizon = dict()
        key_down_sample_steps = dict()
        key_latency_steps = dict()
        
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            # solve obs type
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

            # solve horizons and sampling
            key_horizon[key] = attr['horizon']
            key_latency_steps[key] = attr['latency_steps']
            key_down_sample_steps[key] = attr['down_sample_steps']

        # solve action
        key_horizon['action'] = shape_meta['action']['horizon']
        key_latency_steps['action'] = shape_meta['action']['latency_steps']
        key_down_sample_steps['action'] = shape_meta['action']['down_sample_steps']

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask

        sampler = SequenceSampler(
            shape_meta=shape_meta,
            replay_buffer=replay_buffer,
            rgb_keys=rgb_keys,
            lowdim_keys=lowdim_keys,
            key_horizon=key_horizon,
            key_latency_steps=key_latency_steps,
            key_down_sample_steps=key_down_sample_steps,
            episode_mask=train_mask,
            action_padding=action_padding,
            repeat_frame_prob=repeat_frame_prob,
            max_duration=max_duration
        )
        
        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        self.val_mask = val_mask
        self.action_padding = action_padding
        self.repeat_frame_prob = repeat_frame_prob
        self.max_duration = max_duration
        self.sampler = sampler
        self.temporally_independent_normalization = temporally_independent_normalization
        self.threadpool_limits_is_applied = False

    def _convert_to_delta_representation(self, action_sequence):
        """
        Convert absolute joint positions to delta representation.
        Input shape: (T, 14) where T is sequence length
        Output shape: (T, 14) where first timestep is kept as-is, rest are deltas
        """
        if action_sequence.shape[0] <= 1:
            return action_sequence
        
        delta_actions = np.zeros_like(action_sequence)
        delta_actions[0] = action_sequence[0]  # Keep first timestep as absolute
        delta_actions[1:] = action_sequence[1:] - action_sequence[:-1]  # Compute deltas
        return delta_actions
    
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            shape_meta=self.shape_meta,
            replay_buffer=self.replay_buffer,
            rgb_keys=self.rgb_keys,
            lowdim_keys=self.lowdim_keys,
            key_horizon=self.key_horizon,
            key_latency_steps=self.key_latency_steps,
            key_down_sample_steps=self.key_down_sample_steps,
            episode_mask=self.val_mask,
            action_padding=self.action_padding,
            repeat_frame_prob=self.repeat_frame_prob,
            max_duration=self.max_duration
        )
        val_set.val_mask = ~self.val_mask
        return val_set
    
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # enumerate the dataset and save low_dim data
        data_cache = {key: list() for key in self.lowdim_keys + ['action']}
        self.sampler.ignore_rgb(True)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=16,
            num_workers=4,
        )
        for batch in tqdm(dataloader, desc='iterating dataset to get normalization'):
            for key in self.lowdim_keys:
                data_cache[key].append(copy.deepcopy(batch['obs'][key]))
            data_cache['action'].append(copy.deepcopy(batch['action']))
        self.sampler.ignore_rgb(False)

        for key in data_cache.keys():
            data_cache[key] = np.concatenate(data_cache[key])
            assert data_cache[key].shape[0] == len(self.sampler)
            assert len(data_cache[key].shape) == 3
            B, T, D = data_cache[key].shape
            if not self.temporally_independent_normalization:
                data_cache[key] = data_cache[key].reshape(B*T, D)

        # Action normalization (14 dims total) - similar to UMI approach
        # 2 robots with 7 DOF each (6 joints + 1 gripper)
        num_robots = 2
        dim_per_robot = 7
        action_normalizers = list()
        
        for i in range(num_robots):
            # Joint positions (6 dims per robot) - use range normalization
            start_idx = i * dim_per_robot
            joint_end_idx = start_idx + 6
            action_normalizers.append(get_range_normalizer_from_stat(
                array_to_stats(data_cache['action'][..., start_idx:joint_end_idx])))
            
            # Gripper position (1 dim per robot) - use range normalization  
            gripper_idx = joint_end_idx
            action_normalizers.append(get_range_normalizer_from_stat(
                array_to_stats(data_cache['action'][..., gripper_idx:gripper_idx+1])))

        normalizer['action'] = concatenate_normalizer(action_normalizers)

        # Observation normalization - similar to UMI approach
        for key in self.lowdim_keys:
            stat = array_to_stats(data_cache[key])
            
            # Apply range normalization to joint positions (robots_joint_pos)
            if 'joint_pos' in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif 'action' in key:
                this_normalizer = normalizer['action']
            else:
                # Default to range normalization for other low-dim observations
                this_normalizer = get_range_normalizer_from_stat(stat)

                
            normalizer[key] = this_normalizer

        # Image normalization
        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_normalizer()
            
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.threadpool_limits_is_applied:
            threadpool_limits(1)
            self.threadpool_limits_is_applied = True
            
        data = self.sampler.sample_sequence(idx)

        # Process observations
        obs_dict = dict()
        
        # Process RGB images
        for key in self.rgb_keys:
            if key not in data:
                continue
            # Convert from T,H,W,C to T,C,H,W and normalize to [0,1]
            obs_dict[key] = np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.
            del data[key]
            
        # Process low-dim observations
        for key in self.lowdim_keys:
            obs_data = data[key].astype(np.float32)
            
            if self.obs_pose_repr == 'delta':
                obs_data = self._convert_to_delta_representation(obs_data)
            # If 'absolute', keep as-is
            
            obs_dict[key] = obs_data
            del data[key]

        # Process actions based on pose representation
        action_data = data['action'].astype(np.float32)
        
        if self.action_pose_repr == 'delta':
            action_data = self._convert_to_delta_representation(action_data)
        # If 'absolute', keep as-is
        
        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action_data)
        }
        return torch_data
