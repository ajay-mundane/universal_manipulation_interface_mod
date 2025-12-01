#!/usr/bin/env python3

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import pickle
from tqdm import tqdm

from diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseDataset
from diffusion_policy.common.pytorch_util import dict_apply
import zmq
OmegaConf.register_new_resolver("eval", eval, replace=True)


class MundaneEpisodeEvaluator:
    """
    Evaluate model performance on a single episode from the mundane dataset.
    """
    
    def __init__(self, cfg: DictConfig):
        """
        Initialize the evaluator with Hydra config.
        
        Args:
            cfg: Hydra config containing dataset configuration
        """
        self.cfg = cfg
        
        # Load dataset using Hydra instantiation (same as training)
        self.dataset: BaseDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(self.dataset, BaseImageDataset) or isinstance(self.dataset, BaseDataset)
        
        print(f"Loaded dataset with {len(self.dataset)} samples across {self.dataset.replay_buffer.n_episodes} episodes")

        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f"tcp://localhost:8766")
        
    
    def clear_episode_memory(self, episode_data):
        """
        Clear memory-intensive episode data to prevent accumulation.
        Updated for streaming episode data structure.
        """
        if episode_data is not None:
            # Clear dataset reference to prevent holding onto it
            if 'dataset' in episode_data:
                del episode_data['dataset']
            # Clear sample indices 
            if 'sample_indices' in episode_data:
                del episode_data['sample_indices']
    
    def get_episode_indices_for_episode_id(self, episode_id: int):
        """
        Get all dataset indices that correspond to a specific episode_id.
        This mimics how the dataset creates training samples from episodes.
        
        Args:
            episode_id: The episode ID to get indices for
            
        Returns:
            list: List of dataset indices that correspond to this episode
        """
        if episode_id >= self.dataset.replay_buffer.n_episodes:
            raise ValueError(f"Episode {episode_id} does not exist. Dataset has {self.dataset.replay_buffer.n_episodes} episodes.")
        
        # Get episode start/end from replay buffer
        episode_ends = self.dataset.replay_buffer.episode_ends[:]
        start_idx = 0 if episode_id == 0 else episode_ends[episode_id - 1]
        end_idx = episode_ends[episode_id]
        
        # Find all dataset sampler indices that belong to this episode
        episode_indices = []
        for dataset_idx in range(len(self.dataset.sampler.indices)):
            current_idx, sampler_start_idx, sampler_end_idx, _ = self.dataset.sampler.indices[dataset_idx]
            
            # Check if this sample comes from our target episode
            if sampler_start_idx == start_idx and sampler_end_idx == end_idx:
                episode_indices.append(dataset_idx)
        
        print(f"Episode {episode_id} spans timesteps {start_idx}-{end_idx} and has {len(episode_indices)} dataset samples")
        return episode_indices
    
    def get_random_episode_data(self, episode_id: int = None):
        """
        Get all dataset samples for a random (or specified) episode.
        Uses the same data processing as training (via dataset.__getitem__).
        
        Args:
            episode_id: Specific episode ID to use, or None for random selection
            
        Returns:
            dict: Episode data containing observations and actions as processed by dataset
        """
        if episode_id is None:
            episode_id = random.randint(0, self.dataset.replay_buffer.n_episodes - 1)
        
        print(f"Selected episode {episode_id}")
        
        # Get all dataset indices for this episode
        episode_indices = self.get_episode_indices_for_episode_id(episode_id)
        
        if len(episode_indices) == 0:
            print(f"Warning: No dataset samples found for episode {episode_id}")
            return None
        
        # Create streaming episode data structure instead of loading all samples
        episode_data = {
            'episode_id': episode_id,
            'num_samples': len(episode_indices),
            'sample_indices': episode_indices,
            'dataset': self.dataset,  # Keep reference to dataset for streaming
            'samples': None  # Will be loaded on-demand
        }
        
        print(f"Episode {episode_id} configured for streaming with {len(episode_indices)} samples")
        return episode_data

        # # Organize the data
        # obs_keys = list(episode_samples[0]['obs'].keys())
        # action_shape = episode_samples[0]['action'].shape
        # episode_data = {
        #     'episode_id': episode_id,
        #     'num_samples': len(episode_samples),
        #     'observations': {key: [] for key in obs_keys},
        #     'actions': [],
        #     'sample_indices': episode_indices
        # }
        
        # # Collect all observations and actions
        # for sample in episode_samples:
        #     for key in obs_keys:
        #         episode_data['observations'][key].append(sample['obs'][key])
        #     episode_data['actions'].append(sample['action'])
        
        # # Convert lists to tensors/arrays
        # for key in obs_keys:
        #     episode_data['observations'][key] = torch.stack(episode_data['observations'][key])
        #     print(f"Observations {key} shape: {episode_data['observations'][key].shape}")
        
        # episode_data['actions'] = torch.stack(episode_data['actions'])
        # print(f"Actions shape: {episode_data['actions'].shape}")
        
        # return episode_data
    
    def get_model_predictions(self, episode_data):
        """
        Get model predictions for the episode data.
        This is a placeholder for your model inference script.
        
        Args:
            episode_data: Episode data dict from get_random_episode_data()
            
        Returns:
            torch.Tensor: Predicted actions with same structure as ground truth
        """
        # =============================================
        # TODO: IMPLEMENT YOUR MODEL INFERENCE HERE
        # =============================================
        # 
        # Steps to implement:
        # 1. Load your trained model checkpoint
        # 2. Apply the same normalizer used during training
        # 3. For each sample in episode_data:
        #    - Extract observations
        #    - Run model prediction
        #    - Collect predicted actions
        # 4. Return all predictions
        #
        predictions = []
        
        # Stream samples one by one instead of loading all at once
        for i, dataset_idx in enumerate(tqdm(episode_data['sample_indices'], desc="Running model inference")):
            # Load sample on-demand
            sample = episode_data['dataset'][dataset_idx]
            
            obs_dict_np = sample['obs']
            self.socket.send_pyobj(obs_dict_np)
            action = self.socket.recv_pyobj()
            predictions.append(torch.from_numpy(action))
            
            # Clear sample data immediately after use
            del sample
            del obs_dict_np
        
        result = torch.stack(predictions, dim=0)
        
        # Clear predictions list to free memory
        del predictions
        
        return result
    
    def compute_mse_loss(self, predictions, ground_truth):
        """
        Compute Mean Squared Error between predictions and ground truth.
        
        Args:
            predictions: Model predictions (num_samples, action_horizon, action_dim)
            ground_truth: Ground truth actions (num_samples, action_horizon, action_dim)
            
        Returns:
            dict: MSE metrics
        """
        # Convert to numpy if tensors
        if torch.is_tensor(predictions):
            predictions = predictions.numpy()
        if torch.is_tensor(ground_truth):
            ground_truth = ground_truth.numpy()
        
        # Overall MSE
        mse_total = np.mean((predictions - ground_truth) ** 2)
        
        # Per-sample MSE (across action_horizon and action_dim)
        mse_per_sample = np.mean((predictions - ground_truth) ** 2, axis=(1, 2))
        
        # Per-action-dimension MSE (averaged across samples and time)
        mse_per_dim = np.mean((predictions - ground_truth) ** 2, axis=(0, 1))
        
        # MSE per action horizon step (averaged across samples and action dims)
        mse_per_timestep = np.mean((predictions - ground_truth) ** 2, axis=(0, 2))
        
        # MSE for each robot (assuming 7 dims per robot for 14D action space)
        if predictions.shape[-1] == 14:
            # Robot 0: joints [0:6], gripper [6]
            mse_robot0 = np.mean((predictions[..., :7] - ground_truth[..., :7]) ** 2)
            mse_robot0_joints = np.mean((predictions[..., :6] - ground_truth[..., :6]) ** 2)
            mse_robot0_gripper = np.mean((predictions[..., 6] - ground_truth[..., 6]) ** 2)
            
            # Robot 1: joints [7:13], gripper [13]
            mse_robot1 = np.mean((predictions[..., 7:] - ground_truth[..., 7:]) ** 2)
            mse_robot1_joints = np.mean((predictions[..., 7:13] - ground_truth[..., 7:13]) ** 2)
            mse_robot1_gripper = np.mean((predictions[..., 13] - ground_truth[..., 13]) ** 2)
            
            # Per-sample MSE for robot components (averaged over action horizon and joint dimensions)
            mse_per_sample_robot0_joints = np.mean((predictions[..., :6] - ground_truth[..., :6]) ** 2, axis=(1, 2))
            mse_per_sample_robot0_gripper = np.mean((predictions[..., 6] - ground_truth[..., 6]) ** 2, axis=1)
            mse_per_sample_robot1_joints = np.mean((predictions[..., 7:13] - ground_truth[..., 7:13]) ** 2, axis=(1, 2))
            mse_per_sample_robot1_gripper = np.mean((predictions[..., 13] - ground_truth[..., 13]) ** 2, axis=1)
            
            # Per-sample MSE for early action horizon (0:4) for robot components
            mse_per_sample_robot0_joints_early = np.mean((predictions[:, :4, :6] - ground_truth[:, :4, :6]) ** 2, axis=(1, 2))
            mse_per_sample_robot0_gripper_early = np.mean((predictions[:, :4, 6] - ground_truth[:, :4, 6]) ** 2, axis=1)
            mse_per_sample_robot1_joints_early = np.mean((predictions[:, :4, 7:13] - ground_truth[:, :4, 7:13]) ** 2, axis=(1, 2))
            mse_per_sample_robot1_gripper_early = np.mean((predictions[:, :4, 13] - ground_truth[:, :4, 13]) ** 2, axis=1)
        else:
            # Fallback for different action dimensions
            mid_dim = predictions.shape[-1] // 2
            mse_robot0 = np.mean((predictions[..., :mid_dim] - ground_truth[..., :mid_dim]) ** 2)
            mse_robot1 = np.mean((predictions[..., mid_dim:] - ground_truth[..., mid_dim:]) ** 2)
            
            # Assume last dimension of each robot is gripper
            joints_per_robot = mid_dim - 1
            mse_robot0_joints = np.mean((predictions[..., :joints_per_robot] - ground_truth[..., :joints_per_robot]) ** 2)
            mse_robot0_gripper = np.mean((predictions[..., joints_per_robot] - ground_truth[..., joints_per_robot]) ** 2)
            mse_robot1_joints = np.mean((predictions[..., mid_dim:mid_dim+joints_per_robot] - ground_truth[..., mid_dim:mid_dim+joints_per_robot]) ** 2)
            mse_robot1_gripper = np.mean((predictions[..., -1] - ground_truth[..., -1]) ** 2)
            
            # Per-sample MSE for robot components (averaged over action horizon and joint dimensions)
            mse_per_sample_robot0_joints = np.mean((predictions[..., :joints_per_robot] - ground_truth[..., :joints_per_robot]) ** 2, axis=(1, 2))
            mse_per_sample_robot0_gripper = np.mean((predictions[..., joints_per_robot] - ground_truth[..., joints_per_robot]) ** 2, axis=1)
            mse_per_sample_robot1_joints = np.mean((predictions[..., mid_dim:mid_dim+joints_per_robot] - ground_truth[..., mid_dim:mid_dim+joints_per_robot]) ** 2, axis=(1, 2))
            mse_per_sample_robot1_gripper = np.mean((predictions[..., -1] - ground_truth[..., -1]) ** 2, axis=1)
            
            # Per-sample MSE for early action horizon (0:4) for robot components
            mse_per_sample_robot0_joints_early = np.mean((predictions[:, :4, :joints_per_robot] - ground_truth[:, :4, :joints_per_robot]) ** 2, axis=(1, 2))
            mse_per_sample_robot0_gripper_early = np.mean((predictions[:, :4, joints_per_robot] - ground_truth[:, :4, joints_per_robot]) ** 2, axis=1)
            mse_per_sample_robot1_joints_early = np.mean((predictions[:, :4, mid_dim:mid_dim+joints_per_robot] - ground_truth[:, :4, mid_dim:mid_dim+joints_per_robot]) ** 2, axis=(1, 2))
            mse_per_sample_robot1_gripper_early = np.mean((predictions[:, :4, -1] - ground_truth[:, :4, -1]) ** 2, axis=1)
        
        metrics = {
            'mse_total': mse_total,
            'mse_per_sample': mse_per_sample,
            'mse_per_timestep': mse_per_timestep,
            'mse_per_dimension': mse_per_dim,
            'mse_robot0': mse_robot0,
            'mse_robot1': mse_robot1,
            'mse_robot0_joints': mse_robot0_joints,
            'mse_robot0_gripper': mse_robot0_gripper,
            'mse_robot1_joints': mse_robot1_joints,
            'mse_robot1_gripper': mse_robot1_gripper,
            'rmse_total': np.sqrt(mse_total),
            # Per-sample metrics for video visualization
            'mse_per_sample_robot0_joints': mse_per_sample_robot0_joints,
            'mse_per_sample_robot0_gripper': mse_per_sample_robot0_gripper,
            'mse_per_sample_robot1_joints': mse_per_sample_robot1_joints,
            'mse_per_sample_robot1_gripper': mse_per_sample_robot1_gripper,
            'mse_per_sample_robot0_joints_early': mse_per_sample_robot0_joints_early,
            'mse_per_sample_robot0_gripper_early': mse_per_sample_robot0_gripper_early,
            'mse_per_sample_robot1_joints_early': mse_per_sample_robot1_joints_early,
            'mse_per_sample_robot1_gripper_early': mse_per_sample_robot1_gripper_early
        }
        
        return metrics
    
    def visualize_comparison(self, episode_data, predictions, metrics, save_dir=None):
        """
        Create visualizations comparing predictions vs ground truth.
        
        Args:
            episode_data: Episode data dict
            predictions: Model predictions
            metrics: MSE metrics dict
            save_dir: Optional directory to save plots
        """
        # Extract ground truth actions by streaming samples instead of from pre-loaded samples
        ground_truth_actions = []
        for dataset_idx in episode_data['sample_indices']:
            sample = episode_data['dataset'][dataset_idx]
            ground_truth_actions.append(sample['action'])
            del sample  # Clear immediately
        
        ground_truth = torch.stack(ground_truth_actions)
        del ground_truth_actions  # Clear list
        
        episode_id = episode_data['episode_id']
        
        # Convert to numpy if tensors
        if torch.is_tensor(predictions):
            predictions = predictions.numpy()
        if torch.is_tensor(ground_truth):
            ground_truth = ground_truth.numpy()
        
        # Store for video visualization
        self._current_predictions = predictions
        self._current_ground_truth = ground_truth
        self._current_metrics = metrics
        
        # Average over action horizon for visualization (num_samples, action_dim)
        pred_avg = np.mean(predictions, axis=1) 
        gt_avg = np.mean(ground_truth, axis=1)
        
        # Create larger plot to accommodate all joints plus metrics
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        fig.suptitle(f'Episode {episode_id} - Model vs Ground Truth Comparison\n'
                    f'Total RMSE: {metrics["rmse_total"]:.4f} '
                    f'(Robot0: {np.sqrt(metrics["mse_robot0"]):.4f}, '
                    f'Robot1: {np.sqrt(metrics["mse_robot1"]):.4f})', fontsize=16)
        
        sample_indices = np.arange(len(gt_avg))
        
        # Plot all dimensions in one loop
        for i in range(min(14, predictions.shape[-1])):
            row = i // 4
            col = i % 4
            
            if row < 4 and col < 4:  # Make sure we don't exceed subplot grid
                ax = axes[row, col]
                
                # Determine robot and joint type
                if i < 7:  # Robot 0 territory (0-6)
                    robot_name = "Robot 0"
                    if i == 6:
                        joint_type = "Gripper"
                    else:
                        joint_type = f"Joint {i}"
                else:  # Robot 1 territory (7-13)
                    robot_name = "Robot 1"
                    if i == 13:
                        joint_type = "Gripper"
                    else:
                        joint_type = f"Joint {i-7}"
                
                ax.plot(sample_indices, gt_avg[:, i], label=f'GT', alpha=0.7, linewidth=2)
                ax.plot(sample_indices, pred_avg[:, i], label=f'Pred', alpha=0.7, linestyle='--', linewidth=2)
                ax.set_title(f'{robot_name} - {joint_type}', fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Action Value')
        
        # Use remaining subplots for metrics
        # Plot MSE over samples
        if len(axes.flat) > 14:
            ax_mse_sample = axes[3, 2]
            ax_mse_sample.plot(sample_indices, metrics['mse_per_sample'])
            ax_mse_sample.set_title('MSE per Sample')
            ax_mse_sample.set_xlabel('Sample Index')
            ax_mse_sample.set_ylabel('MSE')
            ax_mse_sample.grid(True)
        
        # Plot MSE per dimension
        if len(axes.flat) > 15:
            ax_mse_dim = axes[3, 3]
            ax_mse_dim.bar(range(len(metrics['mse_per_dimension'])), metrics['mse_per_dimension'])
            ax_mse_dim.set_title('MSE per Action Dimension')
            ax_mse_dim.set_xlabel('Action Dimension')
            ax_mse_dim.set_ylabel('MSE')
            ax_mse_dim.grid(True)
        
        # plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / f"episode_{episode_id}_comparison.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison plot to {save_path}")
        
        # plt.show()
        
        # Display sample camera feeds - always available with streaming
        self._visualize_camera_feeds(episode_data, save_dir)
    
    def _visualize_camera_feeds(self, episode_data, save_dir=None):
        """Create video visualization with camera feeds and per-timestep metrics."""
        episode_id = episode_data['episode_id']
        num_samples = episode_data['num_samples']
        sample_indices = episode_data['sample_indices']
        
        # Get first sample to determine camera structure
        if len(sample_indices) == 0:
            print("No samples found for visualization")
            return
            
        first_sample = episode_data['dataset'][sample_indices[0]]
        first_obs = first_sample['obs']
        camera_keys = [key for key in first_obs.keys() if 'camera' in key and 'rgb' in key]
        
        # Clean up first sample after getting camera keys
        del first_sample
        
        if len(camera_keys) == 0:
            print("No camera feeds found for visualization")
            return
            
        # Only create video if save_dir is provided
        if not save_dir:
            print("No save_dir provided, skipping video creation")
            return
            
        import cv2
        save_path = Path(save_dir) / f"episode_{episode_id}_camera_metrics.mp4"
        
        # Get first frame to determine dimensions by loading first sample
        first_sample_temp = episode_data['dataset'][sample_indices[0]]
        first_cam = camera_keys[0]
        obs_data = first_sample_temp['obs'][first_cam]
        
        # Get current timestep frame (last index is current timestep)
        if len(obs_data.shape) == 4:  # (obs_horizon, C, H, W)
            sample_frame = obs_data[-1]  # Last timestep is current
        elif len(obs_data.shape) == 3:  # (C, H, W)
            sample_frame = obs_data
        else:
            print(f"Unexpected observation shape for {first_cam}: {obs_data.shape}")
            return
        
        # Convert from CxHxW to HxWxC for display
        if torch.is_tensor(sample_frame):
            sample_frame = sample_frame.numpy()
        sample_frame = np.moveaxis(sample_frame, 0, 2)
        
        # Clean up temporary sample
        del first_sample_temp
        
        # Get individual frame dimensions
        frame_height, frame_width = sample_frame.shape[:2]
        
        # Create 2x3 grid (for 5 cameras, one slot will be empty)
        grid_rows, grid_cols = 2, 3
        grid_height = grid_rows * frame_height + 100  # Extra space for metrics text
        grid_width = grid_cols * frame_width
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(save_path), fourcc, 30.0, (grid_width, grid_height))
        
        # Debug: Print first 10 current_idx values to verify they're sequential
        debug_current_idxs = []
        for sample_idx in range(min(10, num_samples)):
            dataset_idx = sample_indices[sample_idx]
            current_idx, _, _, _ = self.dataset.sampler.indices[dataset_idx]
            debug_current_idxs.append(current_idx)
        print(f"First 10 current_idx values: {debug_current_idxs}")
        
        print(f"Creating video visualization for {num_samples} timesteps...")
        
        for i, sample_idx in enumerate(tqdm(range(num_samples), desc="Generating video frames")):
            # Load sample on-demand
            dataset_idx = sample_indices[sample_idx]
            current_sample = episode_data['dataset'][dataset_idx]
            
            # Create blank grid frame
            grid_frame = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
            
            # Calculate per-timestep metrics if predictions are available
            metrics_lines = [f"Timestep: {sample_idx}"]
            if hasattr(self, '_current_metrics') and hasattr(self, '_current_predictions'):
                # Use pre-computed per-sample metrics
                metrics = self._current_metrics
                
                # Full action horizon metrics for current sample
                robot0_joints_mse = metrics['mse_per_sample_robot0_joints'][sample_idx]
                robot0_gripper_mse = metrics['mse_per_sample_robot0_gripper'][sample_idx]
                robot1_joints_mse = metrics['mse_per_sample_robot1_joints'][sample_idx]
                robot1_gripper_mse = metrics['mse_per_sample_robot1_gripper'][sample_idx]
                
                # Early action horizon (0:4) metrics for current sample
                robot0_joints_mse_early = metrics['mse_per_sample_robot0_joints_early'][sample_idx]
                robot0_gripper_mse_early = metrics['mse_per_sample_robot0_gripper_early'][sample_idx]
                robot1_joints_mse_early = metrics['mse_per_sample_robot1_joints_early'][sample_idx]
                robot1_gripper_mse_early = metrics['mse_per_sample_robot1_gripper_early'][sample_idx]
                
                metrics_lines.append(f"Full Horizon RMSE:")
                metrics_lines.append(f"R0J: {np.sqrt(robot0_joints_mse):.4f} R0G: {np.sqrt(robot0_gripper_mse):.4f}")
                metrics_lines.append(f"R1J: {np.sqrt(robot1_joints_mse):.4f} R1G: {np.sqrt(robot1_gripper_mse):.4f}")
                metrics_lines.append(f"Early (0:4) RMSE:")
                metrics_lines.append(f"R0J: {np.sqrt(robot0_joints_mse_early):.4f} R0G: {np.sqrt(robot0_gripper_mse_early):.4f}")
                metrics_lines.append(f"R1J: {np.sqrt(robot1_joints_mse_early):.4f} R1G: {np.sqrt(robot1_gripper_mse_early):.4f}")
            
            # Place camera frames in 2x3 grid
            for idx, cam_name in enumerate(camera_keys):
                if idx >= 6:  # Only support 6 cameras max in 2x3 grid
                    break
                    
                row = idx // grid_cols
                col = idx % grid_cols
                
                obs_data = current_sample['obs'][cam_name]
                
                # Get current timestep frame (last index is current timestep)
                if len(obs_data.shape) == 4:  # (obs_horizon, C, H, W)
                    current_frame = obs_data[-1]  # Last timestep is current
                # elif len(obs_data.shape) == 3:  # (C, H, W)
                #     current_frame = obs_data
                else:
                    print(f"Unexpected observation shape for {cam_name}: {obs_data.shape}")
                    continue
                
                # Convert from CxHxW to HxWxC for display
                if torch.is_tensor(current_frame):
                    current_frame = current_frame.numpy()
                current_frame = np.moveaxis(current_frame, 0, 2)
                
                # Ensure values are in [0, 255] range for proper video encoding
                if current_frame.max() <= 1.0:
                    # If values are in [0,1], scale to [0,255]
                    current_frame = (current_frame * 255).astype(np.uint8)
                else:
                    # If already in [0,255] range, just clip and convert
                    current_frame = np.clip(current_frame, 0, 255).astype(np.uint8)
                
                # Place frame in grid
                y_start = row * frame_height
                y_end = y_start + frame_height
                x_start = col * frame_width
                x_end = x_start + frame_width
                
                grid_frame[y_start:y_end, x_start:x_end] = current_frame
                
                # Add camera name text
                cv2.putText(grid_frame, cam_name, (x_start + 10, y_start + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add metrics text at the bottom
            text_y_start = grid_rows * frame_height + 20
            for i, line in enumerate(metrics_lines):
                y_pos = text_y_start + i * 20
                cv2.putText(grid_frame, line, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame to video
            video_writer.write(grid_frame)
            
            # Clear frame and sample data to prevent memory accumulation
            del grid_frame
            del current_sample
        
        # Finalize video
        video_writer.release()
        print(f"Saved video visualization to {save_path}")
        print("Video visualization completed!")
        
        # Force garbage collection after video generation
        import gc
        gc.collect()
    
    def run_evaluation(self, episode_id=None, save_dir=None):
        """
        Run full evaluation pipeline on a single episode.
        
        Args:
            episode_id: Specific episode to evaluate, or None for random
            save_dir: Directory to save visualizations
            
        Returns:
            dict: Complete evaluation results
        """
        print("="*50)
        print("Starting Mundane Episode Evaluation")
        print("="*50)
        
        # Get episode data
        episode_data = self.get_random_episode_data(episode_id)
        if episode_data is None:
            print("Failed to get episode data")
            return None
            
        # Get model predictions
        predictions = self.get_model_predictions(episode_data)
        
        # Extract ground truth actions by streaming samples
        ground_truth_actions = []
        for dataset_idx in episode_data['sample_indices']:
            sample = episode_data['dataset'][dataset_idx]
            ground_truth_actions.append(sample['action'])
            del sample  # Clear immediately
        
        ground_truth = torch.stack(ground_truth_actions)
        del ground_truth_actions  # Clear list
        
        # Compute evaluation metrics
        metrics = self.compute_mse_loss(predictions, ground_truth)
        
        # Clear ground truth tensor after metrics computation
        del ground_truth
        
        # Print results
        print("\nEvaluation Results:")
        print(f"Episode ID: {episode_data['episode_id']}")
        print(f"Number of samples: {episode_data['num_samples']}")
        print(f"Total MSE: {metrics['mse_total']:.6f}")
        print(f"Total RMSE: {metrics['rmse_total']:.6f}")
        print(f"\nRobot-wise MSE:")
        print(f"Robot 0 Total MSE: {metrics['mse_robot0']:.6f}")
        print(f"  - Joints MSE: {metrics['mse_robot0_joints']:.6f}")
        print(f"  - Gripper MSE: {metrics['mse_robot0_gripper']:.6f}")
        print(f"Robot 1 Total MSE: {metrics['mse_robot1']:.6f}")
        print(f"  - Joints MSE: {metrics['mse_robot1_joints']:.6f}")
        print(f"  - Gripper MSE: {metrics['mse_robot1_gripper']:.6f}")
        
        # Create visualizations
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.visualize_comparison(episode_data, predictions, metrics, save_dir)
        
        # Clear cached visualization data to free memory
        if hasattr(self, '_current_predictions'):
            del self._current_predictions
        if hasattr(self, '_current_ground_truth'):
            del self._current_ground_truth
        if hasattr(self, '_current_metrics'):
            del self._current_metrics
        
        results = {
            'episode_data': episode_data,
            'predictions': predictions,
            'metrics': metrics
        }
        
        # Clear large data immediately after creating results dict
        # Keep references in results but clear our local variables
        del episode_data
        del predictions
        
        print("\nEvaluation completed!")
        return results


def main():
    """
    Main function to run the evaluation.
    Now uses Hydra config system like the training script.
    """
    pass


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "diffusion_policy" / "config"), 
    config_name="train_diffusion_mundane_bimanual_workspace")
def hydra_main(cfg: DictConfig):
    """
    Hydra main function with config loading.
    Usage: python training_eval.py task=mundane_bimanual [other config overrides]
    """
    
    try:
        # Initialize evaluator with config
        evaluator = MundaneEpisodeEvaluator(cfg)
        
        # Optional: specify output directory for visualizations
        save_dir = cfg.get('save_dir', "./eval_results")
        episode_ids = cfg.get('episode_ids', None)  # List of episode IDs or None for random
        episode_id = cfg.get('episode_id', None)  # Single episode ID (fallback)
        
        # Handle episode selection
        if episode_ids is not None:
            # Multiple episodes specified
            for ep_id in episode_ids:
                print(f"\n{'='*60}")
                print(f"Evaluating Episode {ep_id}")
                print(f"{'='*60}")
                
                results = evaluator.run_evaluation(
                    episode_id=ep_id,
                    save_dir=save_dir
                )
                
                # Clear results to free memory between episodes
                if results is not None:
                    # Clear episode samples which take most memory
                    if 'episode_data' in results:
                        evaluator.clear_episode_memory(results['episode_data'])
                    del results
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Print memory usage for debugging
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                print(f"Memory usage after episode {ep_id}: {memory_mb:.1f} MB")
        else:
            # Single episode (or random if None)
            results = evaluator.run_evaluation(
                episode_id=episode_id,
                save_dir=save_dir
            )
        
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        print("Please check your config and dataset path.")


if __name__ == "__main__":
    hydra_main()