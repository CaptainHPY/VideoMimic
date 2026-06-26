from __future__ import annotations
import torch
import warp as wp
import numpy as np
from typing import Optional, Tuple, List
from abc import ABC, abstractmethod

from .sensor_cfg import RaycastingSensorCfg, HeightfieldSensorCfg, DepthCameraSensorCfg, MultiLinkHeightSensorCfg
from .raycaster_patterns import grid_pattern, pinhole_camera_pattern
from .warp_utils import convert_to_warp_mesh, raycast_mesh
from legged_gym.tensor_utils.torch_jit_utils import quat_rotate, calc_heading_quat, quat_mul
from legged_gym.utils.history import HistoryHandler

# @torch.jit.script # JIT compilation fails with type annotations
def euler_xyz_to_quat(euler_angles: torch.Tensor) -> torch.Tensor:
    """Converts Euler angles (roll, pitch, yaw) to quaternions (xyzw)."""
    roll, pitch, yaw = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    # Stack as (x, y, z, w) which is the convention used in Isaac Gym/torch_jit_utils
    return torch.stack([qx, qy, qz, qw], dim=-1)

class RaycastingSensor(ABC):
    """Base class for raycasting sensors."""
    
    def __init__(self, robot, cfg: RaycastingSensorCfg, terrain_vertices: np.ndarray, terrain_indices: np.ndarray):
        """Initialize the raycasting sensor.
        
        Args:
            robot: Robot instance that the sensor is attached to
            cfg: Sensor configuration
            terrain_vertices: Terrain mesh vertices
            terrain_indices: Terrain mesh triangle indices
        """
        self.robot = robot
        self.cfg = cfg
        
        # Initialize warp
        wp.init()
        wp.set_device(cfg.device)
        
        # Create terrain mesh
        self.terrain_mesh = convert_to_warp_mesh(terrain_vertices, terrain_indices, cfg.device)
        
        # Initialize ray buffers - will be set by subclasses
        self.ray_starts = None
        self.ray_directions = None
        self.ray_hits = None
        self.num_rays = 0
        
        # Initialize world frame buffers
        self.ray_starts_world = None
        self.ray_directions_world = None
        
        # Initialize depth buffer
        self.depth_map = None

        self.offset_noise_roll = torch.randn(
            (self.robot.num_envs, ), 
            dtype=torch.float32, 
            device=self.cfg.device
        )

        self.offset_noise_pitch = torch.randn(
            (self.robot.num_envs, ), 
            dtype=torch.float32, 
            device=self.cfg.device
        )

        self.offset_noise_yaw = torch.randn(
            (self.robot.num_envs, ), 
            dtype=torch.float32, 
            device=self.cfg.device
        )

        
        if self.cfg.update_frequency_max > 1:
            self.update_frequency = torch.randint(self.cfg.update_frequency_min, self.cfg.update_frequency_max, (self.robot.num_envs, ), device=self.cfg.device)
            self.update_frequency_offset = torch.randint(0, self.cfg.update_frequency_max, (self.robot.num_envs, ), device=self.cfg.device)

        if self.cfg.max_delay > 0:
            self.delay = torch.randint(0, self.cfg.max_delay, (self.robot.num_envs, ), device=self.cfg.device)
            self.history = HistoryHandler(self.robot.num_envs, {"body_pos": self.cfg.max_delay, "body_quat": self.cfg.max_delay}, {"body_pos": (3,), "body_quat": (4,)}, self.cfg.device)
        else:
            self.delay = torch.zeros(self.robot.num_envs, device=self.cfg.device)
            self.history = None
            
    @abstractmethod
    def _init_ray_pattern(self):
        """Initialize the ray pattern specific to the sensor type."""
        pass
    
    def reset(self, env_ids: torch.Tensor = ...):
        """Reset the sensor buffers for the given environments.
        
        Args:
            env_ids: Environment IDs to reset. If ... (Ellipsis), resets all environments.
        """
        if self.history is not None:
            self.history.reset(env_ids)
        
        if self.cfg.update_frequency_max > 1:
            self.update_frequency[env_ids] = torch.randint(self.cfg.update_frequency_min, self.cfg.update_frequency_max, (len(env_ids), ), device=self.cfg.device)
            self.update_frequency_offset[env_ids] = torch.randint(0, self.cfg.update_frequency_max, (len(env_ids), ), device=self.cfg.device)
        
    def update_buffers(self, episode_step: int, env_ids: torch.Tensor = ...):
        """Update the sensor buffers for the given environments.
        
        Args:
            env_ids: Environment IDs to update. If ... (Ellipsis), updates all environments.
        """

        # Get body pose
        body_idx = self.robot.body_names.index(self.cfg.body_name)
        if self.history is None:
            body_pos = self.robot.rigid_body_pos[env_ids][:, body_idx]
            body_quat = self.robot.rigid_body_quat[env_ids][:, body_idx]
        else:
            self.history.add("body_pos", self.robot.rigid_body_pos[env_ids][:, body_idx])
            self.history.add("body_quat", self.robot.rigid_body_quat[env_ids][:, body_idx])
            # body_pos = self.robot.rigid_body_pos[env_ids][:, body_idx]
            # body_quat = self.robot.rigid_body_quat[env_ids][:, body_idx]
            body_pos = self.history.query_at_history(self.delay, "body_pos")
            body_quat = self.history.query_at_history(self.delay, "body_quat")

            if self.cfg.update_frequency_max > 0 and hasattr(self, 'last_body_pos'):
                update = (torch.remainder(episode_step + self.update_frequency_offset, self.update_frequency) == 0) | (episode_step < 2)

                body_pos = torch.where(update.unsqueeze(-1), body_pos, self.last_body_pos)
                body_quat = torch.where(update.unsqueeze(-1), body_quat, self.last_body_quat)

        # Calculate batch size
        batch_size = len(env_ids) if env_ids is not ... else self.robot.num_envs
        
        # Determine the quaternion to apply for rotation
        quat_to_apply = calc_heading_quat(body_quat) if self.cfg.only_heading else body_quat

        # --- Start Noise Injection ---
        # Apply orientation noise if configured (specifically check for HeightfieldSensorCfg)
        if isinstance(self.cfg, HeightfieldSensorCfg):
            roll_noise = self.cfg.roll_noise_scale
            pitch_noise = self.cfg.pitch_noise_scale
            yaw_noise = self.cfg.yaw_noise_scale

            if roll_noise > 0 or pitch_noise > 0 or yaw_noise > 0:
                # Generate random roll, pitch, yaw angles for the batch
                roll = self.offset_noise_roll[env_ids] * roll_noise
                pitch = self.offset_noise_pitch[env_ids] * pitch_noise
                yaw = self.offset_noise_yaw[env_ids] * yaw_noise

                # Convert Euler angles to quaternion
                euler_angles = torch.stack([roll, pitch, yaw], dim=-1)
                noise_quat = euler_xyz_to_quat(euler_angles) # Shape: [batch_size, 4]

                # Apply noise: Multiply the noise quaternion with the original rotation
                quat_to_apply = quat_mul(noise_quat, quat_to_apply)
        # --- End Noise Injection ---

        # Transform rays to world frame
        if len(self.ray_starts.shape) == 2:
            # Expand static ray pattern for batch
            ray_starts_w = self.ray_starts.unsqueeze(0).repeat(batch_size, 1, 1)
            ray_directions_w = self.ray_directions.unsqueeze(0).repeat(batch_size, 1, 1)
        elif len(self.ray_starts.shape) == 3:
             # Use per-link ray pattern if already batched (e.g., MultiLinkHeightSensor)
            ray_starts_w = self.ray_starts.repeat(batch_size, 1, 1) # This might need adjustment based on specific sensor logic
            ray_directions_w = self.ray_directions.repeat(batch_size, 1, 1)
        else:
            raise ValueError(f"Unsupported ray_starts shape: {self.ray_starts.shape}")

        # Apply rotation using the potentially noisy quat_to_apply
        # Handle reshaping for 3D ray tensors if needed (although ray_starts/directions are typically 2D here)
        if len(ray_starts_w.shape) == 3:  # [batch_size, num_rays, 3]
            num_rays = ray_starts_w.shape[1]
            
            # Reshape rays to 2D for rotation
            rays_flat = ray_starts_w.reshape(-1, 3)
            dirs_flat = ray_directions_w.reshape(-1, 3)
            
            # Expand quaternions to match each ray
            quat_expanded = quat_to_apply.repeat_interleave(num_rays, dim=0)
            
            # Apply rotation
            rotated_rays = quat_rotate(quat_expanded, rays_flat)
            rotated_dirs = quat_rotate(quat_expanded, dirs_flat)
            
            # Reshape back to 3D
            ray_starts_w = rotated_rays.reshape(batch_size, num_rays, 3)
            ray_directions_w = rotated_dirs.reshape(batch_size, num_rays, 3)
        else:
            # Should not happen if ray_starts are always expanded to 3D first?
            # Keeping for safety, but might indicate an issue if reached.
            ray_starts_w = quat_rotate(quat_to_apply, ray_starts_w)
            ray_directions_w = quat_rotate(quat_to_apply, ray_directions_w)
    
        # Apply body translation
        ray_starts_w += body_pos.unsqueeze(1)
        
        # Store world frame rays
        self.ray_starts_world[env_ids] = ray_starts_w
        self.ray_directions_world[env_ids] = ray_directions_w
        
        # Flatten for raycasting
        ray_starts_w_flat = ray_starts_w.view(-1, 3)
        ray_directions_w_flat = ray_directions_w.view(-1, 3)
        
        # Cast rays
        ray_hits, ray_distances, _, _ = raycast_mesh(
            ray_starts_w_flat,
            ray_directions_w_flat,
            max_dist=self.cfg.max_distance,
            mesh=self.terrain_mesh,
            return_distance=True
        )
        
        # Store hits
        self.ray_hits[env_ids] = ray_hits.view(batch_size, self.num_rays, 3)
        
        # Update depth map
        self._update_depth_map(ray_distances.view(batch_size, self.num_rays), env_ids)

        # TODO -- make this agnostic to the env ids. currently, it is not.
        self.last_body_pos = body_pos
        self.last_body_quat = body_quat
        
    @abstractmethod
    def _update_depth_map(self, distances: torch.Tensor, env_ids: torch.Tensor):
        """Update the depth map from ray distances.
        
        Args:
            distances: Ray distances tensor
            env_ids: Environment IDs being updated
        """
        pass

class HeightfieldSensor(RaycastingSensor):
    """Heightfield sensor that casts rays in a grid pattern."""
    
    def __init__(self, robot, cfg: HeightfieldSensorCfg, terrain_vertices: np.ndarray, terrain_indices: np.ndarray):
        super().__init__(robot, cfg, terrain_vertices, terrain_indices)
        
        # Calculate grid dimensions
        self.grid_width = int(self.cfg.size[0] / self.cfg.resolution) + 1
        self.grid_height = int(self.cfg.size[1] / self.cfg.resolution) + 1
        
        self._init_ray_pattern()
        
    def _init_ray_pattern(self):
        """Initialize grid pattern of rays."""
        self.ray_starts, self.ray_directions = grid_pattern(self.cfg, self.cfg.device)
        self.num_rays = len(self.ray_starts)
        
        # Initialize world frame buffers
        self.ray_starts_world = torch.zeros(self.robot.num_envs, self.num_rays, 3, device=self.cfg.device)
        self.ray_directions_world = torch.zeros_like(self.ray_starts_world)
        self.ray_hits = torch.zeros_like(self.ray_starts_world)
        
        # Initialize heightfield buffer with appropriate data type
        if self.cfg.use_float:
            self.depth_map = torch.zeros(
                (self.robot.num_envs, self.grid_height, self.grid_width), 
                dtype=torch.float32, 
                device=self.cfg.device
            )
        else:
            self.depth_map = torch.zeros(
                (self.robot.num_envs, self.grid_height, self.grid_width), 
                dtype=torch.uint8, 
                device=self.cfg.device
            )
        
        self.offset_noise = torch.randn(
            (self.robot.num_envs, ), 
            dtype=torch.float32, 
            device=self.cfg.device
        )
        
    def _update_depth_map(self, distances: torch.Tensor, env_ids: torch.Tensor):
        """Update heightfield from ray distances.
        
        Args:
            distances: Ray distances tensor
            env_ids: Environment IDs being updated
        """
        if self.cfg.use_float:
            # Store actual distance values
            heights = distances.clone()
        else:
            # Convert distances to heights (0-255 range)
            heights = torch.clamp(distances / self.cfg.max_distance * 255, 0, 255).to(torch.uint8)
        
        # Reshape into heightfield
        self.depth_map[env_ids] = heights.view(-1, self.grid_height, self.grid_width)
        # remove inf and nan
        # unsure if we should replace with max dist or whatever
        filled = torch.where(torch.isinf(self.depth_map[env_ids]) | torch.isnan(self.depth_map[env_ids]), torch.zeros_like(self.depth_map[env_ids]), self.depth_map[env_ids])
        mean_filled = torch.mean(filled)
        self.depth_map[env_ids] = torch.where(torch.isnan(self.depth_map[env_ids]) | torch.isinf(self.depth_map[env_ids]), mean_filled * torch.ones_like(self.depth_map[env_ids]), self.depth_map[env_ids])

        if self.cfg.bad_distance_prob > 0:
            bad_distances = torch.rand_like(self.depth_map[env_ids]) < self.cfg.bad_distance_prob
            self.depth_map[env_ids][bad_distances] = torch.rand_like(self.depth_map[env_ids][bad_distances]) * mean_filled * 2.0

        if self.cfg.white_noise_scale > 0:
            self.depth_map[env_ids] = self.depth_map[env_ids] + torch.randn_like(self.depth_map[env_ids]) * self.cfg.white_noise_scale

        if self.cfg.offset_noise_scale > 0:
            self.depth_map[env_ids] = self.depth_map[env_ids] + self.offset_noise[env_ids].view(-1, 1, 1) * self.cfg.offset_noise_scale
        
class MultiLinkHeightSensor(RaycastingSensor):
    """Sensor for measuring heights of multiple links."""
    
    def __init__(self, robot, cfg: MultiLinkHeightSensorCfg, terrain_vertices: np.ndarray, terrain_indices: np.ndarray):
        super().__init__(robot, cfg, terrain_vertices, terrain_indices)
        
        if cfg.link_names is None or len(cfg.link_names) == 0:
            raise ValueError("link_names must be specified and non-empty")
            
        self.link_names = cfg.link_names
        self._init_ray_pattern()
        
    def _init_ray_pattern(self):
        """Initialize ray pattern for each link."""
        num_links = len(self.link_names)
        
        # Create a single ray in the specified direction for each link
        direction = torch.tensor(self.cfg.direction, device=self.cfg.device, dtype=torch.float32)
        direction = direction / torch.norm(direction)
        
        # Initialize depth buffer (one value per link) with appropriate data type
        self.depth_map = torch.zeros(
            (self.robot.num_envs, num_links), 
            dtype=torch.float32 if self.cfg.use_float else torch.uint8,
            device=self.cfg.device
        )
        
        # Since rays originate from the links, initialize with zeros
        # We'll set them properly in update_buffers
        self.ray_starts = torch.zeros((num_links, 3), device=self.cfg.device)
        self.ray_directions = direction.unsqueeze(0).repeat(num_links, 1)
        self.num_rays = num_links
        
        # Initialize world frame buffers
        self.ray_starts_world = torch.zeros(self.robot.num_envs, self.num_rays, 3, device=self.cfg.device)
        self.ray_directions_world = torch.zeros_like(self.ray_starts_world)
        self.ray_hits = torch.zeros_like(self.ray_starts_world)
    
    def update_buffers(self, episode_step: int, env_ids: torch.Tensor = ...):
        """Update the sensor buffers for the given environments.
        
        Args:
            env_ids: Environment IDs to update. If ... (Ellipsis), updates all environments.
        """

        self.link_indices = [self.robot.body_names.index(name) for name in self.link_names]

        # Calculate batch size
        batch_size = len(env_ids) if env_ids is not ... else self.robot.num_envs
        
        # Get link positions
        ray_starts_w = self.robot.rigid_body_pos[env_ids][:, self.link_indices]
        
        # Set ray directions (already in world frame since we use global -Z)
        direction = torch.tensor(self.cfg.direction, device=self.cfg.device, dtype=torch.float32)
        direction = direction / torch.norm(direction)
        ray_directions_w = direction.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.num_rays, 1)

        # Store world frame rays
        self.ray_starts_world[env_ids] = ray_starts_w
        self.ray_directions_world[env_ids] = ray_directions_w
        
        # Flatten for raycasting
        ray_starts_w_flat = ray_starts_w.view(-1, 3)
        ray_directions_w_flat = ray_directions_w.view(-1, 3)
        
        # Cast rays
        ray_hits, ray_distances, _, _ = raycast_mesh(
            ray_starts_w_flat,
            ray_directions_w_flat,
            max_dist=self.cfg.max_distance,
            mesh=self.terrain_mesh,
            return_distance=True
        )
        
        # Store hits
        self.ray_hits[env_ids] = ray_hits.view(batch_size, self.num_rays, 3)
        
        # Update depth map
        self._update_depth_map(ray_distances.view(batch_size, self.num_rays), env_ids)
    
    def _update_depth_map(self, distances: torch.Tensor, env_ids: torch.Tensor):
        """Update depth map from ray distances.
        
        Args:
            distances: Ray distances tensor [batch_size, num_links]
            env_ids: Environment IDs being updated
        """
        # Raycast can return NaN/Inf for misses or numerical edge-cases.
        # Keep observations finite by mapping invalid values to sensor max range.
        cleaned = torch.nan_to_num(
            distances,
            nan=self.cfg.max_distance,
            posinf=self.cfg.max_distance,
            neginf=0.0,
        )

        if self.cfg.use_float:
            self.depth_map[env_ids] = torch.clamp(cleaned, 0.0, self.cfg.max_distance)
        else:
            heights = torch.clamp(cleaned / self.cfg.max_distance * 255, 0, 255).to(torch.uint8)
            self.depth_map[env_ids] = heights

class DepthCameraSensor(RaycastingSensor):
    """Depth camera sensor that casts rays in a pinhole camera pattern."""
    
    def __init__(self, robot, cfg: DepthCameraSensorCfg, terrain_vertices: np.ndarray, terrain_indices: np.ndarray):
        super().__init__(robot, cfg, terrain_vertices, terrain_indices)
        
        # Get intrinsic matrix
        intrinsic_matrix = (cfg.intrinsic_matrix if cfg.intrinsic_matrix is not None 
                           else cfg.default_intrinsic_matrix)
        self.intrinsic_matrix = intrinsic_matrix.reshape(3, 3).unsqueeze(0).to(cfg.device)
        
        # Calculate camera parameters
        fy = self.intrinsic_matrix[0, 1, 1].item()
        self.fov = 2 * np.arctan2(cfg.height/2, fy)
        self.aspect = cfg.width / cfg.height
        
        self._init_ray_pattern()
        
    def _init_ray_pattern(self):
        """Initialize pinhole camera pattern of rays."""
        self.ray_starts, self.ray_directions = pinhole_camera_pattern(
            self.cfg, self.intrinsic_matrix, self.cfg.device
        )
        self.num_rays = len(self.ray_starts) if len(self.ray_starts.shape) == 2 else self.ray_starts.shape[1]
        
        # Initialize world frame buffers
        self.ray_starts_world = torch.zeros(self.robot.num_envs, self.num_rays, 3, device=self.cfg.device)
        self.ray_directions_world = torch.zeros_like(self.ray_starts_world)
        self.ray_hits = torch.zeros_like(self.ray_starts_world)
        
        # Initialize depth image buffer
        self.depth_map = torch.zeros(
            (self.robot.num_envs, self.cfg.height, self.cfg.width), 
            dtype=torch.float32, 
            device=self.cfg.device
        )
        
    def _update_depth_map(self, distances: torch.Tensor, env_ids: torch.Tensor):
        """Update depth image from ray distances.
        
        Args:
            distances: Ray distances tensor
            env_ids: Environment IDs being updated
        """
        # Convert distances to depth values (0-255 range)
        depths = torch.clamp(distances / self.cfg.max_distance * 255, 0, 255)
        
        # Reshape into image
        self.depth_map[env_ids] = depths.view(-1, self.cfg.height, self.cfg.width) 