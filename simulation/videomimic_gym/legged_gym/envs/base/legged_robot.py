from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os

from legged_gym.tensor_utils.torch_jit_utils import  *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.math import wrap_to_pi
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg
from legged_gym.utils.raycaster import DepthCameraSensorCfg, DepthCameraSensor, HeightfieldSensorCfg, HeightfieldSensor
from legged_gym.utils.history import HistoryHandler

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self.use_viser_viz = hasattr(self.cfg, 'viser') and self.cfg.viser.enable  # Flag to enable/disable viser visualization
        
        # Initialize viser if enabled
        if self.use_viser_viz:
            from legged_gym.utils.viser_visualizer import LeggedRobotViser

            import viser
            from viser.extras import ViserUrdf
            from robot_descriptions.loaders.yourdfpy import load_robot_description
            self.viser_viz = LeggedRobotViser(urdf_path=self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), dt=self.cfg.control.decimation * self.sim_params.dt)
            self.viser_viz.init_isaacgym_robot(self)
        
        # Initialize trajectory export variables
        self.export_trajectory = self.cfg.env.export_trajectory
        if self.export_trajectory:
            import os
            self.export_dir = self.cfg.env.export_dir
            os.makedirs(self.export_dir, exist_ok=True)
            
            # Initialize trajectory data for each environment
            self.trajectory_data = [{
                'joint_names': [],
                'joints': [],
                'root_quat': [],
                'root_pos': [],
                'link_names': [],
                'link_pos': [],
                'link_quat': [],
                'contacts': {},
                'trajectory_name': None,  # Will be set for deepmimic environments
                'joint_targets': [], 
                'obs': {}
            } for _ in range(self.cfg.env.num_envs)]
            
            # Track which environments have completed their first episode
            self.env_episode_done = torch.zeros(self.cfg.env.num_envs, dtype=torch.bool, device=sim_device)
            self.num_envs_completed = 0

            self.stepped_before_export = False
        
        # Validate sensor configuration vs requested observations
        self._validate_sensor_observations()
            
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_viewer_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._prepare_reward_function()
        self._prepare_observation_function()
        self._init_buffers()
        # self.compute_observations()
        # self.obs_dict = self.get_observations()
        # self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)

        self.init_done = True

        # Initialize history handler as None - will be lazily initialized
        self.history_handler = None

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        actions = actions + self.actions_offset_seed * self.cfg.noise.offset_scales.action

        self.actions_pre_clip = actions

        clip_actions = self.cfg.normalization.clip_actions
        actions_clipped = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        beta = self.cfg.control.beta
        self.actions = beta * actions_clipped + (1-beta) * self.actions

        if self.cfg.domain_rand.action_delays:
            self.action_queue.add('actions', self.actions)
            self.actions = self.action_queue.query_at_history(self.action_delay_idx, 'actions')

        # Export trajectory data if enabled (only for environments that haven't completed)
        if self.export_trajectory and not torch.all(self.env_episode_done):
            if not self.stepped_before_export:
                # there's a done on the first step, so we need to skip the export on the very first step after the sim boots up
                self.stepped_before_export = True
            else:
                self._export_trajectory_step()

        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            if not self.cfg.control.control_type == 'POS':
                self.torques = self._compute_torques(self.actions).view(self.torques.shape)
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            else:
                self.dof_pos_targets = self._compute_dof_pos_targets(self.actions).view(self.dof_pos_targets.shape)
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_pos_targets))
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time-elapsed_time>0:
                    time.sleep(sim_time-elapsed_time)
            
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations

        for key in self.obs_dict.keys():
            # Only clip if the tensor's dtype is floating-point
            if self.obs_dict[key].is_floating_point():
                self.obs_dict[key] = torch.clip(self.obs_dict[key], -clip_obs, clip_obs)
        
        if self.use_dict_obs:
            return self.obs_dict, self.rew_buf, self.reset_buf, self.extras
        else:
            return self.obs_buf, self.privileged_obs_buf
        # return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        bad_root_vel_mask = ~torch.isfinite(self.root_states[:, 7:13]).all(dim=1)
        if torch.any(bad_root_vel_mask):
            bad_env_ids = torch.where(bad_root_vel_mask)[0]
            print(f"[LeggedRobot] Non-finite root velocities for envs: {bad_env_ids.tolist()}, zeroing root velocities")
            self.root_states[bad_env_ids, 7:13] = 0.0
            bad_env_ids_int32 = bad_env_ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_states),
                                                         gymtorch.unwrap_tensor(bad_env_ids_int32), len(bad_env_ids_int32))
        self.nonfinite_root_vel_buf = bad_root_vel_mask

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        if not torch.isfinite(self.base_ang_vel[:]).all():
            print(f"root_states[:, 3:7]: {torch.isfinite(self.root_states[:, 3:7]).all()}")
            print(f"root_states[:, 10:13]: {torch.isfinite(self.root_states[:, 10:13]).all()}")
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # Update all sensors
        for sensor_name, sensor in self.sensors.items():
            sensor.update_buffers(episode_step=self.episode_length_buf, env_ids=...)

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.reset_buf |= self.nonfinite_root_vel_buf
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        
        if self.cfg.domain_rand.push_robots or (self.use_viser_viz and self.viser_viz.enable_push_robots):
            self._push_robots()

        self.compute_observations()

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        self.update_visualization()

    def update_visualization(self):
        """Update visualization elements like ray visualizations, contact forces, etc."""
        # Update viser visualization if enabled
        if self.use_viser_viz:
            self.viser_viz.update_from_torch(self.root_states, self.dof_pos, env_idx=0)
            self.viser_viz.update_contact_force_visualization(
                self.rigid_body_pos[0],  # Only visualize env 0
                self.contact_forces[0]   # Only visualize env 0
            )
            # Call the visualization update explicitly
            self.viser_viz.update_visualization()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        self._resample_episodic_randomisations(env_ids)

        # reset buffers
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['Episode/rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.commands.curriculum:
            self.extras["episode"]["Episode/max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
        if self.history_handler is not None:
            self.history_handler.reset(env_ids)
        
        for sensor_name, sensor in self.sensors.items():
            sensor.reset(env_ids)
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            self.current_reward_value[name] = rew
        
        label = "total_pre_clip" if self.cfg.rewards.only_positive_rewards else "total"
        self.current_reward_value[label] = self.rew_buf.clone()

        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
            self.current_reward_value["total"] = self.rew_buf

        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
            self.current_reward_value["termination"] = rew
    
    def compute_observations(self):
        """ Computes observations
        """
        for i, obs_fn in enumerate(self.observation_functions):
            self.obs_dict[self.observation_names[i]] = obs_fn()

        # Initialize history handler if not already initialized
        if self.history_handler is None and hasattr(self.cfg.env, 'obs_history'):
            from legged_gym.utils.history import HistoryHandler
            obs_dims = {k: v.shape[1:] for k, v in self.obs_dict.items()}
            self.history_handler = HistoryHandler(self.num_envs, self.cfg.env.obs_history, obs_dims, self.device)

        # Update history if handler exists
        if self.history_handler is not None:
            for key in self.cfg.env.obs_history.keys():
                self.history_handler.add(key, self.obs_dict[key])

            # Add history observations to the observation dictionary
            for key in self.cfg.env.obs_history.keys():
                self.obs_dict[f'history_{key}'] = self.history_handler.query(key)

        # Expose per-env replay clip index for debugging/logging (not used by policy input heads).
        if hasattr(self, 'replay_data_loader') and hasattr(self.replay_data_loader, 'episode_indices'):
            self.obs_dict['clip_index'] = self.replay_data_loader.episode_indices.unsqueeze(1)

        self.obs_dict['teacher'] = self._manual_obs_teacher()
        return self.obs_dict
    
    def _obs_torso(self):
        return torch.cat((#  self.base_lin_vel * self.obs_scales.lin_vel,
                     self.base_ang_vel  * self.obs_scales.ang_vel,
                     self.projected_gravity,
                     self.commands[:, :3] * self.commands_scale,
                     (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                     self.dof_vel * self.obs_scales.dof_vel,
                     self.actions
                     ),dim=-1)
    
    def _obs_sensor(self, sensor_name):
        """Generic observation method for any sensor
        
        Args:
            sensor_name (str): Name of the sensor
            
        Returns:
            Normalized sensor data as a tensor
        """
        if sensor_name not in self.sensors:
            raise ValueError(f"Sensor '{sensor_name}' not initialized but observation requested")
        
        sensor = self.sensors[sensor_name]
        
        # For uint8 depth maps (camera and legacy heightfields), normalize to 0-1
        if sensor.depth_map.dtype == torch.uint8:
            return sensor.depth_map.float() / 255.0
        
        # For float tensors, return directly
        return sensor.depth_map

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        if self.cfg.terrain.mesh_type == 'trimesh':
            assert hasattr(self, 'terrain')
            self._create_trimesh()
        else:
            self._create_ground_plane()

        
        # Initialize sensor variables
        self.sensors = {}
        
        # Create and initialize sensors based on config
        self._init_sensors()
        
        self._create_envs()

    def set_viewer_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
            
        
        if hasattr(self.cfg.asset, 'dont_collide_groups'):
            dont_collide_groups = self.cfg.asset.dont_collide_groups

            for i, shape_idx in enumerate(self.rigid_shape_indices):
                start = shape_idx.start
                count = shape_idx.count
                if count > 0:
                    for j in range(start, start + count):

                        for collision_group in dont_collide_groups:
                            if self.body_names[i] in dont_collide_groups[collision_group]:
                                props[j].filter = 1 << collision_group
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if self.cfg.domain_rand.randomize_dof_friction:
            # set all dofs together -- separately is worse
            max_value = self.cfg.domain_rand.max_dof_friction
            # bucketing cuz isaac gym needs it
            value = np.random.uniform(0.0, max_value) 
            value = (round((value / max_value) * self.cfg.domain_rand.dof_friction_buckets) / self.cfg.domain_rand.dof_friction_buckets) * max_value
            # props['friction'][:] = np.random.uniform(0.0, 0.1)
            props['friction'][:] = value
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        # if self.cfg.domain_rand.randomize_base_mass:
        #     rng = self.cfg.domain_rand.added_mass_range
        #     props[0].mass += np.random.uniform(rng[0], rng[1])
        
                # No need to use tensors as only called upon env creation
            
        self.torso_index = self.body_names.index('torso_link')
        # import pdb; pdb.set_trace()
        if self.cfg.domain_rand.randomize_base_mass:
            rng_mass = self.cfg.domain_rand.added_mass_range
            rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1,))
            props[self.torso_index].mass += rand_mass
        else:
            rand_mass = np.zeros((1,))

        if self.cfg.domain_rand.randomize_base_com:
            rng_com = self.cfg.domain_rand.added_com_range
            rand_com = np.random.uniform(rng_com[0], rng_com[1], size=(3,))
            # rand_com = [-0.1, 0.1, 0.0]
            props[self.torso_index].com += gymapi.Vec3(*rand_com)
        else:
            rand_com = np.zeros(3)
        mass_params = np.concatenate([rand_mass, rand_com])

        return props, mass_params

        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)
        
        if hasattr(self, 'viser_viz') and self.viser_viz.manual_control.value:
            self.commands[:, :3] = 0.
            if self.viser_viz.move_forward.value:
                self.commands[:, 0] = 1.0
            if self.viser_viz.move_back.value:
                self.commands[:, 0] = -1.0
            if self.viser_viz.move_left.value:
                self.commands[:, 1] = 1.0
            if self.viser_viz.move_right.value:
                self.commands[:, 1] = -1.0
            if self.viser_viz.rotate_left.value:
                self.commands[:, 2] = 1.0
            if self.viser_viz.rotate_right.value:
                self.commands[:, 2] = -1.0


    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    
    def _init_randomisation_buffers(self):
        """Initialize buffers for episodic randomisations
        """
        self.enable_torque_rfi = self.cfg.domain_rand.torque_rfi_rand
        self.torque_rfi_rand_scale = self.cfg.domain_rand.torque_rfi_rand_scale
        self.torque_rfi_seed = torch.randn_like(self.dof_pos)
        self.p_gain_rand = self.cfg.domain_rand.p_gain_rand
        self.p_gain_rand_scale = self.cfg.domain_rand.p_gain_rand_scale
        self.d_gain_rand = self.cfg.domain_rand.d_gain_rand
        self.d_gain_rand_scale = self.cfg.domain_rand.d_gain_rand_scale
        self.p_gain_rand_seed = torch.randn_like(self.dof_pos)
        self.d_gain_rand_seed = torch.randn_like(self.dof_pos)
        self.actions_offset_seed = torch.zeros_like(self.dof_pos)

        if self.cfg.domain_rand.control_delays:
            self.control_queue = HistoryHandler(self.num_envs,
                                    {'controls': self.cfg.domain_rand.control_delay_max+1},
                                    {'controls': (self.num_dof,)},
                                    self.device
            )
            self.control_delay_idx = torch.randint(self.cfg.domain_rand.control_delay_min, 
                                                self.cfg.domain_rand.control_delay_max+1, (self.num_envs,), device=self.device, requires_grad=False)
        
        if self.cfg.domain_rand.action_delays:
            self.action_queue = HistoryHandler(self.num_envs,
                                    {'actions': self.cfg.domain_rand.action_delay_max+1},
                                    {'actions': (self.num_dof,)},
                                    self.device
            )


    def _resample_episodic_randomisations(self, env_ids):
        """ Resample episodic randomisations
        """
        self.torque_rfi_seed[env_ids] = torch.randn_like(self.dof_pos[env_ids])
        self.p_gain_rand_seed[env_ids] = torch.randn_like(self.dof_pos[env_ids])
        self.d_gain_rand_seed[env_ids] = torch.randn_like(self.dof_pos[env_ids])
        self.actions_offset_seed[env_ids] = torch.randn_like(self.dof_pos[env_ids])

        if self.cfg.domain_rand.control_delays:
            self.control_delay_idx[env_ids] = torch.randint(self.cfg.domain_rand.control_delay_min, 
                                                    self.cfg.domain_rand.control_delay_max+1, (len(env_ids),), device=self.device, requires_grad=False)
            self.control_queue.reset(env_ids)

        if self.cfg.domain_rand.action_delays:
            self.action_delay_idx[env_ids] = torch.randint(self.cfg.domain_rand.action_delay_min, 
                                                    self.cfg.domain_rand.action_delay_max+1, (len(env_ids),), device=self.device, requires_grad=False)
            self.action_queue.reset(env_ids)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """

        if self.cfg.domain_rand.control_delays:
            self.control_queue.add('controls', actions)
            actions = self.control_queue.query_at_history(self.control_delay_idx, 'controls')

        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type

        if self.p_gain_rand:
            p_gains = self.p_gains * (1 + self.p_gain_rand_seed * self.p_gain_rand_scale)
        else:
            p_gains = self.p_gains
        if self.d_gain_rand:
            d_gains = self.d_gains * (1 + self.d_gain_rand_seed * self.d_gain_rand_scale)
        else:
            d_gains = self.d_gains


        if control_type=="P":
            torques = p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        elif control_type=="DIRECT":
            torques = self.p_gains*(actions - self.dof_pos) - self.d_gains*self.dof_vel
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        torque_randomisation_scale = self.torque_rfi_rand_scale
        if self.enable_torque_rfi:
            torques += self.torque_rfi_seed * self.torque_limits * torque_randomisation_scale

        final_torques = torch.clip(torques, -self.torque_limits, self.torque_limits)

        # if self.cfg.domain_rand.control_delays:
        #     self.control_queue.add('controls', final_torques)
        #     final_torques = self.control_queue.query_at_history(self.control_delay_idx, 'controls')

        return final_torques
    
    def _compute_dof_pos_targets(self, actions):
        """Compute joint position targets from actions"""
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type

        if control_type == "P":
            return actions_scaled + self.default_dof_pos
        elif control_type == "DIRECT":
            return actions
        elif control_type == "DEEPMIMIC_DELTA":
            return actions_scaled + self.target_motors
        else:
            return None  # For other control types (V, T), no position targets

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        # Check if pushing is enabled in viser (if viser is being used)
        if self.use_viser_viz and not self.viser_viz.enable_push_robots.value:
            return

        env_ids = torch.arange(self.num_envs, device=self.device)
        push_env_ids = env_ids[self.episode_length_buf[env_ids] % int(self.cfg.domain_rand.push_interval) == 0]
        if len(push_env_ids) == 0:
            return

        # Get the base max velocity and apply the scale factor if viser is enabled
        max_vel_xy = self.cfg.domain_rand.max_push_vel_xy
        max_vel_z = 0.5  # Default to small vertical push
        if self.use_viser_viz:
            max_vel_xy *= self.viser_viz.push_force_scale.value
            max_vel_z = self.viser_viz.push_force_z_scale.value

        # Apply random pushes in XY plane
        self.root_states[:, 7:9] = torch_rand_float(-max_vel_xy, max_vel_xy, (self.num_envs, 2), device=self.device)  # lin vel x/y
        # Apply random pushes in Z direction
        self.root_states[:, 9:10] = torch_rand_float(-max_vel_z, max_vel_z, (self.num_envs, 1), device=self.device)  # lin vel z
        
        env_ids_int32 = push_env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self.root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

   
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:12+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[12+self.num_actions:12+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[12+2*self.num_actions:12+3*self.num_actions] = 0. # previous actions

        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_pos = self.rigid_body_states.view(self.num_envs, -1, 13)[..., 0:3]
        self.rigid_body_vel = self.rigid_body_states.view(self.num_envs, -1, 13)[..., 7:10]
        self.rigid_body_quat = self.rigid_body_states.view(self.num_envs, -1, 13)[..., 3:7]
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))

        if self.cfg.control.control_type == 'POS':
            self.dof_pos_targets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        else:
            self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_contacts_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.feet_air_max_height = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self._init_randomisation_buffers()
      

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}
        self.current_reward_value = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                                    for name in self.reward_scales.keys()}

    def _prepare_observation_function(self):
        """Prepares a list of observation functions, which will be called to compute the observations."""
        self.observation_functions = []
        self.observation_names = []
        
        for name in self.cfg.env.obs:
            # Check if this is a sensor observation (if the name exists in sensors dictionary)
            if name in self.sensors:
                # Use the generic sensor observation method with the sensor name
                self.observation_functions.append(lambda sensor_name=name: self._obs_sensor(sensor_name))
                self.observation_names.append(name)
            
            else:
                # Use the named observation method
                func_name = f'_obs_{name}'
                if hasattr(self, func_name):
                    self.observation_functions.append(getattr(self, func_name))
                else:
                    if name == 'teacher':
                        self.observation_functions.append(lambda: torch.zeros(self.num_envs, 415, device=self.device))
                    else:
                        raise ValueError(f"Missing observation method {func_name} for observation {name}")
            
                self.observation_names.append(name)

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation and viser visualization """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

        if self.use_viser_viz:
            self.viser_viz.add_ground_plane()


    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation and viser visualization """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = 0.0
        tm_params.transform.p.y = 0.0
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        
        self.gym.add_triangle_mesh(
            self.sim,
            self.terrain.vertices.flatten(order='C'),
            self.terrain.triangles.flatten(order='C'),
            tm_params
        )

        # Add terrain mesh to viser if enabled
        if self.use_viser_viz:
            self.viser_viz.add_mesh(
                "/terrain",
                vertices=self.terrain.vertices,
                faces=self.terrain.triangles,
                color=(0.282, 0.247, 0.361),
            )

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)

        self.dof_names = self.gym.get_asset_dof_names(robot_asset)

        for i in range(self.num_dof):
            dof_props_asset['effort'][i] *= 1
            dof_props_asset['velocity'][i] *= 1
            # dof_props_asset['friction'][i] = 0.1
            # dof_props_asset['friction'][i] = 0.0

            if self.cfg.control.control_type == 'POS':
                print(f'setting stiffness and damping for {self.dof_names[i]}')

                found = False
                for dof_name in self.cfg.control.stiffness.keys():
                    if dof_name in self.dof_names[i]:
                        dof_props_asset['stiffness'][i] = self.cfg.control.stiffness[dof_name]
                        dof_props_asset['damping'][i] = self.cfg.control.damping[dof_name]
                        found = True
                        break
                if not found:
                    raise ValueError(f'stiffness and damping for {dof_name} not found')

            dof_props_asset['driveMode'][i] = gymapi.DOF_MODE_POS if self.cfg.asset.default_dof_drive_mode == 'POS' else gymapi.DOF_MODE_EFFORT

        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)

        self.rigid_shape_indices = self.gym.get_asset_rigid_body_shape_indices(robot_asset)

        self.body_names = body_names
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        self.feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            # self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, rigid_shape_props)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(self.feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
      
        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
     

        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)


    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.root_states[:, 2]
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_energy(self):
        # Penalize energy
        return torch.sum(torch.square(self.torques * self.dof_vel), dim=1)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_action_accel(self):
        # penalise changes in the change in actions ( acceleration 
        accel = (self.actions - 2*self.last_actions + self.last_last_actions) 
        return torch.sum(torch.square(accel), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        # rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime = torch.sum((self.feet_air_time - 0.25) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    def _init_sensors(self):
        """Initialize sensors based on the configuration.
        
        This method creates sensors based on the configurations in cfg.sensors.sensor_cfgs.
        To use sensors in your robot:
        
        1. In your robot config class, define sensor configurations:
            ```python
            sensors = LeggedRobotSensorsCfg(
                sensor_cfgs = [
                    DepthCameraCfg(name="front_camera", body_name="head"),
                    DepthCameraCfg(name="rear_camera", body_name="torso", width=120, height=80),
                    HeightfieldCfg(name="terrain_height", body_name="pelvis"),
                    MultiLinkHeightCfg(name="link_heights", link_names=["pelvis", "left_foot", "right_foot"]),
                ]
            )
            ```
            
        2. Add the sensor names to the obs list:
            ```python
            env = LeggedRobotEnvCfg(
                obs = ['torso', 'front_camera', 'rear_camera', 'terrain_height', 'link_heights'],
                ...
            )
            ```
            
        3. The observation methods will be automatically handled
        """

        if not hasattr(self, 'terrain'):
            assert len(self.cfg.sensors.sensor_cfgs) == 0, "No terrain found, but sensors are configured"
            return

        # Check if we have a terrain for the sensors
        terrain_vertices = getattr(self.terrain, 'vertices', None)
        terrain_triangles = getattr(self.terrain, 'triangles', None)

        # Create sensors from configurations
        for sensor_index, sensor_cfg in enumerate(self.cfg.sensors.sensor_cfgs):
            if not sensor_cfg.enabled:
                continue
                
            # If name is not provided, use the type name (and add index if multiple of same type)
            if not sensor_cfg.name:
                same_type_count = sum(1 for i, s in enumerate(self.cfg.sensors.sensor_cfgs) 
                                  if i < sensor_index and s.type == sensor_cfg.type and s.enabled)
                sensor_cfg.name = f"{sensor_cfg.type}_{same_type_count}" if same_type_count > 0 else sensor_cfg.type
            
            # Create the appropriate sensor based on type
            if sensor_cfg.type == 'depth_camera':
                self._init_depth_camera(sensor_cfg, terrain_vertices, terrain_triangles)
                
            elif sensor_cfg.type == 'heightfield':
                self._init_heightfield(sensor_cfg, terrain_vertices, terrain_triangles)
                
            elif sensor_cfg.type == 'multi_link_height':
                self._init_multi_link_height(sensor_cfg, terrain_vertices, terrain_triangles)

    def _init_depth_camera(self, sensor_cfg, terrain_vertices, terrain_triangles):
        """Initialize a depth camera sensor"""
        from legged_gym.utils.raycaster.sensors import DepthCameraSensor
        from legged_gym.utils.raycaster.sensor_cfg import DepthCameraSensorCfg
        
        # Create depth camera configuration
        camera_device_cfg = DepthCameraSensorCfg(
            device=self.sim_device,
            width=sensor_cfg.width,
            height=sensor_cfg.height,
            downsample_factor=sensor_cfg.downsample_factor,
            body_name=sensor_cfg.body_name,
            max_distance=sensor_cfg.max_distance,
            only_heading=sensor_cfg.only_heading,
            intrinsic_matrix=sensor_cfg.intrinsic_matrix
        )
        
        # Create the depth camera sensor
        self.sensors[sensor_cfg.name] = DepthCameraSensor(
            self, camera_device_cfg, terrain_vertices, terrain_triangles
        )

    def _init_heightfield(self, sensor_cfg, terrain_vertices, terrain_triangles):
        """Initialize a heightfield sensor"""
        from legged_gym.utils.raycaster.sensors import HeightfieldSensor
        from legged_gym.utils.raycaster.sensor_cfg import HeightfieldSensorCfg
        
        # todo -- why tf do we recreate this twice. Thanks, claude :( )
        # Create heightfield configuration
        sensor_args = {k: v for k, v in sensor_cfg.to_dict().items() if k in HeightfieldSensorCfg(device=None).__dict__ and k is not 'type'}
        heightfield_device_cfg = HeightfieldSensorCfg(
            device=self.sim_device,
            **sensor_args
            # size=sensor_cfg.size,
            # resolution=sensor_cfg.resolution,
            # body_name=sensor_cfg.body_name,
            # max_distance=sensor_cfg.max_distance,
            # only_heading=sensor_cfg.only_heading,
            # use_float=sensor_cfg.use_float,
            # gaussian_noise_scale=sensor_cfg.gaussian_noise_scale
        )
        
        # Create the heightfield sensor
        self.sensors[sensor_cfg.name] = HeightfieldSensor(
            self, heightfield_device_cfg, terrain_vertices, terrain_triangles
        )

    def _init_multi_link_height(self, sensor_cfg, terrain_vertices, terrain_triangles):
        """Initialize a multi-link height sensor"""
        from legged_gym.utils.raycaster.sensors import MultiLinkHeightSensor
        from legged_gym.utils.raycaster.sensor_cfg import MultiLinkHeightSensorCfg
        
        # Create multi-link height sensor configuration
        multi_link_device_cfg = MultiLinkHeightSensorCfg(
            device=self.sim_device,
            body_name=sensor_cfg.body_name,
            max_distance=sensor_cfg.max_distance,
            only_heading=sensor_cfg.only_heading,
            link_names=sensor_cfg.link_names,
            use_float=sensor_cfg.use_float
        )
        
        # Create the multi-link height sensor
        self.sensors[sensor_cfg.name] = MultiLinkHeightSensor(
            self, multi_link_device_cfg, terrain_vertices, terrain_triangles
        )

    def _validate_sensor_observations(self):
        """Validate that all sensor observations have corresponding enabled sensors"""
        if not hasattr(self.cfg, 'sensors'):
            return
            
        # Get all configured sensor names
        sensor_names = []
        for sensor_cfg in self.cfg.sensors.sensor_cfgs:
            if sensor_cfg.enabled:
                sensor_names.append(sensor_cfg.name or sensor_cfg.type)
        
        sensor_names = set(sensor_names)
        
        # Check if any observation is a sensor name but not in the enabled sensors
        for obs in self.cfg.env.obs:
            if obs not in sensor_names and obs in {'depth_camera', 'heightfield'} or obs.startswith(('depth_camera_', 'heightfield_')):
                print(f"Warning: Observation '{obs}' appears to be a sensor but is not configured in sensors.sensor_cfgs")
                print(f"         This will cause an error when computing observations.")

    def get_sensor(self, sensor_name):
        """Get a sensor by name.
        
        Args:
            sensor_name (str): Name of the sensor to retrieve
            
        Returns:
            The sensor object if found, None otherwise
            
        Example:
            ```python
            # Get the depth camera sensor
            depth_camera = robot.get_sensor('depth_camera')
            if depth_camera:
                # Use the sensor
                depth_map = depth_camera.depth_map
            ```
        """
        return self.sensors.get(sensor_name, None)
        
    def get_sensor_data(self, sensor_name):
        """Get sensor data for a specific sensor.
        
        Args:
            sensor_name (str): Name of the sensor
            
        Returns:
            The sensor's data (usually a depth_map tensor) if found, None otherwise
            
        Example:
            ```python
            # Get depth map from the depth camera
            depth_map = robot.get_sensor_data('depth_camera')
            if depth_map is not None:
                # Use the depth map
                print(f"Depth map shape: {depth_map.shape}")
            ```
        """
        sensor = self.get_sensor(sensor_name)
        if sensor:
            return sensor.depth_map
        return None
    
    @property
    def env_root_pos(self):
        return self.root_states[:, 0:3]
    
    @property
    def env_rigid_body_pos(self):
        return self.rigid_body_pos

    def _export_trajectory_step(self):
        """Export trajectory data for all environments that haven't completed their first episode"""
        # Process each environment that hasn't completed its first episode
        if self.episode_length_buf[0] % 100 == 0:
            print(f'Working on export...')
        for env_id in range(self.cfg.env.num_envs):
            if self.env_episode_done[env_id]:
                continue

            # Check if episode is done for this environment
            if self.reset_buf[env_id]:
                self._save_trajectory_data(env_id)
                self.env_episode_done[env_id] = True
                self.num_envs_completed += 1
                
                # Print progress
                print(f"Saved trajectory for environment {env_id} ({self.num_envs_completed}/{self.cfg.env.num_envs} completed)")
                
                # Check if all environments are done
                if self.num_envs_completed == self.cfg.env.num_envs:
                    print(f"\nAll {self.cfg.env.num_envs} environments have completed their first episode!")
                    print(f"Trajectory data saved in: {self.export_dir}/")
            else:
                
                # If this is the first step for this environment, initialize names
                if len(self.trajectory_data[env_id]['joint_names']) == 0:
                    self.trajectory_data[env_id]['joint_names'] = self.dof_names
                    self.trajectory_data[env_id]['link_names'] = self.body_names
                    self.trajectory_data[env_id]['joint_targets'] = []  # Add joint targets list
                    self.trajectory_data[env_id]['actions'] = []

                    # Initialize contact dictionaries for each foot
                    if hasattr(self, 'feet_indices'):
                        # Check if we have G1-style foot links (left_ankle_roll_link and right_ankle_roll_link)
                        foot_names = [self.body_names[idx] for idx in self.feet_indices]
                        use_standardized_names = len(foot_names) == 2 and \
                            'left_ankle_roll_link' in foot_names and \
                            'right_ankle_roll_link' in foot_names
                        
                        if use_standardized_names:
                            # Use standardized foot names for G1
                            self.trajectory_data[env_id]['contacts']['left_foot'] = []
                            self.trajectory_data[env_id]['contacts']['right_foot'] = []
                        else:
                            # Use actual link names for other robots
                            for foot_idx in range(len(self.feet_indices)):
                                foot_name = self.body_names[self.feet_indices[foot_idx]]
                                self.trajectory_data[env_id]['contacts'][foot_name] = []

                    # For deepmimic environments, store the trajectory name
                    if hasattr(self, 'replay_data_loader'):
                        clip_index = self.replay_data_loader.episode_indices[env_id].item()
                        trajectory_path = self.replay_data_loader.pkl_paths[clip_index]
                        trajectory_name = os.path.splitext(os.path.basename(trajectory_path))[0]
                        self.trajectory_data[env_id]['trajectory_name'] = trajectory_name

                # Compute joint position targets
                joint_targets = self._compute_dof_pos_targets(self.actions[env_id:env_id+1])
                if joint_targets is not None:
                    self.trajectory_data[env_id]['joint_targets'].append(joint_targets[0].cpu().numpy())

                # Append current step data
                self.trajectory_data[env_id]['joints'].append(self.dof_pos[env_id].cpu().numpy())
                self.trajectory_data[env_id]['root_quat'].append(self.base_quat[env_id].cpu().numpy())
                self.trajectory_data[env_id]['root_pos'].append(self.env_root_pos[env_id].cpu().numpy())
                self.trajectory_data[env_id]['actions'].append(self.actions[env_id].cpu().numpy())
                if len(self.trajectory_data[env_id]['obs']) == 0:
                    self.trajectory_data[env_id]['obs'] = {k: [] for k in self.obs_dict.keys()}
                for key, value in self.obs_dict.items():
                    self.trajectory_data[env_id]['obs'][key].append(value[env_id].cpu().numpy())
                
                # Get link positions and orientations
                link_pos = []
                link_quat = []
                for i in range(len(self.body_names)):
                    pos = self.env_rigid_body_pos[env_id, i].cpu().numpy()
                    quat = self.rigid_body_quat[env_id, i].cpu().numpy()
                    link_pos.append(pos)
                    link_quat.append(quat)
                
                self.trajectory_data[env_id]['link_pos'].append(link_pos)
                self.trajectory_data[env_id]['link_quat'].append(link_quat)

                # Record foot contacts
                if hasattr(self, 'feet_indices'):
                    # Check if we have G1-style foot links
                    foot_names = [self.body_names[idx] for idx in self.feet_indices]
                    use_standardized_names = len(foot_names) == 2 and \
                        'left_ankle_roll_link' in foot_names and \
                        'right_ankle_roll_link' in foot_names

                    if use_standardized_names:
                        # Use standardized names for G1
                        left_idx = foot_names.index('left_ankle_roll_link')
                        right_idx = foot_names.index('right_ankle_roll_link')
                        left_contact = bool(torch.norm(self.contact_forces[env_id, self.feet_indices[left_idx], :]) > 1.0)
                        right_contact = bool(torch.norm(self.contact_forces[env_id, self.feet_indices[right_idx], :]) > 1.0)
                        self.trajectory_data[env_id]['contacts']['left_foot'].append(left_contact)
                        self.trajectory_data[env_id]['contacts']['right_foot'].append(right_contact)
                    else:
                        # Use actual link names for other robots
                        for foot_idx in range(len(self.feet_indices)):
                            foot_name = self.body_names[self.feet_indices[foot_idx]]
                            contact = bool(torch.norm(self.contact_forces[env_id, self.feet_indices[foot_idx], :]) > 1.0)
                            self.trajectory_data[env_id]['contacts'][foot_name].append(contact)

    def _save_trajectory_data(self, env_id):
        """Save the collected trajectory data for a specific environment"""
        import pickle
        import numpy as np
        import os
        
        # Convert lists to numpy arrays
        export_data = {
            'joint_names': self.trajectory_data[env_id]['joint_names'],
            'joints': np.array(self.trajectory_data[env_id]['joints']),
            'joint_targets': np.array(self.trajectory_data[env_id]['joint_targets']) if len(self.trajectory_data[env_id]['joint_targets']) > 0 else None,
            'root_quat': np.array(self.trajectory_data[env_id]['root_quat']),
            'root_pos': np.array(self.trajectory_data[env_id]['root_pos']),
            'link_names': self.trajectory_data[env_id]['link_names'],
            'link_pos': np.array(self.trajectory_data[env_id]['link_pos']),
            'link_quat': np.array(self.trajectory_data[env_id]['link_quat']),
            'actions': np.array(self.trajectory_data[env_id]['actions']),
            'contacts': {name: np.array(contacts) for name, contacts in self.trajectory_data[env_id]['contacts'].items()},
            'stiffness': {name: float(self.p_gains[i].cpu().numpy()) for i, name in enumerate(self.dof_names)},
            'damping': {name: float(self.d_gains[i].cpu().numpy()) for i, name in enumerate(self.dof_names)},
            'obs': {key: np.array(value) for key, value in self.trajectory_data[env_id]['obs'].items()}
        }

        # Include trajectory name if it exists (for deepmimic)
        if self.trajectory_data[env_id]['trajectory_name'] is not None:
            export_data['trajectory_name'] = self.trajectory_data[env_id]['trajectory_name']
            filename = f"env_{env_id}_{self.trajectory_data[env_id]['trajectory_name']}.pkl"
        else:
            filename = f"env_{env_id}.pkl"

        # Save to file
        filepath = os.path.join(self.export_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(export_data, f)
        
        print(f"Trajectory data saved to {filepath}")
