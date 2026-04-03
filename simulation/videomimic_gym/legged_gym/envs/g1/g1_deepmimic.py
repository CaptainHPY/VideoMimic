from legged_gym.envs.base.robot_deepmimic import RobotDeepMimic
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.g1.g1_env import G1Robot
from typing import Tuple, List
import glob
import os
import shutil
import yaml
import torch
import time # For periodic logging

class G1DeepMimic(RobotDeepMimic, G1Robot):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        # Store default patterns, these might be overridden by YAML
        self.default_human_video_data_pattern = cfg.deepmimic.human_video_data_pattern
        self.default_human_video_terrain_pattern = cfg.deepmimic.human_video_terrain_pattern
    
        # Initialize lists for teacher checkpoint mapping
        self.teacher_checkpoints: List[str] = []
        # self.terrain_to_checkpoint_idx: List[int] = [] # Maps global terrain/motion index to teacher_checkpoints index

        # self.human_video_data_pattern = 'env_0_retarget_poses_g1_fit.h5'l
        # # self.data_pattern = 'retarget_poses_g1_fit.h5'

        # Initialize RobotDeepMimic first, which creates self.replay_data_loader
        RobotDeepMimic.__init__(self, cfg, sim_params, physics_engine, sim_device, headless)

        # --- Success Rate Tracking Initialization ---
        self.success_history_length = 1000
        # Get the original list of paths used by the replay loader
        self.original_replay_data_paths = self.replay_data_loader.get_pkl_paths()
        self.num_clips = len(self.original_replay_data_paths) # Total number of clips (can have duplicates)

        # Find unique paths and create mappings
        self.unique_clip_paths = sorted(list(dict.fromkeys(self.original_replay_data_paths)))
        self.num_unique_clips = len(self.unique_clip_paths)

        if self.num_unique_clips > 0:
            self.clip_path_to_unique_idx = {path: i for i, path in enumerate(self.unique_clip_paths)}

            # Map original index (from replay_data_loader episode_indices, 0 to num_clips-1) to unique index (0 to num_unique_clips-1)
            self.original_idx_to_unique_idx = torch.tensor(
                [self.clip_path_to_unique_idx[path] for path in self.original_replay_data_paths],
                dtype=torch.long,
                device=self.device
            )

            # History buffer: Stores 1 for success (timeout), 0 for failure (termination)
            self.clip_success_history = torch.full(
                (self.num_unique_clips, self.success_history_length),
                fill_value=-1, # Use -1 to indicate no data yet
                dtype=torch.long,
                device=self.device
            )
            # Pointer for circular buffer insertion
            self.clip_history_ptr = torch.zeros(self.num_unique_clips, dtype=torch.long, device=self.device)
            # Count of rollouts recorded for each clip (up to history_length)
            self.clip_rollout_count = torch.zeros(self.num_unique_clips, dtype=torch.long, device=self.device)
        else:
            print("Warning: No unique clips found for success rate tracking.")
            # Initialize with empty/dummy tensors to avoid errors later
            self.clip_path_to_unique_idx = {}
            self.original_idx_to_unique_idx = torch.empty((0,), dtype=torch.long, device=self.device)
            self.clip_success_history = torch.empty((0, self.success_history_length), dtype=torch.long, device=self.device)
            self.clip_history_ptr = torch.empty((0,), dtype=torch.long, device=self.device)
            self.clip_rollout_count = torch.empty((0,), dtype=torch.long, device=self.device)

        self.log_success_rate_interval = 100 # Log every 100 steps
        self.last_log_step = 0
        # Interval for updating adaptive weights (can be same or different from logging)
        self.adaptive_weight_update_interval = cfg.deepmimic.adaptive_weight_update_interval
        self.last_weight_update_step = 0
        # ----------------------------------------------

        # --- Clip Distribution Tracking Initialization ---
        # History buffer: Stores counts of envs per unique clip at each step
        if self.num_unique_clips > 0:
            self.step_clip_distribution_history = torch.zeros(
                (self.success_history_length, self.num_unique_clips),
                dtype=torch.long,
                device=self.device
            )
            # Pointer for circular buffer insertion
            self.step_dist_history_ptr = torch.zeros((), dtype=torch.long, device=self.device)
            # Count of steps recorded in the distribution history
            self.step_dist_rollout_count = torch.zeros((), dtype=torch.long, device=self.device)
        else:
            # Initialize with empty/dummy tensors if no clips
            self.step_clip_distribution_history = torch.empty((self.success_history_length, 0), dtype=torch.long, device=self.device)
            self.step_dist_history_ptr = torch.zeros((), dtype=torch.long, device=self.device)
            self.step_dist_rollout_count = torch.zeros((), dtype=torch.long, device=self.device)
        # ----------------------------------------------

    def get_replay_terrain_path(self, cfg: LeggedRobotCfg) -> Tuple[List[str], List[str]]:
        """
        Get the replay data paths and terrain paths for the given config.
        Handles loading from AMASS, a single human video folder, or a YAML list of folders.
        Also populates the teacher checkpoint mapping (`self.teacher_checkpoints` and `self.terrain_to_checkpoint_idx`).
        """
        # Use local lists during generation, assign to self at the end
        replay_data_paths = []
        terrain_paths = []
        local_teacher_checkpoints = [] # Stores unique names encountered
        local_terrain_to_checkpoint_idx = [] # Stores mapping for paths generated in this call
        local_data_fps_override = []

        # Helper to get/add checkpoint index
        def _get_checkpoint_idx(name):
            if name and isinstance(name, str) and name.strip():
                name = name.strip()
                if name not in local_teacher_checkpoints:
                    local_teacher_checkpoints.append(name)
                return local_teacher_checkpoints.index(name)
            return -1 # Default to -1 if no valid name provided

        # corresponds to a neighbouring folder of unitree_rl_gym
        # data_root = os.path.join(LEGGED_GYM_ROOT_DIR, '../retargeted_data')
        # data_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'demo_data/output_postprocessed')
        data_root = os.path.join(LEGGED_GYM_ROOT_DIR, cfg.deepmimic.data_root)
        # TODO -- merge these this is pretty bad
        alt_data_root = os.path.join(LEGGED_GYM_ROOT_DIR, cfg.deepmimic.alt_data_root)

        amass_data_root = os.path.join(LEGGED_GYM_ROOT_DIR, cfg.deepmimic.amass_data_root)

        # --- AMASS Data --- #
        if hasattr(cfg.deepmimic, 'use_amass') and cfg.deepmimic.use_amass:
            amass_teacher_name = cfg.deepmimic.amass_teacher_checkpoint_run_name
            amass_ckpt_idx = _get_checkpoint_idx(amass_teacher_name)

            search_path = os.path.join(amass_data_root, cfg.deepmimic.amass_replay_data_path)
            amass_replay_data_paths_init = glob.glob(search_path)
            
            if not amass_replay_data_paths_init:
                print(f"Warning: No AMASS replay data paths found at {search_path}")
            else:
                for replay_data_path in amass_replay_data_paths_init:
                    for difficulty in range(cfg.deepmimic.amass_terrain_difficulty):
                        terrain_name = "flat" if difficulty == 0 else f"rough_d{difficulty}"
                        terrain_path = f'{LEGGED_GYM_ROOT_DIR}/resources/motions/amass/ground_mesh_{terrain_name}.obj'
                        terrain_paths.append(terrain_path)
                        replay_data_paths.append(replay_data_path)
                        local_data_fps_override.append(None)
                        local_terrain_to_checkpoint_idx.append(amass_ckpt_idx) # Map this terrain/motion to the AMASS teacher
        
        # --- Human Video Data --- #
        if hasattr(cfg.deepmimic, 'use_human_videos') and cfg.deepmimic.use_human_videos:
            source = cfg.deepmimic.human_motion_source
            if isinstance(source, str) and source.lower().endswith('.h5'):
                if not getattr(cfg.deepmimic, 'allow_h5_direct_source', False):
                    raise ValueError(
                        "Direct .h5 human_motion_source is disabled. "
                        "Set --env.deepmimic.allow_h5_direct_source=True to enable it."
                    )

                source_path = source if os.path.isabs(source) else os.path.join(LEGGED_GYM_ROOT_DIR, source)
                replay_h5_paths = sorted(glob.glob(source_path)) if any(ch in source_path for ch in ['*', '?', '[']) else [source_path]
                replay_h5_paths = [p for p in replay_h5_paths if p.lower().endswith('.h5')]

                if not replay_h5_paths:
                    raise ValueError(
                        "No .h5 files found for human_motion_source. "
                        f"Resolved pattern/path: {source_path}"
                    )

                shared_flat_terrain_token = '__shared_flat_terrain__'
                single_h5_ckpt_idx = -1
                for replay_h5_path in replay_h5_paths:
                    for _ in range(cfg.deepmimic.human_video_oversample_factor):
                        replay_data_paths.append(replay_h5_path)
                        terrain_paths.append(shared_flat_terrain_token)
                        local_data_fps_override.append(None)
                        local_terrain_to_checkpoint_idx.append(single_h5_ckpt_idx)

            elif source.lower().endswith('.yaml'):
                # Load from YAML file
                yaml_path = os.path.join(LEGGED_GYM_ROOT_DIR, source)
                try:
                    with open(yaml_path, 'r') as f:
                        motion_list = yaml.safe_load(f)
                    if not isinstance(motion_list, list):
                        raise ValueError(f"YAML file {yaml_path} does not contain a list.")

                    for item in motion_list:
                        folder_path_rel = item.get('folder_path')
                        # split for legacy files
                        folder_path_rel = folder_path_rel.split('/')[-1]
                        # folder_path_rel = f'megahunter_megasam_reconstruction_results_{folder_path_rel}_cam01_frame_0_300_subsample_1'
                        if not folder_path_rel:
                            print(f"Warning: Skipping item in {yaml_path} due to missing 'folder_path'. Item: {item}")
                            continue

                        # Get teacher checkpoint name for this specific clip from YAML
                        human_teacher_name = item.get('teacher_checkpoint_run_name', None)
                        human_ckpt_idx = _get_checkpoint_idx(human_teacher_name)

                        # Use specific patterns from YAML or default from config
                        data_pattern = item.get('human_video_data_pattern', self.default_human_video_data_pattern)
                        terrain_pattern = item.get('human_video_terrain_pattern', self.default_human_video_terrain_pattern)

                        # Find matching folder in data_root using partial matching
                        folder_path_abs = None
                        for dirpath in os.listdir(data_root):
                            if folder_path_rel in dirpath:
                                folder_path_abs = os.path.join(data_root, dirpath)
                                break
                        if folder_path_abs is None:
                            for dirpath in os.listdir(alt_data_root):
                                if folder_path_rel in dirpath:
                                    folder_path_abs = os.path.join(alt_data_root, dirpath)
                                    break

                        
                        # If no matching folder found, use the original concatenation
                        if folder_path_abs is None:
                            folder_path_abs = os.path.join(data_root, folder_path_rel)
                            print(f"Warning: No partial match found for '{folder_path_rel}' in {data_root}, using direct path.")

                        source_replay_path = os.path.join(folder_path_abs, data_pattern)
                        source_terrain_path = os.path.join(folder_path_abs, terrain_pattern)

                        # # NOTE: The following code assumes that 'final_replay_path' and 'final_terrain_path'
                        # # have been defined immediately before this loop. This definition should include
                        # # logic to potentially copy files to a target directory as per the instructions.
                        # # Example logic to place *before* this selection:
                        # target_base_dir = "/home/arthur/Desktop/demo_data_backup" # TODO: Specify target folder (e.g., "/path/to/copied_data")
                        # if target_base_dir:
                        #     relative_folder_name = os.path.basename(folder_path_abs)
                        #     target_folder_path = os.path.join(target_base_dir, relative_folder_name)
                        #     os.makedirs(target_folder_path, exist_ok=True)
                        #     dest_replay_path = os.path.join(target_folder_path, data_pattern)
                        #     dest_terrain_path = os.path.join(target_folder_path, terrain_pattern)

                        #     # Copy files (consider adding checks if files exist or overwrite behavior)
                        #     if os.path.exists(source_replay_path):
                        #             shutil.copy2(source_replay_path, dest_replay_path)
                        #     else:
                        #             print(f"Warning: Source file not found, cannot copy: {source_replay_path}")
                        #     if os.path.exists(source_terrain_path):
                        #             shutil.copy2(source_terrain_path, dest_terrain_path)
                        #     else:
                        #             print(f"Warning: Source file not found, cannot copy: {source_terrain_path}")

                        # The loop now uses the prepared paths (either original or copied destination)
                        for _ in range(cfg.deepmimic.human_video_oversample_factor):
                            replay_data_paths.append(source_replay_path) # Use the potentially copied path
                            terrain_paths.append(source_terrain_path)   # Use the potentially copied path
                            # import pdb; pdb.set_trace() # Removed debugger
                            local_data_fps_override.append(item.get('default_data_fps_override', None))
                            local_terrain_to_checkpoint_idx.append(human_ckpt_idx) # Map this terrain/motion to its specific teacher

                except FileNotFoundError:
                    print(f"Error: Human motion YAML file not found at {yaml_path}")
                except yaml.YAMLError as e:
                    print(f"Error parsing YAML file {yaml_path}: {e}")
                except Exception as e:
                    print(f"Error processing YAML file {yaml_path}: {e}")

            elif isinstance(source, str) and source: # Treat as single folder name
                folder_path_rel = source
                
                # Find matching folder in data_root using partial matching
                folder_path_abs = None
                for dirpath in os.listdir(data_root):
                    if folder_path_rel in dirpath:
                        folder_path_abs = os.path.join(data_root, dirpath)
                        break
                
                # If no matching folder found, use the original concatenation
                if folder_path_abs is None:
                    folder_path_abs = os.path.join(data_root, folder_path_rel)
                    print(f"Warning: No partial match found for '{folder_path_rel}' in {data_root}, using direct path.")

                # Use default patterns from config
                data_pattern = self.default_human_video_data_pattern
                terrain_pattern = self.default_human_video_terrain_pattern

                # No specific teacher defined for single folder mode, use default -1
                single_folder_ckpt_idx = -1

                for _ in range(cfg.deepmimic.human_video_oversample_factor):
                    replay_data_paths.append(os.path.join(folder_path_abs, data_pattern))
                    terrain_paths.append(os.path.join(folder_path_abs, terrain_pattern))
                    local_terrain_to_checkpoint_idx.append(single_folder_ckpt_idx)
            else:
                print(f"Warning: Invalid human_motion_source format: {source}. Expected YAML path, folder name, or .h5 path/glob.")

        if not replay_data_paths:
            amass_search_path = os.path.join(amass_data_root, cfg.deepmimic.amass_replay_data_path)
            raise ValueError(
                "No replay data paths were loaded. "
                f"AMASS search path: {amass_search_path}. "
                "Check --env.deepmimic.amass_replay_data_path and data root settings."
            )

        # Assign the generated lists to instance variables
        self.teacher_checkpoints = local_teacher_checkpoints
        self.terrain_to_checkpoint_idx = torch.tensor(local_terrain_to_checkpoint_idx, device=self.device)

        # Ensure mapping length matches path lengths
        if len(self.terrain_to_checkpoint_idx) != len(replay_data_paths):
             print(f"Warning: Mismatch in length between terrain_to_checkpoint_idx ({len(self.terrain_to_checkpoint_idx)}) and replay_data_paths ({len(replay_data_paths)}). This should not happen.")

        return replay_data_paths, terrain_paths, local_data_fps_override

    def get_available_episodes(self) -> List[str]:
        """
        Returns a list of available episode names/paths that can be used for visualization.
        Adapts based on whether AMASS or human video (single or YAML) is used.
        """
        # data_root = os.path.join(LEGGED_GYM_ROOT_DIR, '../retargeted_data')
        amass_data_root = os.path.join(LEGGED_GYM_ROOT_DIR, self.cfg.deepmimic.amass_data_root)
        available_episodes = []

        # --- AMASS Episodes --- #
        if hasattr(self.cfg.deepmimic, 'use_amass') and self.cfg.deepmimic.use_amass:
            search_path = os.path.join(amass_data_root, self.cfg.deepmimic.amass_replay_data_path)
            amass_episodes = glob.glob(search_path)
            amass_episodes_base = [os.path.splitext(os.path.basename(ep))[0] for ep in amass_episodes]
            available_episodes.extend([f'{ep}_d{difficulty}' for ep in amass_episodes_base for difficulty in range(self.cfg.deepmimic.amass_terrain_difficulty)])

        # --- Human Video Episodes --- #
        if hasattr(self.cfg.deepmimic, 'use_human_videos') and self.cfg.deepmimic.use_human_videos:
            source = self.cfg.deepmimic.human_motion_source
            if isinstance(source, str) and source.lower().endswith('.h5'):
                source_path = source if os.path.isabs(source) else os.path.join(LEGGED_GYM_ROOT_DIR, source)
                replay_h5_paths = sorted(glob.glob(source_path)) if any(ch in source_path for ch in ['*', '?', '[']) else [source_path]
                available_episodes.extend([os.path.splitext(os.path.basename(p))[0] for p in replay_h5_paths])
            elif source.lower().endswith('.yaml'):
                yaml_path = os.path.join(LEGGED_GYM_ROOT_DIR, source)
                try:
                    with open(yaml_path, 'r') as f:
                        motion_list = yaml.safe_load(f)
                    if isinstance(motion_list, list):
                        for item in motion_list:
                            folder_path_rel = item.get('folder_path')
                            if folder_path_rel:
                                folder_path_rel = folder_path_rel.split('/')[-1]
                                # Use the relative folder path as the episode identifier
                                available_episodes.append(folder_path_rel)
                except Exception as e:
                    print(f"Warning: Could not load or parse YAML {yaml_path} for available episodes: {e}")
            elif isinstance(source, str) and source:
                # Use the folder name directly as the episode identifier
                available_episodes.append(source)

        return available_episodes

    
    def _obs_teacher_checkpoint_index(self):

        row_indices = self.replay_data_loader.episode_indices
        checkpoint_indices = self.terrain_to_checkpoint_idx[row_indices]
        # print(f'checkpoint_indices: {checkpoint_indices}')
        return checkpoint_indices

    def check_termination(self):
        """ Checks if environments need to be reset and updates success rate history."""
        # Call parent class method first to determine reset_buf and time_out_buf
        super().check_termination()

        # --- Update Success Rate History ---
        if self.num_unique_clips > 0:
            # Find environments that are resetting in this step
            reset_env_ids = torch.where(self.reset_buf)[0]

            if len(reset_env_ids) > 0:
                # Determine success (1 if timeout, 0 otherwise)
                # Note: time_out_buf is True only if it's the primary reason for reset.
                # If reset_buf is True due to other reasons (fall, etc.), time_out_buf might be False even if max steps reached.
                # We consider success *only* if the episode timed out.
                is_success = self.time_out_buf[reset_env_ids].long()

                # Get the original clip indices for the resetting environments
                # Need the indices *before* reset_idx potentially changes them
                original_clip_indices = self.replay_data_loader.episode_indices[reset_env_ids]

                # Map to unique clip indices
                unique_clip_indices = self.original_idx_to_unique_idx[original_clip_indices]

                # Get current history pointers for these unique clips
                history_pointers = self.clip_history_ptr[unique_clip_indices]

                # Update history buffer at the pointer locations
                self.clip_success_history[unique_clip_indices, history_pointers] = is_success

                # Increment pointers (circularly)
                self.clip_history_ptr[unique_clip_indices] = (history_pointers + 1) % self.success_history_length

                # Increment rollout counts (clamped at history_length)
                current_counts = self.clip_rollout_count[unique_clip_indices]
                new_counts = torch.min(
                    current_counts + 1,
                    torch.tensor(self.success_history_length, device=self.device, dtype=torch.long)
                )
                self.clip_rollout_count[unique_clip_indices] = new_counts
        # -----------------------------------

    def _compute_and_log_success_rates(self):
        """ Calculates and logs the success rate for each clip. """
        if self.num_unique_clips == 0:
            # Return NaN success rates if no clips
            current_success_rates = torch.full((0,), float('nan'), device=self.device, dtype=torch.float32)
            return current_success_rates

        overall_rollout_count = 0
        overall_success_count = 0
        # Tensor to store success rates for each unique clip
        current_success_rates = torch.full((self.num_unique_clips,), float('nan'), device=self.device, dtype=torch.float32)

        for i in range(self.num_unique_clips):
            count = self.clip_rollout_count[i].item()
            clip_path = "_".join(self.unique_clip_paths[i].split("/")[-2:])
            overall_rollout_count += count
            overall_success_count += torch.sum(self.clip_success_history[i, :count]).item()

            if count > 0:
                # Get the valid history entries (ignore -1 placeholders)
                history = self.clip_success_history[i, :count]
                valid_history = history[history != -1] # Filter out initial -1 values
                valid_count = len(valid_history)

                if valid_count > 0:
                    success_rate = torch.mean(valid_history.float()).item()
                    self.extras["episode"][f"success/clip_{clip_path}"] = success_rate
                    current_success_rates[i] = success_rate # Store for adaptive weighting
                else:
                    self.extras["episode"][f"success/clip_{clip_path}"] = 0.0
                    # Keep NaN for current_success_rates if no valid history
            else:
                self.extras["episode"][f"success/clip_{clip_path}"] = 0.0
                # Keep NaN for current_success_rates if no history

        # Calculate overall success rate (only based on clips with history)
        valid_overall_mask = ~torch.isnan(current_success_rates)
        if valid_overall_mask.any():
            # Weighted average based on rollout count for each valid clip
            valid_counts = self.clip_rollout_count[valid_overall_mask]
            valid_successes = torch.sum(self.clip_success_history[valid_overall_mask, :], dim=1)
            # Filter successes further for only valid entries (-1)
            for idx, unique_idx in enumerate(torch.where(valid_overall_mask)[0]):
                history = self.clip_success_history[unique_idx, :self.clip_rollout_count[unique_idx]]
                valid_history = history[history != -1]
                valid_successes[idx] = valid_history.sum()
                valid_counts[idx] = len(valid_history)
            
            overall_success_count = valid_successes.sum().item()
            overall_rollout_count = valid_counts.sum().item()
            overall_success_rate = overall_success_count / (overall_rollout_count + 1e-6)
        else:
            overall_success_rate = 0.0
        
        self.extras["episode"]["success/overall"] = overall_success_rate

        return current_success_rates # Return per-unique-clip rates

    def compute_observations(self):
        # --- Record Clip Distribution --- 
        if self.num_unique_clips > 0:
            # Get current original clip indices for all environments
            original_clip_indices = self.replay_data_loader.episode_indices

            # Map to unique clip indices
            unique_clip_indices = self.original_idx_to_unique_idx[original_clip_indices]

            # Count occurrences of each unique index across all environments
            current_step_distribution = torch.bincount(
                unique_clip_indices, 
                minlength=self.num_unique_clips
            ).long() # Ensure result is long tensor

            # Store distribution in history buffer
            ptr = self.step_dist_history_ptr.item()
            self.step_clip_distribution_history[ptr] = current_step_distribution

            # Increment pointer (circularly)
            self.step_dist_history_ptr = (self.step_dist_history_ptr + 1) % self.success_history_length

            # Increment step count (clamped at history_length)
            self.step_dist_rollout_count = torch.min(
                self.step_dist_rollout_count + 1,
                torch.tensor(self.success_history_length, device=self.device, dtype=torch.long)
            )
        # ----------------------------- 

        # Log success rates periodically
        current_step = self.gym.get_frame_count(self.sim) # Using frame count as a proxy for steps
        if current_step >= self.last_log_step + self.log_success_rate_interval:
            # Compute success rates (also needed for potential weight update)
            current_unique_success_rates = self._compute_and_log_success_rates()
            self._compute_and_log_clip_distribution()
            self.last_log_step = current_step

            # Update adaptive weights if strategy is active and interval passed
            if self.cfg.deepmimic.clip_weighting_strategy == 'success_rate_adaptive' and self.num_unique_clips > 0:
                 # Map unique success rates back to the full list of original clips
                success_rates_full = torch.full((self.num_clips,), float('nan'), device=self.device, dtype=torch.float32)
                # Use the mapping: original_idx -> unique_idx -> success_rate
                valid_unique_mask = ~torch.isnan(current_unique_success_rates)
                valid_unique_indices = torch.where(valid_unique_mask)[0]
                
                # Create a mask for original indices that map to valid unique indices
                original_indices_with_valid_rates_mask = torch.zeros(self.num_clips, dtype=torch.bool, device=self.device)
                for unique_idx in valid_unique_indices:
                     original_indices_with_valid_rates_mask |= (self.original_idx_to_unique_idx == unique_idx)
                
                # Apply the rates
                valid_original_indices = torch.where(original_indices_with_valid_rates_mask)[0]
                if len(valid_original_indices) > 0:
                    unique_map_for_valid_originals = self.original_idx_to_unique_idx[valid_original_indices]
                    success_rates_full[valid_original_indices] = current_unique_success_rates[unique_map_for_valid_originals]

                # Update weights in ReplayDataLoader
                self.replay_data_loader.update_adaptive_weights(success_rates_full)
                # print(f"Step {current_step}: Updated adaptive weights.") # Optional: for debugging
                self.last_weight_update_step = current_step # Reset timer after update

        return super().compute_observations()

    def _compute_and_log_clip_distribution(self):
        """ Calculates and logs the distribution of steps spent on each clip over the history. """
        if self.num_unique_clips == 0 or self.step_dist_rollout_count == 0:
            return

        num_recorded_steps = self.step_dist_rollout_count.item()

        # Sum distributions over the recorded history
        # Need to handle the circular buffer correctly if it hasn't filled up yet
        if num_recorded_steps < self.success_history_length:
            summed_dist = torch.sum(self.step_clip_distribution_history[:num_recorded_steps], dim=0)
        else:
            summed_dist = torch.sum(self.step_clip_distribution_history, dim=0)
        
        total_steps_in_history = summed_dist.sum().float()
        
        if total_steps_in_history > 0:
            clip_percentages = (summed_dist.float() / total_steps_in_history)
            for i in range(self.num_unique_clips):
                clip_path = "_".join(self.unique_clip_paths[i].split("/")[-2:]) # Use same naming as success rate
                percentage = clip_percentages[i].item()
                self.extras["episode"][f"dist/clip_{clip_path}"] = percentage
        else:
            # Handle case where no steps recorded yet (shouldn't happen if count > 0)
            for i in range(self.num_unique_clips):
                clip_path = "_".join(self.unique_clip_paths[i].split("/")[-2:])
                self.extras["episode"][f"dist/clip_{clip_path}"] = 0.0


