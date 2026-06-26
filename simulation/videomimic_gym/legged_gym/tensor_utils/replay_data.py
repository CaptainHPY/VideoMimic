import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import pickle
import os
import re
from typing import Union, List, Optional
from tqdm import tqdm
import omegaconf
from legged_gym.tensor_utils.torch_jit_utils import *

# Process-local cache for clip index -> source file path, used by debugging tools.
LATEST_CLIP_PATHS = []

# mocap_xia filenames encode style/content/clip indices: style_content_clip.h5
MOCAP_XIA_CLIP_NAME_RE = re.compile(r'^(?P<style>[A-Za-z]+)_(?P<content>\d+)_(?P<clip>\d+)$')

# TODO document the file format :)

default_joint_orders = {
    "g1": [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint", 
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint"
    ],
}

class ReplayState:
    def __init__(self, root_pos=None, root_quat=None, root_vel=None, root_ang_vel=None,
                 dofs=None, dof_vels=None, motors=None, motor_vels=None,
                 link_pos=None, link_quat=None, link_vels=None, link_ang_vels=None, contacts=None, clip_index=None,
                 extra_link_pos=None, extra_link_quat=None, extra_link_vels=None, extra_link_ang_vels=None):
        """Replay state for a certain number of environments.

        Args:
            root_pos: Global frame positions (num_envs, 7)
            root_quat: Global frame quaternions (num_envs, 4)
            root_vel: Global frame velocities (num_envs, 6)
            root_ang_vel: Global frame angular velocities (num_envs, 3)
            dofs: DOF positions (num_envs, num_dofs)
            dof_vels: DOF velocities (num_envs, num_dofs)
            motors: Motor positions (num_envs, num_motors)
            motor_vels: Motor velocities (num_envs, num_motors)
            link_pos: Local frame link positions (relative to root) (num_envs, num_links, 3)
            link_quat: Local frame link quaternions (relative to root) (num_envs, num_links, 4)
            link_vels: Local frame link velocities (relative to root) (num_envs, num_links, 3)
            link_ang_vels: Local frame link angular velocities (relative to root) (num_envs, num_links, 3)
            contacts: Whether the foot contacts are active (num_envs, num_contacts)
            clip_index: Index of the clip in the replay data (1,)
            extra_link_pos: Local frame link positions (relative to root) (num_envs, num_extra_links, 3)
            extra_link_quat: Local frame link quaternions (relative to root) (num_envs, num_extra_links, 4)
            extra_link_vels: Local frame link velocities (relative to root) (num_envs, num_extra_links, 3)
            extra_link_ang_vels: Local frame link angular velocities (relative to root) (num_envs, num_extra_links, 3)
        """
        self.root_pos = root_pos
        self.root_quat = root_quat
        self.root_vel = root_vel
        self.root_ang_vel = root_ang_vel
        self.dofs = dofs
        self.dof_vels = dof_vels
        self.motors = motors
        self.motor_vels = motor_vels
        self.link_pos = link_pos
        self.link_quat = link_quat
        self.link_vels = link_vels
        self.link_ang_vels = link_ang_vels
        self.contacts = contacts
        self.clip_index = clip_index
        self.extra_link_pos = extra_link_pos
        self.extra_link_quat = extra_link_quat
        self.extra_link_vels = extra_link_vels
        self.extra_link_ang_vels = extra_link_ang_vels


class ReplayDataLoader:
    def __init__(self,
                 pkl_paths: Union[List[str], str],
                 num_envs: int,
                 device: torch.device,
                 dt: float,
                 dof_names: List[str],
                 motor_names: List[str],
                 link_names: List[str],
                 contact_names: List[str],
                 data_quat_format: str = 'wxyz',
                 randomize_start_offset: bool = False,
                 reference_height: float = None,
                 height_direct_offset: float = None,
                 start_offset: int = None,
                 adjust_root_pos: bool = True,
                 n_prepend: int = 0,
                 n_append: int = 0,
                 extra_link_names: List[str] = None,
                 is_csv_joint_only: bool = False,
                 default_joint_order_type: Optional[str] = None,
                 cut_off_import_length: int = -1,
                 default_data_fps: int = -1,
                 data_fps_override: List[float] = None,
                 upsample_data: bool = False,
                 weighting_strategy: str = 'uniform',
                 inorder_envs: bool = False,
                 clip_weighting_strategy: str = 'uniform_step',
                 min_weight_factor: float = 1.0/3.0,
                 max_weight_factor: float = 3.0,
                 ):
        """
        Args:
            pkl_paths: List of paths to the pkl files containing the replay data or a directory containing pkl files
            num_envs: Number of environments to load the data for
            device: Device to load the data to
            dt: Time step size. Used to compute velocities from positions in the data.
            dof_names: Names of the degrees of freedom to load (will be the order of the columns in the dofs tensor)
            motor_names: Names of the motors to load (will be the order of the columns in the motors tensor)
            link_names: Names of the links to load (will be the order of the columns in the link_pos and link_quat tensors)
            contact_names: Names of the contacts to load (will be the order of the columns in the contacts tensor)
            data_quat_format: Format of the quaternion data (either 'wxyz' or 'xyzw')
            randomize_start_offset: Whether to randomize the starting offset within each episode
            reference_height: Height offset to add to the root position
            height_direct_offset: Direct height offset added to root and link positions
            start_offset: Fixed starting offset within each episode (if randomize_start_offset is False)
            adjust_root_pos: Whether to adjust the root position to make it relative to the first frame and add a height offset
            n_prepend: Number of frames to prepend to the data (copies of the first frame)
            n_append: Number of frames to append to the data (copies of the last frame)
            extra_link_names: list of names of extra links to load (will be the order of the columns in the extra_link_pos and extra_link_quat tensors).
            is_csv_joint_only: Whether the input files are CSV files containing only joint data without link positions
            default_joint_order_type: Type of default joint order to use if no joint order is provided.
            cut_off_import_length: Length of the data to cut off at. If -1, no cutting will be done.
            default_data_fps: Default FPS of the data if not specified in the file
            data_fps_override: List of FPS values to override the default data FPS for each pkl file.
            upsample_data: Whether to upsample ALL data types to match the target FPS (1/dt)
            weighting_strategy: Weighting strategy to use for sampling the start positions *within* episodes if randomize_start_offset is True. Options are "uniform" or "linear".
            inorder_envs: If True, environment idx will correspond to the order of the pkl files. Useful when re-exporting data.
            clip_weighting_strategy: Weighting strategy to use for sampling *across* different clips. Options:
                'uniform_step': Each step across all clips has equal probability.
                'uniform_clip': Each clip has equal total probability, distributed uniformly among its steps.
                'success_rate_adaptive': Each clip's probability is inversely proportional to its success rate (requires calling `update_adaptive_weights`).
            min_weight_factor: Minimum weight multiplier for adaptive weighting (relative to uniform clip weight).
            max_weight_factor: Maximum weight multiplier for adaptive weighting (relative to uniform clip weight).
        Example pkl schema:
        {
            "joint_names": joint_names, # list of joint names
            "joints": np.zeros((timeseries_length, num_joints)), # joint positions (x y z)
            "root_quat": np.zeros((timeseries_length, 4)), # global frame quaternions (x y z w)
            "root_pos": np.zeros((timeseries_length, 3)), # global frame positions (x y z)
            "link_names": body_names, # list of link names
            "link_pos": np.zeros((timeseries_length, len(body_names), 3)), # local frame positions relative to root (x y z)
            "link_quat": np.zeros((timeseries_length, len(body_names), 4)), # local frame quaternions relative to root (x y z w)
            "contacts": { # optional contact info
                "left_foot": np.zeros(timeseries_length, dtype=bool),
                "right_foot": np.zeros(timeseries_length, dtype=bool)
            },
            "fps": 30.,
        }
        """
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.randomize_start_offset = randomize_start_offset
        self.reference_height = reference_height
        self.height_direct_offset = height_direct_offset
        self.data_quat_format = data_quat_format
        self.dof_names = dof_names
        self.motor_names = motor_names
        self.link_names = link_names
        self.start_offset = start_offset
        self.adjust_root_pos = adjust_root_pos
        self.n_prepend = n_prepend
        self.n_append = n_append
        self.contact_names = contact_names
        self.extra_link_names = extra_link_names
        self.is_csv_joint_only = is_csv_joint_only
        self.default_joint_order_type = default_joint_order_type
        self.cut_off_import_length = cut_off_import_length
        self.default_data_fps = default_data_fps
        self.data_fps_override = data_fps_override
        self.upsample_data = upsample_data
        self.target_fps = 1.0 / dt if dt > 0 else default_data_fps
        self.weighting_strategy = weighting_strategy
        self.inorder_envs = inorder_envs
        self.clip_weighting_strategy = clip_weighting_strategy
        self.min_weight_factor = min_weight_factor
        self.max_weight_factor = max_weight_factor

        if inorder_envs:
            assert not self.randomize_start_offset, "Randomizing start offset is not allowed when inorder_envs is True"

        # Process pkl_paths argument
        if not isinstance(pkl_paths, list) and not isinstance(pkl_paths, omegaconf.ListConfig):
            if os.path.isdir(pkl_paths):
                # if directory, get all pkl files in the directory
                pkl_paths = sorted([os.path.join(pkl_paths, f) for f in os.listdir(pkl_paths) if f.endswith(".pkl") or f.endswith(".h5")])
            else:
                # Assume single file path
                pkl_paths = [pkl_paths]
        else:
            pkl_paths = list(pkl_paths)
        
        self._pkl_paths = pkl_paths # Store the original list
        global LATEST_CLIP_PATHS
        LATEST_CLIP_PATHS = [os.path.abspath(str(p)) for p in pkl_paths]

        self._load_data(self._pkl_paths)

        # Initialize tensor index for each environment
        self.env_indices = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        # Initialize episode lengths for each environment
        self.episode_lengths = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        self.episode_indices = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        # For each environment, store the starting index and ending index of the episode
        self.env_start_indices = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.env_end_indices = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        # Keep track of index within the episode for each environment
        self.index_within_episode = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        # For each environment, store the length of the episode clip
        # irrespective of the start offset (so that we can compute phase of a motion)
        self.episode_clip_length = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        # Precompute weights and indices for sampling
        self._precompute_sampling_weights()

    def get_pkl_paths(self) -> List[str]:
        """Returns the original list of pkl/h5 paths used to initialize the loader."""
        return self._pkl_paths

    def _parse_mocap_xia_clip_name(self, pkl_path: str):
        """Parse mocap_xia clip metadata from a filename.

        The mocap_xia h5 dataset uses filenames like `angry_01_000.h5`, where
        the first token is the style, the middle token is the content id, and
        the last token is the clip variant.
        """
        clip_name = os.path.splitext(os.path.basename(str(pkl_path)))[0]
        match = MOCAP_XIA_CLIP_NAME_RE.match(clip_name)
        if match is None:
            return None

        return {
            "clip_name": clip_name,
            "style_name": match.group("style"),
            "content_id": int(match.group("content")),
            "clip_id": int(match.group("clip")),
        }

    def _pick_alternate_clip_index(self, clip_index: int, relation: str = "style") -> int:
        """Pick another clip that matches the requested relation.

        Args:
            clip_index: Index of the reference clip in the loaded clip list.
            relation: `style` picks the same style but a different content id.
                `content` picks the same content id but a different style.

        Returns:
            A different clip index when one exists, otherwise the original index.
        """
        if not hasattr(self, "clip_metadata") or clip_index < 0 or clip_index >= len(self.clip_metadata):
            return clip_index

        clip_meta = self.clip_metadata[clip_index]
        if clip_meta is None:
            return clip_index

        if relation == "style":
            candidate_indices = self.style_to_clip_indices.get(clip_meta["style_name"], [])
            candidate_indices = [idx for idx in candidate_indices if self.clip_metadata[idx]["content_id"] != clip_meta["content_id"]]
        elif relation == "content":
            candidate_indices = self.content_to_clip_indices.get(clip_meta["content_id"], [])
            candidate_indices = [idx for idx in candidate_indices if self.clip_metadata[idx]["style_name"] != clip_meta["style_name"]]
        else:
            raise ValueError(f"Invalid relation: {relation}")

        if not candidate_indices:
            return clip_index

        return candidate_indices[clip_meta["clip_id"] % len(candidate_indices)]

    def pick_paired_clip_index(self, clip_index: int, relation: str = "style") -> int:
        """Public wrapper for selecting a paired clip index.

        Args:
            clip_index: Reference clip index.
            relation: "style" or "content" pairing rule.
        """
        return self._pick_alternate_clip_index(clip_index, relation=relation)

    def get_content_style_clip_splits(self):
        """Return clip paths ordered by content and style groupings.

        The same mocap_xia dataset can be viewed in two ways:
        - content ordering: grouped and sorted by content id
        - style ordering: grouped and sorted by style name

        This mirrors the content/style split used by the stage-3 training code
        in the video pipeline, while keeping the underlying replay loader API
        unchanged.
        """
        if not hasattr(self, "content_sequence_paths"):
            return [], []
        return self.content_sequence_paths, self.style_sequence_paths

    def _compute_angular_velocity(self, q1, q0):
        delta_q = quat_mul(q1, quat_conjugate(q0))
        delta_q = delta_q / delta_q.norm(dim=-1, keepdim=True)
        cos_theta_over_two = delta_q[..., 3]
        sin_theta_over_two = delta_q[..., :3].norm(dim=-1)
        angle = 2 * torch.atan2(sin_theta_over_two, cos_theta_over_two)
        small_angle = sin_theta_over_two < 1e-6
        sin_theta_over_two[small_angle] = 1.0  # prevent division by zero
        axis = delta_q[..., :3] / sin_theta_over_two.unsqueeze(-1)
        omega = (angle / self.dt).unsqueeze(-1) * axis
        omega[small_angle] = 2 * delta_q[small_angle, :3] / self.dt  # For small angles
        return omega

    def _upsample_data(self, data, source_fps):
        """
        Upsample data to match the target FPS.
        
        Args:
            data: Dictionary containing the data to upsample
            source_fps: FPS of the source data
            
        Returns:
            Dictionary with upsampled data
        """
        upsampling_factor = self.target_fps / source_fps
        
        # If upsampling factor is close to 1.0, no need to upsample
        if abs(upsampling_factor - 1.0) < 0.01:
            return data
            
        # Calculate number of frames after upsampling
        num_frames = data['root_pos'].shape[0]
        upsampled_frames = int(np.ceil(num_frames * upsampling_factor))
        
        # Create time arrays for original and upsampled data
        original_time = np.linspace(0, num_frames / source_fps, num_frames)
        upsampled_time = np.linspace(0, num_frames / source_fps, upsampled_frames)
        
        # Create upsampled data dictionary
        upsampled_data = {}
        
        # Upsample positions (linear interpolation)
        upsampled_data['root_pos'] = np.zeros((upsampled_frames, data['root_pos'].shape[1]))
        for i in range(data['root_pos'].shape[1]):
            upsampled_data['root_pos'][:, i] = np.interp(upsampled_time, original_time, data['root_pos'][:, i])
        
        # Upsample joint positions (linear interpolation)
        upsampled_data['joints'] = np.zeros((upsampled_frames, data['joints'].shape[1]))
        for i in range(data['joints'].shape[1]):
            upsampled_data['joints'][:, i] = np.interp(upsampled_time, original_time, data['joints'][:, i])
        
        # Upsample quaternions (SLERP)
        upsampled_data['root_quat'] = np.zeros((upsampled_frames, 4))
        
        # Create a full set of interpolators for all frames
        # Need at least 2 frames for interpolation
        if num_frames >= 2:
            # Interpolate root quaternions
            rot_times = np.arange(num_frames)
            rot_quats = R.from_quat(data['root_quat'])
            slerp = Slerp(rot_times, rot_quats)
            
            # Calculate the interpolation times
            interp_times = np.interp(upsampled_time, original_time, rot_times)
            
            # Apply the interpolation
            upsampled_rots = slerp(interp_times)
            upsampled_data['root_quat'] = upsampled_rots.as_quat()
        else:
            # If there's only one frame, just copy it
            upsampled_data['root_quat'] = np.tile(data['root_quat'], (upsampled_frames, 1))
        
        # Copy link data if it exists
        if 'link_pos' in data and 'link_quat' in data:
            num_links = data['link_pos'].shape[1]
            
            # Upsample link positions
            upsampled_data['link_pos'] = np.zeros((upsampled_frames, num_links, 3))
            for l in range(num_links):
                for i in range(3):
                    upsampled_data['link_pos'][:, l, i] = np.interp(
                        upsampled_time, original_time, data['link_pos'][:, l, i])
            
            # Upsample link quaternions
            upsampled_data['link_quat'] = np.zeros((upsampled_frames, num_links, 4))
            
            # Need at least 2 frames for interpolation
            if num_frames >= 2:
                for l in range(num_links):
                    # Create rotation sequence for this link
                    link_quats = data['link_quat'][:, l]
                    rot_quats = R.from_quat(link_quats)
                    
                    # Create interpolator
                    link_slerp = Slerp(rot_times, rot_quats)
                    
                    # Apply interpolation
                    upsampled_link_rots = link_slerp(interp_times)
                    upsampled_data['link_quat'][:, l] = upsampled_link_rots.as_quat()
            else:
                # If there's only one frame, just copy it for all links
                for l in range(num_links):
                    upsampled_data['link_quat'][:, l] = np.tile(data['link_quat'][0, l], (upsampled_frames, 1))
        
        # Upsample contacts if they exist
        if 'contacts' in data:
            upsampled_data['contacts'] = {}
            for key, values in data['contacts'].items():
                # For contact boolean values, use nearest neighbor interpolation
                indices = np.round(np.linspace(0, len(values) - 1, upsampled_frames)).astype(int)
                upsampled_data['contacts'][key] = values[indices]
        
        # Copy non-array fields
        for key in data:
            if key not in upsampled_data and key not in ['root_pos', 'root_quat', 'joints', 'link_pos', 'link_quat', 'contacts']:
                upsampled_data[key] = data[key]
        
        upsampled_data['fps'] = self.target_fps
        
        return upsampled_data

    def _load_data(self, pkl_paths):
        """
        Loads data from multiple pkl files and concatenates them into single tensors.
        Also computes the start and end indices of each file within the concatenated tensors.
        """
        # Initialize lists to collect data from all files
        all_dofs = []
        all_motors = []
        all_root_pos = []
        all_root_vel = []
        all_root_ang_vel = []
        all_root_quat = []
        all_link_pos = []
        all_link_quat = []
        all_link_vel = []
        all_link_ang_vel = []
        sequence_lengths = []
        file_start_indices = []
        file_end_indices = []
        all_dof_vels = []
        all_motor_vels = []
        all_contacts = []
        all_extra_link_pos = []
        all_extra_link_quat = []
        all_extra_link_vel = []
        all_extra_link_ang_vel = []
        clip_metadata = []

        current_start_index = 0
        for pkl_idx, pkl_path in enumerate(tqdm(pkl_paths, desc="Loading replay data")):
            clip_metadata.append(self._parse_mocap_xia_clip_name(pkl_path))

            if self.is_csv_joint_only:
                # Load CSV data
                data = np.loadtxt(pkl_path, delimiter=',')
                replay_data = {
                    'root_pos': data[:, 0:3],
                    'root_quat': data[:, 3:7],
                    'joints': data[:, 7:],  # Rest of the columns are joint data
                    'joint_names': default_joint_orders[self.default_joint_order_type],
                }
                if self.link_names is not None:
                    # Create zero-filled arrays for link data
                    num_frames = data.shape[0]
                    num_links = len(self.link_names) + len(self.extra_link_names)
                    replay_data['link_pos'] = np.zeros((num_frames, num_links, 3))
                    replay_data['link_quat'] = np.zeros((num_frames, num_links, 4))
                    replay_data['link_quat'][..., 3] = 1.0  # Set w component to 1 for unit quaternions
                    replay_data['link_names'] = self.link_names + self.extra_link_names

                    contacts = {
                        "left_foot": np.zeros(num_frames, dtype=bool),
                        "right_foot": np.zeros(num_frames, dtype=bool),
                    }
                    replay_data['contacts'] = contacts
                    # assume whatever the default FPS is given it wont be in the CSV file
                    replay_data['fps'] = self.default_data_fps
                
            else:
                # find extension
                extension = os.path.splitext(pkl_path)[1]
                if extension == '.h5':
                    import h5py
                    # Load H5 data
                    data = h5py.File(pkl_path, 'r')

                    replay_data = {
                        'root_pos': data['root_pos'][:],
                        'root_quat': data['root_quat'][:],
                        'joints': data['joints'][:],
                        'link_pos': data['link_pos'][:],
                        'link_quat': data['link_quat'][:],
                        'joint_names': data.attrs['/joint_names'].tolist(),
                        'link_names': data.attrs['/link_names'].tolist(),
                        'fps': data.attrs['/fps'] if '/fps' in data.attrs else self.default_data_fps,
                    }

                    if not 'waist_yaw_joint' in replay_data['joint_names']:
                        replay_data['joint_names'].extend(['waist_yaw_joint', 'waist_pitch_joint', 'waist_roll_joint'])
                        replay_data['joints'] = np.concatenate([replay_data['joints'], np.zeros((replay_data['joints'].shape[0], 3))], axis=-1)
                    if 'contacts' in data:
                        replay_data['contacts'] = {
                            'left_foot': data['contacts']['left_foot'][:],
                            'right_foot': data['contacts']['right_foot'][:],
                        }
                else:
                    # Load PKL data as before
                    with open(pkl_path, 'rb') as f:
                        replay_data = pickle.load(f)
                
                if 'fps' not in replay_data:
                    if self.data_fps_override is not None and self.data_fps_override[pkl_idx] is not None:
                        replay_data['fps'] = self.data_fps_override[pkl_idx]
                    else:
                        replay_data['fps'] = self.default_data_fps
            
            data_fps_matches_target = np.round(replay_data['fps']) == np.round(self.target_fps)
            # Upsample data if requested
            if self.upsample_data and not data_fps_matches_target:
                replay_data = self._upsample_data(replay_data, replay_data['fps'])
            elif not self.upsample_data and not data_fps_matches_target:
                raise ValueError(f"Data FPS {replay_data['fps']} does not match target FPS {self.target_fps} and we have chosen not to upsample the data.")
            
            if self.cut_off_import_length != -1:
                replay_data['root_pos'] = replay_data['root_pos'][:self.cut_off_import_length]
                replay_data['root_quat'] = replay_data['root_quat'][:self.cut_off_import_length]
                replay_data['joints'] = replay_data['joints'][:self.cut_off_import_length]
                replay_data['link_pos'] = replay_data['link_pos'][:self.cut_off_import_length]
                replay_data['link_quat'] = replay_data['link_quat'][:self.cut_off_import_length]
                if 'contacts' in replay_data:
                    for key in replay_data['contacts'].keys():
                        replay_data['contacts'][key] = replay_data['contacts'][key][:self.cut_off_import_length]
            
            if self.n_prepend > 0:
                for key, value in replay_data.items():
                    if isinstance(value, np.ndarray):
                        start = self.start_offset
                        replay_data[key] = np.concatenate([np.repeat(value[start:start+1], self.n_prepend, axis=0), value[start:]])
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, np.ndarray):
                                start = self.start_offset
                                replay_data[key][subkey] = np.concatenate([np.repeat(subvalue[start:start+1], self.n_prepend, axis=0), subvalue[start:]])
                    else:
                        continue

            if self.n_append > 0:
                for key, value in replay_data.items():
                    if isinstance(value, np.ndarray):
                        replay_data[key] = np.concatenate([value, np.repeat(value[-1:], self.n_append, axis=0)])
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, np.ndarray):
                                replay_data[key][subkey] = np.concatenate([subvalue, np.repeat(subvalue[-1:], self.n_append, axis=0)])
                    else:
                        continue

            # Get joint names and link names from the file
            joint_names = replay_data['joint_names']
            link_names_in_file = replay_data['link_names']
            extra_link_names_in_file = replay_data['link_names']

            # Ensure that motor_names are in joint_names
            if self.motor_names is not None:
                # for some robots, joint names have a _joint suffix, so we need to remove it if it exists
                joint_names = [name.replace('_joint', '') for name in joint_names]
                assert all(name in joint_names for name in self.motor_names), \
                    f"Not all motor_names are in joint_names in file {pkl_path}"
            if self.dof_names is not None:
                assert all(name in joint_names for name in self.dof_names), \
                    f"Not all dof_names are in joint_names in file {pkl_path}"
            if self.link_names is not None:
                # TODO: remove this once we have the correct link names
                link_names_in_file = [name.replace('.STL', '') for name in link_names_in_file]
                assert all(name in link_names_in_file for name in self.link_names), \
                    f"Not all link_names are in link_names in file {pkl_path}"
            if self.extra_link_names is not None:
                # TODO: remove this once we have the correct link names
                extra_link_names_in_file = [name.replace('.STL', '') for name in extra_link_names_in_file]
                assert all(name in extra_link_names_in_file for name in self.extra_link_names), \
                    f"Not all extra_link_names are in extra_link_names in file {pkl_path}"
            
            # Get data
            joints = torch.tensor(np.array(replay_data['joints']), dtype=torch.float32, device=self.device)
            root_pos = torch.tensor(np.array(replay_data['root_pos']), dtype=torch.float32, device=self.device)
            root_quat = torch.tensor(np.array(replay_data['root_quat']), dtype=torch.float32, device=self.device)

            if self.adjust_root_pos:
                # Adjust root position: make xy pos relative to first frame and add height offset
                root_pos[:, 0:2] -= root_pos[0, 0:2].clone()

            if self.reference_height is not None:
                relative_z = root_pos[:, 2] - root_pos[0, 2]
                root_pos[:, 2] = relative_z + self.reference_height

            if self.height_direct_offset is not None:
                root_pos[:, 2] += self.height_direct_offset

            # Convert quaternion format if necessary to xyzw as expected
            # fix this -- we need to do this for all quaternions
            if self.data_quat_format == 'wxyz':
                raise NotImplementedError("We need to do this for all quaternions including link quaternions, not working at this time")
                root_quat = root_quat[:, [1, 2, 3, 0]]

            # Collect DOF data if requested
            if self.dof_names is not None:
                dofs = joints[:, [joint_names.index(name) for name in self.dof_names]]
                all_dofs.append(dofs)

                # Collect DOF velocitiees
                dof_vel = (dofs[1:] - dofs[:-1]) / self.dt
                dof_vel = torch.cat([dof_vel, dof_vel[-1:]], dim=0)  # Pad last value
                all_dof_vels.append(dof_vel)
            else:
                dofs = None

            # Collect motor data if requested
            if self.motor_names is not None:
                motors = joints[:, [joint_names.index(name) for name in self.motor_names]]
                all_motors.append(motors)

                # Collect motor velocities
                motor_vel = (motors[1:] - motors[:-1]) / self.dt
                motor_vel = torch.cat([motor_vel, motor_vel[-1:]], dim=0)  # Pad last value
                all_motor_vels.append(motor_vel)
            else:
                motors = None
                motor_vels = None

            # Collect link positions and orientations if requested
            if self.link_names is not None:
                link_pos_in_file = torch.tensor(np.array(replay_data['link_pos']), dtype=torch.float32, device=self.device)
                link_quat_in_file = torch.tensor(np.array(replay_data['link_quat']), dtype=torch.float32, device=self.device)

                link_pos = link_pos_in_file[:, [link_names_in_file.index(name) for name in self.link_names]]
                link_quat = link_quat_in_file[:, [link_names_in_file.index(name) for name in self.link_names]]

                if self.height_direct_offset is not None:
                    link_pos[:, :, 2] += self.height_direct_offset

                # Compute link velocities
                link_vel = (link_pos[1:] - link_pos[:-1]) / self.dt  # Shape (seq_len - 1, num_links, 3)
                link_vel = torch.cat([link_vel, link_vel[-1:]], dim=0)  # Pad last value
                all_link_vel.append(link_vel)

                # Compute link angular velocities
                link_quat_next = link_quat[1:].reshape(-1, 4)
                link_quat_prev = link_quat[:-1].reshape(-1, 4)
                link_ang_vel_flat = self._compute_angular_velocity(link_quat_next, link_quat_prev)
                link_ang_vel = link_ang_vel_flat.view(-1, len(self.link_names), 3)
                link_ang_vel = torch.cat([link_ang_vel, link_ang_vel[-1:]], dim=0)  # Pad last value
                all_link_ang_vel.append(link_ang_vel)

                all_link_pos.append(link_pos)
                all_link_quat.append(link_quat)
            else:
                link_pos = None
                link_quat = None

            # Collect extra link positions and orientations if requested
            if self.extra_link_names is not None:
                extra_link_pos_in_file = torch.tensor(np.array(replay_data['link_pos']), dtype=torch.float32, device=self.device)
                extra_link_quat_in_file = torch.tensor(np.array(replay_data['link_quat']), dtype=torch.float32, device=self.device)

                extra_link_pos = extra_link_pos_in_file[:, [link_names_in_file.index(name) for name in self.extra_link_names]]
                extra_link_quat = extra_link_quat_in_file[:, [link_names_in_file.index(name) for name in self.extra_link_names]]

                if self.height_direct_offset is not None:
                    extra_link_pos[:, :, 2] += self.height_direct_offset

                # Compute extra link velocities
                extra_link_vel = (extra_link_pos[1:] - extra_link_pos[:-1]) / self.dt
                extra_link_vel = torch.cat([extra_link_vel, extra_link_vel[-1:]], dim=0)  # Pad last value
                all_extra_link_vel.append(extra_link_vel)

                # Compute extra link angular velocities
                extra_link_quat_next = extra_link_quat[1:].reshape(-1, 4)
                extra_link_quat_prev = extra_link_quat[:-1].reshape(-1, 4)
                extra_link_ang_vel_flat = self._compute_angular_velocity(extra_link_quat_next, extra_link_quat_prev)
                extra_link_ang_vel = extra_link_ang_vel_flat.view(-1, len(self.extra_link_names), 3)
                extra_link_ang_vel = torch.cat([extra_link_ang_vel, extra_link_ang_vel[-1:]], dim=0)  # Pad last value
                all_extra_link_ang_vel.append(extra_link_ang_vel)

                all_extra_link_pos.append(extra_link_pos)
                all_extra_link_quat.append(extra_link_quat)
            else:
                extra_link_pos = None
                extra_link_quat = None

            if self.contact_names is not None:
                contact_names_in_file = replay_data['contacts']
                assert all(name in contact_names_in_file for name in self.contact_names), \
                    f"Not all contact_names are in contact_names in file {pkl_path}"
                contacts = torch.stack([torch.tensor(replay_data['contacts'][name], dtype=torch.float32, device=self.device) for name in self.contact_names], dim=1)
                all_contacts.append(contacts)

            # Compute root velocities
            root_vel = (root_pos[1:] - root_pos[:-1]) / self.dt
            root_vel = torch.cat([root_vel, root_vel[-1:]], dim=0)  # Pad last value
            all_root_vel.append(root_vel)

            # Compute root angular velocities
            root_ang_vel = self._compute_angular_velocity(root_quat[1:], root_quat[:-1])
            root_ang_vel = torch.cat([root_ang_vel, root_ang_vel[-1:]], dim=0)  # Pad last value
            all_root_ang_vel.append(root_ang_vel)

            # Append root positions and quaternions
            all_root_pos.append(root_pos)
            all_root_quat.append(root_quat)

            # Keep track of the sequence length for this file
            seq_len = root_pos.shape[0]
            sequence_lengths.append(seq_len)

            # Keep track of start and end indices
            start_idx = current_start_index
            end_idx = current_start_index + seq_len
            file_start_indices.append(start_idx)
            file_end_indices.append(end_idx)

            current_start_index = end_idx

        # Now concatenate all the collected data
        if self.dof_names is not None:
            self.dofs = torch.cat(all_dofs, dim=0)
            self.dof_vels = torch.cat(all_dof_vels, dim=0)
        else:
            self.dofs = None
            self.dof_vels = None

        if self.motor_names is not None:
            self.motors = torch.cat(all_motors, dim=0)
            self.motor_vels = torch.cat(all_motor_vels, dim=0)
        else:
            self.motors = None
            self.motor_vels = None

        self.root_pos = torch.cat(all_root_pos, dim=0)
        self.root_quat = torch.cat(all_root_quat, dim=0)
        self.root_vel = torch.cat(all_root_vel, dim=0)
        self.root_ang_vel = torch.cat(all_root_ang_vel, dim=0)

        if self.link_names is not None:
            self.link_pos = torch.cat(all_link_pos, dim=0)
            self.link_quat = torch.cat(all_link_quat, dim=0)
            self.link_vel = torch.cat(all_link_vel, dim=0)
            self.link_ang_vel = torch.cat(all_link_ang_vel, dim=0)
            self.num_links = self.link_pos.shape[1]
        else:
            self.link_pos = None
            self.link_quat = None
            self.link_vel = None
            self.link_ang_vel = None

        if self.extra_link_names is not None:
            self.extra_link_pos = torch.cat(all_extra_link_pos, dim=0)
            self.extra_link_quat = torch.cat(all_extra_link_quat, dim=0)
            self.extra_link_vel = torch.cat(all_extra_link_vel, dim=0)
            self.extra_link_ang_vel = torch.cat(all_extra_link_ang_vel, dim=0)
            self.num_extra_links = self.extra_link_pos.shape[1]
        else:
            self.extra_link_pos = None
            self.extra_link_quat = None
            self.extra_link_vel = None
            self.extra_link_ang_vel = None
            self.num_extra_links = 0

        if self.contact_names is not None:
            self.contacts = torch.cat(all_contacts, dim=0)
        else:
            self.contacts = None

        # Compute total sequence length
        self.total_sequence_length = self.root_pos.shape[0]

        # Store per-file start and end indices
        self.file_start_indices = torch.tensor(file_start_indices, device=self.device)
        self.file_end_indices = torch.tensor(file_end_indices, device=self.device)

        # Store per-file sequence lengths
        self.sequence_lengths = torch.tensor(sequence_lengths, device=self.device)
        self.num_clips = len(self.sequence_lengths)

        # For mapping indices to file indices, we can create a mapping from global indices to file indices
        self.file_indices = torch.zeros(self.total_sequence_length, dtype=torch.long, device=self.device)

        for i, (start_idx, end_idx) in enumerate(zip(self.file_start_indices, self.file_end_indices)):
            self.file_indices[start_idx:end_idx] = i

        # Build mocap_xia-style content/style views when the filename schema is available.
        self.clip_metadata = clip_metadata
        self.style_to_clip_indices = {}
        self.content_to_clip_indices = {}
        for clip_idx, clip_meta in enumerate(self.clip_metadata):
            if clip_meta is None:
                continue
            self.style_to_clip_indices.setdefault(clip_meta["style_name"], []).append(clip_idx)
            self.content_to_clip_indices.setdefault(clip_meta["content_id"], []).append(clip_idx)

        self.content_sequence_indices = []
        self.style_sequence_indices = []
        for clip_idx, clip_meta in enumerate(self.clip_metadata):
            if clip_meta is None:
                continue
            self.content_sequence_indices.append(clip_idx)
            self.style_sequence_indices.append(clip_idx)

        if any(meta is not None for meta in self.clip_metadata):
            self.content_sequence_indices = sorted(
                self.content_sequence_indices,
                key=lambda idx: (
                    self.clip_metadata[idx]["content_id"],
                    self.clip_metadata[idx]["style_name"],
                    self.clip_metadata[idx]["clip_id"],
                ),
            )
            self.style_sequence_indices = sorted(
                self.style_sequence_indices,
                key=lambda idx: (
                    self.clip_metadata[idx]["style_name"],
                    self.clip_metadata[idx]["content_id"],
                    self.clip_metadata[idx]["clip_id"],
                ),
            )

            self.content_sequence_paths = [self._pkl_paths[idx] for idx in self.content_sequence_indices]
            self.style_sequence_paths = [self._pkl_paths[idx] for idx in self.style_sequence_indices]
        else:
            self.content_sequence_paths = list(self._pkl_paths)
            self.style_sequence_paths = list(self._pkl_paths)

    def _precompute_sampling_weights(self):
        """
        Precompute weights and indices for sampling start indices.
        """
        # For each episode, compute max_start_index = episode_length - 3
        max_start_indices_per_clip = self.sequence_lengths - 3
        max_start_indices_per_clip = torch.clamp(max_start_indices_per_clip, min=1)

        # Initialize lists to collect weights and indices
        all_weights = []
        all_global_indices = []
        all_episode_indices = [] # Index of the clip (0 to num_clips-1)
        all_start_indices_within_episode = [] # Index within the specific clip's data

        for i in range(self.num_clips):
            max_start_index = max_start_indices_per_clip[i].item()
            indices = torch.arange(max_start_index, device=self.device)

            # Weights *within* this clip (only relevant if randomize_start_offset=True)
            if self.weighting_strategy == 'uniform':
                intra_clip_weights = torch.ones_like(indices, dtype=torch.float32)
            elif self.weighting_strategy == 'linear':
                intra_clip_weights = 2 * (max_start_index - indices) / (max_start_index * (max_start_index + 1))
            else:
                raise ValueError(f"Invalid weighting strategy: {self.weighting_strategy}")

            global_indices = self.file_start_indices[i] + indices
            episode_indices = torch.full((max_start_index,), i, dtype=torch.long, device=self.device)
            start_indices_within_episode = indices

            all_weights.append(intra_clip_weights)
            all_global_indices.append(global_indices)
            all_episode_indices.append(episode_indices)
            all_start_indices_within_episode.append(start_indices_within_episode)

        # Concatenate all lists
        self.all_intra_clip_weights = torch.cat(all_weights)
        self.all_global_indices = torch.cat(all_global_indices)
        self.all_episode_indices = torch.cat(all_episode_indices)
        self.all_start_indices_within_episode = torch.cat(all_start_indices_within_episode)

        # --- Calculate final sampling weights based on clip_weighting_strategy --- 
        if self.clip_weighting_strategy == 'uniform_step':
            # Each step has equal probability - use the intra-clip weights directly
            final_weights = self.all_intra_clip_weights
        elif self.clip_weighting_strategy == 'uniform_clip' or self.clip_weighting_strategy == 'success_rate_adaptive':
            # Each clip gets equal total weight initially
            # For adaptive, this will be updated later
            num_starts_per_clip = torch.bincount(self.all_episode_indices, minlength=self.num_clips)
            # Avoid division by zero for clips with no valid start indices (shouldn't happen with clamp min=1)
            num_starts_per_clip = torch.clamp(num_starts_per_clip, min=1)

            # Calculate weight per clip (uniform)
            weight_per_clip = torch.ones(self.num_clips, device=self.device) / self.num_clips

            # Distribute clip weight uniformly among its start indices
            weight_per_start_index = weight_per_clip[self.all_episode_indices] / num_starts_per_clip[self.all_episode_indices]

            # Multiply by intra-clip weighting if randomize_start_offset is True
            # (If not randomizing, intra_clip_weights are just 1s, so this doesn't change anything)
            # We need to normalize the intra_clip_weights first so they sum to 1 *within each clip*
            normalized_intra_clip_weights = torch.zeros_like(self.all_intra_clip_weights)
            for i in range(self.num_clips):
                mask = (self.all_episode_indices == i)
                clip_weights = self.all_intra_clip_weights[mask]
                if clip_weights.sum() > 0:
                     normalized_intra_clip_weights[mask] = clip_weights / clip_weights.sum()
                else: # Handle cases with zero sum (e.g., single start index)
                    normalized_intra_clip_weights[mask] = 1.0 / mask.sum().float() # Uniformly distribute

            # Apply the intra-clip distribution (scaled by number of starts in the clip)
            final_weights = weight_per_start_index * normalized_intra_clip_weights * num_starts_per_clip[self.all_episode_indices]

        else:
            raise ValueError(f"Invalid clip_weighting_strategy: {self.clip_weighting_strategy}")

        # Store the final weights used for multinomial sampling
        self.all_weights = final_weights / final_weights.sum() # Ensure normalization
        # Store base weights for adaptive updates if needed
        if self.clip_weighting_strategy == 'success_rate_adaptive':
            self._base_uniform_clip_weights_per_start = self.all_weights.clone()

    def sample_start_indices(self, env_mask):
        """
        Sample start indices for each environment specified by env_mask.
        """
        num_envs = self.num_envs
        env_mask = env_mask.to(self.device)

        num_envs_to_reset = env_mask.sum().item()

        if num_envs_to_reset == 0:
            # No environments to reset
            return (self.env_indices.clone(), self.env_end_indices.clone(),
                    self.episode_lengths.clone(), self.index_within_episode.clone(), self.episode_indices.clone(), self.episode_clip_length.clone())

        # Sample num_envs_to_reset indices from all_weights
        sampled_indices = torch.multinomial(self.all_weights, num_envs_to_reset, replacement=True)

        # Get the sampled global indices, episode indices, start indices within episode
        sampled_global_indices = self.all_global_indices[sampled_indices]
        sampled_episode_indices = self.all_episode_indices[sampled_indices]
        sampled_start_indices_within_episode = self.all_start_indices_within_episode[sampled_indices]

        # For each sampled episode, get the episode_length
        episode_lengths_all = self.sequence_lengths
        episode_clip_length = episode_lengths_all[sampled_episode_indices]
        episode_lengths_per_env = episode_lengths_all[sampled_episode_indices] - sampled_start_indices_within_episode

        # The episode end indices are the episode end indices of the episodes
        global_end_indices_all = self.file_end_indices
        global_end_indices = global_end_indices_all[sampled_episode_indices]

        # Now, create tensors of size num_envs, and fill with existing values
        new_env_indices = self.env_indices.clone()
        new_env_end_indices = self.env_end_indices.clone()
        new_episode_lengths = self.episode_lengths.clone()
        new_index_within_episode = self.index_within_episode.clone()
        new_episode_indices = self.episode_indices.clone()
        new_episode_clip_length = self.episode_clip_length.clone()

        # Update the reset environments
        env_ids = torch.where(env_mask)[0]  # Get indices where env_mask is True
        new_env_indices[env_ids] = sampled_global_indices
        new_env_end_indices[env_ids] = global_end_indices
        new_episode_lengths[env_ids] = episode_lengths_per_env
        new_index_within_episode[env_ids] = sampled_start_indices_within_episode
        new_episode_indices[env_ids] = sampled_episode_indices
        new_episode_clip_length[env_ids] = episode_clip_length

        return new_env_indices, new_env_end_indices, new_episode_lengths, new_index_within_episode, new_episode_indices, new_episode_clip_length

    def reset(self, env_mask):
        """
        Resets the environments specified by env_mask.
        Updates the indices and episode lengths for those environments.
        Returns the index within the episode and episode length per environment.
        """
        if self.randomize_start_offset:
            # Sample new indices
            (self.env_indices, self.env_end_indices,
             self.episode_lengths, self.index_within_episode, self.episode_indices, self.episode_clip_length) = self.sample_start_indices(env_mask)
        else:
            # Sample episodes uniformly for the environments to reset
            num_envs_to_reset = env_mask.sum().item()
            if num_envs_to_reset == 0:
                # No environments to reset
                return self.episode_lengths.clone()

            num_episodes = len(self.sequence_lengths)
            if self.inorder_envs:
                episode_indices = (torch.arange(num_envs_to_reset, device=self.device) % num_episodes)[:num_envs_to_reset]
            else:
                episode_indices = torch.randint(0, num_episodes, (num_envs_to_reset,), device=self.device)

            episode_start_indices = torch.tensor(self.file_start_indices, device=self.device)
            episode_end_indices = torch.tensor(self.file_end_indices, device=self.device)
            episode_lengths = torch.tensor(self.sequence_lengths, device=self.device)

            sampled_global_indices = episode_start_indices[episode_indices]
            global_end_indices = episode_end_indices[episode_indices]
            episode_lengths_per_env = episode_lengths[episode_indices]

            sampled_start_indices_within_episode = torch.zeros(num_envs_to_reset, dtype=torch.long, device=self.device)

            # Create new tensors
            new_env_indices = self.env_indices.clone()
            new_env_end_indices = self.env_end_indices.clone()
            new_episode_lengths = self.episode_lengths.clone()
            new_index_within_episode = self.index_within_episode.clone()
            new_episode_indices = self.episode_indices.clone()
            new_episode_clip_length = self.episode_clip_length.clone()

            # Update the reset environments
            env_ids = torch.where(env_mask)[0]
            new_env_indices[env_ids] = sampled_global_indices
            new_env_end_indices[env_ids] = global_end_indices
            new_episode_lengths[env_ids] = episode_lengths_per_env
            new_index_within_episode[env_ids] = sampled_start_indices_within_episode
            new_episode_indices[env_ids] = episode_indices
            # since zero offset, the episode clip length is the same as the episode length
            new_episode_clip_length[env_ids] = episode_lengths_per_env

            self.env_indices = new_env_indices
            self.env_end_indices = new_env_end_indices
            self.episode_lengths = new_episode_lengths
            self.index_within_episode = new_index_within_episode
            self.episode_indices = new_episode_indices
            self.episode_clip_length = new_episode_clip_length
        # Return the index within the episode and episode length per environment
        return self.get_episode_lengths()
    
    def get_episode_lengths(self):
        """
        Returns the episode lengths for all environments.
        """
        return self.episode_lengths.clone()
    
    def get_episode_clip_lengths(self):
        """
        Returns the episode clip lengths for all environments.
        """
        return self.episode_clip_length.clone()
    
    def get_episode_phase(self, env_ids=None):
        """
        Returns the phase of the episode for all environments.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        return self.index_within_episode[env_ids].float() / self.episode_clip_length[env_ids].float()

    def get_current_data(self, env_ids=None):
        """
        Returns the current data for the specified environments.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        env_indices = self.env_indices[env_ids]

        # Sample data for each environment based on its current index
        root_pos = self.root_pos[env_indices]
        root_quat = self.root_quat[env_indices]
        root_vel = self.root_vel[env_indices]
        root_ang_vel = self.root_ang_vel[env_indices]

        if self.dofs is not None:
            dofs = self.dofs[env_indices]
            dof_vels = self.dof_vels[env_indices]
        else:
            dofs = None
            dof_vels = None

        if self.motors is not None:
            motors = self.motors[env_indices]
            motor_vels = self.motor_vels[env_indices]
        else:
            motors = None
            motor_vels = None

        if self.link_pos is not None:
            link_pos = self.link_pos[env_indices]
            link_quat = self.link_quat[env_indices]
            link_vels = self.link_vel[env_indices]
            link_ang_vels = self.link_ang_vel[env_indices]
        else:
            link_pos = None
            link_quat = None
            link_vels = None
            link_ang_vels = None

        if self.extra_link_pos is not None:
            extra_link_pos = self.extra_link_pos[env_indices]
            extra_link_quat = self.extra_link_quat[env_indices]
            extra_link_vels = self.extra_link_vel[env_indices]
            extra_link_ang_vels = self.extra_link_ang_vel[env_indices]
        else:
            extra_link_pos = None
            extra_link_quat = None
            extra_link_vels = None
            extra_link_ang_vels = None

        if self.contact_names is not None:
            contacts = self.contacts[env_indices]
        else:
            contacts = None

        state = ReplayState(
            root_pos=root_pos,
            root_quat=root_quat,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dofs=dofs,
            dof_vels=dof_vels,
            motors=motors,
            motor_vels=motor_vels,
            link_pos=link_pos,
            link_quat=link_quat,
            link_vels=link_vels,
            link_ang_vels=link_ang_vels,
            contacts=contacts,
            clip_index=self.episode_indices[env_ids], 
            extra_link_pos=extra_link_pos,
            extra_link_quat=extra_link_quat,
            extra_link_vels=extra_link_vels,
            extra_link_ang_vels=extra_link_ang_vels
        )

        return state
    
    def get_next_data(self, K=1, env_ids=None):
        """
        Returns the next K data steps for the specified environments.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        env_indices = self.env_indices[env_ids]
        max_indices = self.env_end_indices[env_ids] - 1  # Max index per environment

        # Create steps from 1 to K
        steps = torch.arange(1, K+1, device=self.device).unsqueeze(0)  # Shape: (1, K)
        curr_indices = env_indices.unsqueeze(1)  # Shape: (num_envs, 1)
        indices = curr_indices + steps  # Shape: (num_envs, K)
        max_indices_expanded = max_indices.unsqueeze(1).expand_as(indices)
        indices = torch.min(indices, max_indices_expanded)  # Clamp indices

        # Flatten indices for indexing
        num_envs = env_ids.shape[0]
        indices_flat = indices.reshape(-1)  # Shape: (num_envs * K,)

        # Retrieve and reshape data
        root_pos = self.root_pos[indices_flat].view(num_envs, K, -1)
        root_quat = self.root_quat[indices_flat].view(num_envs, K, -1)
        root_vel = self.root_vel[indices_flat].view(num_envs, K, -1)
        root_ang_vel = self.root_ang_vel[indices_flat].view(num_envs, K, -1)

        if self.dofs is not None:
            dofs = self.dofs[indices_flat].view(num_envs, K, -1)
            dof_vels = self.dof_vels[indices_flat].view(num_envs, K, -1)
        else:
            dofs = None
            dof_vels = None

        if self.motors is not None:
            motors = self.motors[indices_flat].view(num_envs, K, -1)
            motor_vels = self.motor_vels[indices_flat].view(num_envs, K, -1)
        else:
            motors = None
            motor_vels = None

        if self.link_pos is not None:
            link_pos = self.link_pos[indices_flat].view(num_envs, K, self.num_links, -1)
            link_quat = self.link_quat[indices_flat].view(num_envs, K, self.num_links, -1)
            link_vels = self.link_vel[indices_flat].view(num_envs, K, self.num_links, -1)
            link_ang_vels = self.link_ang_vel[indices_flat].view(num_envs, K, self.num_links, -1)
        else:
            link_pos = None
            link_quat = None
            link_vels = None
            link_ang_vels = None
        
        if self.extra_link_pos is not None:
            extra_link_pos = self.extra_link_pos[indices_flat].view(num_envs, K, self.num_extra_links, -1)
            extra_link_quat = self.extra_link_quat[indices_flat].view(num_envs, K, self.num_extra_links, -1)
            extra_link_vels = self.extra_link_vel[indices_flat].view(num_envs, K, self.num_extra_links, -1)
            extra_link_ang_vels = self.extra_link_ang_vel[indices_flat].view(num_envs, K, self.num_extra_links, -1)
        else:
            extra_link_pos = None
            extra_link_quat = None
            extra_link_vels = None
            extra_link_ang_vels = None

        if self.contact_names is not None:
            contacts = self.contacts[indices_flat].view(num_envs, K, -1)
        else:
            contacts = None

        state = ReplayState(
            root_pos=root_pos,
            root_quat=root_quat,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dofs=dofs,
            dof_vels=dof_vels,
            motors=motors,
            motor_vels=motor_vels,
            link_pos=link_pos,
            link_quat=link_quat,
            link_vels=link_vels,
            link_ang_vels=link_ang_vels,
            contacts=contacts,
            clip_index=self.episode_indices[env_ids],
            extra_link_pos=extra_link_pos,
            extra_link_quat=extra_link_quat,
            extra_link_vels=extra_link_vels,
            extra_link_ang_vels=extra_link_ang_vels
        )

        return state


    def increment_indices(self, env_ids=None):
        """
        Increments the indices for the specified environments, but does not go beyond the end of the episode.
        Also increments the index within the episode for each environment.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # Increment indices
        self.env_indices[env_ids] += 1
        self.index_within_episode[env_ids] += 1

        # Clamp env_indices to not go beyond env_end_indices[env_ids] - 1
        max_indices = self.env_end_indices[env_ids] - 1
        self.env_indices[env_ids] = torch.min(self.env_indices[env_ids], max_indices)

        # Clamp index_within_episode to not go beyond episode_clip_length[env_ids] - 1
        max_index_within_episode = self.episode_clip_length[env_ids] - 1
        self.index_within_episode[env_ids] = torch.min(self.index_within_episode[env_ids], max_index_within_episode)

    def set_env_data(self, env_id: int, target_episode_idx: int, start_offset: int = 0):
        """
        Explicitly set the data for a specific environment.
        
        Args:
            env_id: Which environment to set the data for
            target_episode_idx: Which episode/clip to use
            start_offset: Where in the episode to start from
        """

        # Get the episode start and end indices
        episode_start_idx = self.file_start_indices[target_episode_idx]
        episode_end_idx = self.file_end_indices[target_episode_idx]
        episode_length = self.sequence_lengths[target_episode_idx]

        # Ensure start_offset is valid
        start_offset = min(start_offset, episode_length - 1)
        
        # Update the environment's data
        self.env_indices[env_id] = episode_start_idx + start_offset
        self.env_end_indices[env_id] = episode_end_idx
        self.episode_lengths[env_id] = episode_length - start_offset
        self.index_within_episode[env_id] = start_offset
        self.episode_indices[env_id] = target_episode_idx
        self.episode_clip_length[env_id] = episode_length # Store original full length

    def update_adaptive_weights(self, success_rates: torch.Tensor):
        """
        Update sampling weights based on success rates for the 'success_rate_adaptive' strategy.

        Args:
            success_rates: A tensor of shape (num_clips,) containing the success rate for each clip.
                         Values should ideally be in [0, 1]. Missing values can be NaN or a placeholder.
        """
        if self.clip_weighting_strategy != 'success_rate_adaptive':
            print("Warning: update_adaptive_weights called but clip_weighting_strategy is not 'success_rate_adaptive'.")
            return

        if success_rates.shape[0] != self.num_clips:
            raise ValueError(f"success_rates tensor shape ({success_rates.shape}) does not match num_clips ({self.num_clips})")

        # Ensure we have the base uniform clip weights calculated
        if not hasattr(self, '_base_uniform_clip_weights_per_start'):
            print("Warning: Base uniform weights not found for adaptive weighting. Recalculating.")
            # Temporarily set strategy to recalculate base weights
            original_strategy = self.clip_weighting_strategy
            self.clip_weighting_strategy = 'uniform_clip'
            self._precompute_sampling_weights()
            self.clip_weighting_strategy = original_strategy # Restore original strategy
            if not hasattr(self, '_base_uniform_clip_weights_per_start'): # Check again
                 raise RuntimeError("Failed to calculate base uniform weights for adaptive weighting.")

        # 1. Calculate inverse success rates per clip
        # Use a small epsilon to avoid division by zero and handle 0 success rate giving large weight
        # Handle potential NaN values in success_rates (e.g., if a clip hasn't run yet)
        # Assign average weight (1.0) to clips with NaN success rate
        valid_success_mask = ~torch.isnan(success_rates)
        inv_success = torch.ones_like(success_rates)
        if valid_success_mask.any():
            inv_success[valid_success_mask] = 1.0 / (success_rates[valid_success_mask] + 0.05)

        # 2. Normalize adjustment factors so they average to 1
        adj_factors = torch.ones_like(inv_success) 
        if valid_success_mask.any(): # Avoid division by zero if no valid rates yet
            mean_inv_success = inv_success[valid_success_mask].mean()
            if mean_inv_success > 1e-6:
                 adj_factors = inv_success / mean_inv_success
            # else: all valid success rates are ~1.0, factors remain 1.0
        
        # Assign average factor (1.0) to clips without valid success rates yet
        adj_factors[~valid_success_mask] = 1.0 

        # 3. Clamp adjustment factors
        clamped_factors = torch.clamp(adj_factors, self.min_weight_factor, self.max_weight_factor)

        # 4. Get the clamped factor for each *start index* based on its clip
        factors_per_start_index = clamped_factors[self.all_episode_indices]

        # 5. Apply factors to the base uniform clip weights
        new_weights_unnormalized = self._base_uniform_clip_weights_per_start * factors_per_start_index

        # 6. Normalize final weights
        new_total_weight = new_weights_unnormalized.sum()
        if new_total_weight > 1e-9:
            self.all_weights = new_weights_unnormalized / new_total_weight
        else:
            # Fallback to uniform if something went wrong
            print("Warning: Sum of adaptive weights is near zero. Falling back to uniform clip weights.")
            self.all_weights = self._base_uniform_clip_weights_per_start