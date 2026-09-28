import numpy as np
import viser
from viser.extras import ViserUrdf
import yourdfpy
import torch
import trimesh
from typing import Dict, Tuple, Optional, Union, List
import time
import os
import plotly.graph_objects as go
from legged_gym.tensor_utils.torch_jit_utils import quat_mul, quat_conjugate, calc_heading_quat_inv, quat_rotate, calc_heading

class LeggedRobotViser:
    """A robot visualizer using Viser, with the URDF attached under a /world root node."""

    global_servers: Dict[int, viser.ViserServer] = {}

    def __init__(self, urdf_path: str, port: int = 8080, dt: float = 1.0 / 60.0, force_dt: bool = True):
        """
        Initialize visualizer with a URDF model, loaded under a single /world node.
        
        Args:
            urdf_path: Path to the URDF file
            port: Port number for the viser server
            dt: Desired update frequency in Hz
            force_dt: If True, force the update frequency to be dt Hz
        """
        # If there is an existing server on this port, shut it down
        if port in LeggedRobotViser.global_servers:
            print(f"Found existing server on port {port}, shutting it down.")
            LeggedRobotViser.global_servers.pop(port).stop()

        self.server = viser.ViserServer(port=port)
        LeggedRobotViser.global_servers[port] = self.server

        self.dt = dt
        self.force_dt = force_dt

        # Add simulation control buttons
        with self.server.gui.add_folder("Simulation Control"):
            self.play_pause = self.server.gui.add_checkbox(
                "Play",
                initial_value=True,
                hint="Toggle simulation play/pause"
            )
            self.step_button = self.server.gui.add_button(
                "Step",
                hint="Step the simulation forward by one frame"
            )
            self.reset_button = self.server.gui.add_button(
                "Reset",
                hint="Reset both simulators to initial state"
            )
            self.step_requested = False


        # Add checkpoint selection dropdown
        self.checkpoint_selection = None
        with self.server.gui.add_folder("Checkpoint Selection"):
            self.checkpoint_selection = self.server.gui.add_dropdown(
                "Select Checkpoint",
                options=["Initial Checkpoint"],  # Start with a default option
                initial_value="Initial Checkpoint",
                hint="Select which checkpoint directory to load"
            )
            self.refresh_checkpoints = self.server.gui.add_button(
                "Refresh List",
                hint="Refresh the list of available checkpoints"
            )
        
        # Initialize action history
        self.history_length = 300  # 5 seconds at 60Hz
        self.action_history = []
        self.current_time = 0.0
        self.time_history = []
        
        # Initialize orientation error history
        self.orientation_error_history = []
        
        # Initialize joint position error history
        self.joint_error_history = []
        
        # Initialize contact forces history
        self.contact_forces_history = []

        # Initialize reward history
        self.reward_names = []
        self.rewards_history = []
        
        # Add plot controls
        with self.server.gui.add_folder("Plot Controls", expand_by_default=False, order=100):
            self.show_plot = self.server.gui.add_checkbox(
                "Show Action Plot",
                initial_value=False,
                hint="Toggle action plot visibility"
            )
            self.show_orientation_plot = self.server.gui.add_checkbox(
                "Show Orientation Error Plot",
                initial_value=False,
                hint="Toggle orientation error plot visibility"
            )
            self.show_joint_error_plot = self.server.gui.add_checkbox(
                "Show Joint Position Error Plot",
                initial_value=False,
                hint="Toggle joint position error plot visibility"
            )
            self.show_contact_forces_plot = self.server.gui.add_checkbox(
                "Show Contact Forces Plot",
                initial_value=False,
                hint="Toggle contact forces plot visibility"
            )
            self.show_rewards_plot = self.server.gui.add_checkbox(
                "Show Rewards Plot",
                initial_value=False,
                hint="Toggle rewards plot visibility"
            )
            self.rewards_plot = None



        # Initialize plot handles as None
        self.action_plot = None
        self.orientation_plot = None
        self.joint_error_plot = None
        self.contact_forces_plot = None
        self.rewards_plot = None

        @self.show_plot.on_update
        def _(event) -> None:
            if self.show_plot.value:
                # Create the action plot if it doesn't exist
                if self.action_plot is None:
                    self.action_plot = self.server.gui.add_plotly(
                        figure=go.Figure(),
                        aspect=1.0,
                        visible=True
                    )
            else:
                # Remove the plot if it exists
                if self.action_plot is not None:
                    self.action_plot.remove()
                    self.action_plot = None

        @self.show_orientation_plot.on_update
        def _(event) -> None:
            if self.show_orientation_plot.value:
                # Create the orientation plot if it doesn't exist
                if self.orientation_plot is None:
                    self.orientation_plot = self.server.gui.add_plotly(
                        figure=go.Figure(),
                        aspect=1.0,
                        visible=True
                    )
            else:
                # Remove the plot if it exists
                if self.orientation_plot is not None:
                    self.orientation_plot.remove()
                    self.orientation_plot = None

        @self.show_joint_error_plot.on_update
        def _(event) -> None:
            if self.show_joint_error_plot.value:
                # Create the joint error plot if it doesn't exist
                if self.joint_error_plot is None:
                    self.joint_error_plot = self.server.gui.add_plotly(
                        figure=go.Figure(),
                        aspect=1.0,
                        visible=True
                    )
            else:
                # Remove the plot if it exists
                if self.joint_error_plot is not None:
                    self.joint_error_plot.remove()
                    self.joint_error_plot = None

        @self.show_contact_forces_plot.on_update
        def _(event) -> None:
            if self.show_contact_forces_plot.value:
                # Create the contact forces plot if it doesn't exist
                if self.contact_forces_plot is None:
                    self.contact_forces_plot = self.server.gui.add_plotly(
                        figure=go.Figure(),
                        aspect=1.0,
                        visible=True
                    )
            else:
                # Remove the plot if it exists
                if self.contact_forces_plot is not None:
                    self.contact_forces_plot.remove()
                    self.contact_forces_plot = None

        # Add callback
        @self.show_rewards_plot.on_update
        def _(event) -> None:
            if self.show_rewards_plot.value:
                if self.rewards_plot is None:
                    self.rewards_plot = self.server.gui.add_plotly(
                        figure=go.Figure(),
                        aspect=1.0,
                        visible=True
                    )
            else:
                if self.rewards_plot is not None:
                    self.rewards_plot.remove()
                    self.rewards_plot = None


        self._isaac_world_node = self.server.scene.add_frame("/isaac_world", show_axes=False)

        # Load URDF for both simulators
        self.urdf = yourdfpy.URDF.load(urdf_path, load_collision_meshes=True)

        # Attach URDF under both world nodes
        self.isaac_urdf = ViserUrdf(
            target=self.server,
            urdf_or_path=self.urdf,
            root_node_name="/isaac_world"
        )

        # Store references to frames for both URDFs
        self._isaac_joint_frames = {
            frame.name: frame for frame in self.isaac_urdf._joint_frames
        }
        # Also store mesh handles in case you want direct references
        self._mesh_handles = {}

        # GUI Visibility controls
        # with self.server.gui.add_folder("Visualization"):
        with self.server.gui.add_folder("World Viz", expand_by_default=False):
            self.show_robot = self.server.gui.add_checkbox(
                "Show Robot", initial_value=True
            )
            self.show_terrain = self.server.gui.add_checkbox(
                "Show Terrain", initial_value=True
            )

        with self.server.gui.add_folder("Keypoints", expand_by_default=False):
            self.show_target_keypoints = self.server.gui.add_checkbox(
                "Show Target Keypoints", initial_value=True
            )
            self.show_current_keypoints = self.server.gui.add_checkbox(
                "Show Current Keypoints", initial_value=False
            )
            self.show_velocity_keypoints = self.server.gui.add_checkbox(
                "Show Velocity Keypoints", initial_value=False
            )

        with self.server.gui.add_folder("Randomisation", expand_by_default=False):
            self.enable_push_robots = self.server.gui.add_checkbox(
                "Enable Random Pushes",
                initial_value=False,
                hint="Toggle random pushing of robots"
            )
            self.push_force_scale = self.server.gui.add_slider(
                "Push Force Scale",
                min=0.1,
                max=5.0,
                step=0.1,
                initial_value=1.0,
                hint="Scale factor for random push forces in XY plane"
            )
            self.push_force_z_scale = self.server.gui.add_slider(
                "Push Force Z Scale",
                min=0.0,
                max=5.0,
                step=0.1,
                initial_value=0.0,
                hint="Scale factor for random push forces in Z direction"
            )
            self.torque_rfi_rand = self.server.gui.add_checkbox(
                "Enable Torque RFI Randomisation",
                initial_value=False,
                hint="Toggle torque RFI randomisation"
            )
            self.torque_rfi_rand_scale = self.server.gui.add_slider(
                "Torque RFI Randomisation Scale",
                min=0.0,
                max=1.0,
                step=0.01,
                initial_value=0.1,
                hint="Scale factor for torque RFI randomisation"
            )
            self.p_gain_rand = self.server.gui.add_checkbox(
                "Enable P Gain Randomisation",
                initial_value=False,
                hint="Toggle P gain randomisation"
            )
            self.p_gain_rand_scale = self.server.gui.add_slider(
                "P Gain Randomisation Scale",
                min=0.0,
                max=1.0,
                step=0.01,
                initial_value=0.1,
                hint="Scale factor for P gain randomisation"
            )
            self.d_gain_rand = self.server.gui.add_checkbox(
                "Enable D Gain Randomisation",
                initial_value=False,
                hint="Toggle D gain randomisation"
            )
            self.d_gain_rand_scale = self.server.gui.add_slider(
                "D Gain Randomisation Scale",
                min=0.0,
                max=1.0,
                step=0.01,
                initial_value=0.1,
                hint="Scale factor for D gain randomisation"
            )
        
        @self.torque_rfi_rand.on_update
        def _(event) -> None:
            if self.torque_rfi_rand.value:
                self.robot.enable_torque_rfi = True
            else:
                self.robot.enable_torque_rfi = False
        
        @self.torque_rfi_rand_scale.on_update
        def _(event) -> None:
            self.robot.torque_rfi_rand_scale = self.torque_rfi_rand_scale.value
        
        @self.p_gain_rand.on_update
        def _(event) -> None:
            if self.p_gain_rand.value:
                self.robot.p_gain_rand = True
            else:
                self.robot.p_gain_rand = False
        
        @self.p_gain_rand_scale.on_update
        def _(event) -> None:
            self.robot.p_gain_rand_scale = self.p_gain_rand_scale.value
        
        @self.d_gain_rand.on_update
        def _(event) -> None:
            if self.d_gain_rand.value:
                self.robot.d_gain_rand = True
            else:
                self.robot.d_gain_rand = False
        
        @self.d_gain_rand_scale.on_update
        def _(event) -> None:
            self.robot.d_gain_rand_scale = self.d_gain_rand_scale.value

        # Add manual control folder
        with self.server.gui.add_folder("Manual Control", expand_by_default=False):
            self.manual_control = self.server.gui.add_checkbox(
                "Manual Control Mode",
                initial_value=False,
                hint="Toggle between manual control and deepmimic mode"
            )
            # Movement controls
            self.move_forward = self.server.gui.add_checkbox(
                "Move Forward",
                initial_value=False,
                hint="Move in positive X direction"
            )
            self.move_back = self.server.gui.add_checkbox(
                "Move Back",
                initial_value=False,
                hint="Move in negative X direction"
            )
            self.move_left = self.server.gui.add_checkbox(
                "Move Left",
                initial_value=False,
                hint="Move in positive Y direction"
            )
            self.move_right = self.server.gui.add_checkbox(
                "Move Right",
                initial_value=False,
                hint="Move in negative Y direction"
            )
            self.move_up = self.server.gui.add_checkbox(
                "Move Up",
                initial_value=False,
                hint="Move in positive Z direction"
            )
            self.move_down = self.server.gui.add_checkbox(
                "Move Down",
                initial_value=False,
                hint="Move in negative Z direction"
            )
            self.rotate_left = self.server.gui.add_checkbox(
                "Rotate Left",
                initial_value=False,
                hint="Rotate counterclockwise around Z axis"
            )
            self.rotate_right = self.server.gui.add_checkbox(
                "Rotate Right",
                initial_value=False,
                hint="Rotate clockwise around Z axis"
            )

        with self.server.gui.add_folder("JIT Export", expand_by_default=False):
            # Add export policy controls
            self.export_path = self.server.gui.add_text(
                "Export Path",
                initial_value="deploy/checkpoints_sim2real/jit/policy.pt",
                hint="Directory path to export the policy to"
            )
            self.export_button = self.server.gui.add_button(
                "Export Current Policy",
                hint="Export the current policy to the specified path"
            )

        # Contact force visualization controls
        with self.server.gui.add_folder("Contact Forces"):
            self.show_contact_forces = self.server.gui.add_checkbox(
                "Show Contact Forces", initial_value=False,
                hint="Toggle contact force visualization"
            )
            self.contact_force_scale = self.server.gui.add_slider(
                "Force Scale", min=0.001, max=1.0, step=0.001, initial_value=0.01,
                hint="Scale factor for contact force arrows"
            )
            # Add checkbox for target contact visualization
            self.show_target_contacts = self.server.gui.add_checkbox(
                "Show Target Contacts", initial_value=False,
                hint="Toggle target contact state visualization (yellow dots)"
            )

        # Add clip selection dropdown
        self.clip_selection = None  # Will be set later
        self.clip_start_offset = None 
        self.content_clip_selection = None
        self.style_clip_selection = None
        self.content_clip_start_offset = None
        self.style_clip_start_offset = None
        self.phase_start_offset = None
        self.use_kinematic_replay = None

        @self.show_robot.on_update
        def _(_) -> None:
            for frame in self._isaac_joint_frames.values():
                frame.visible = self.show_robot.value

        @self.show_terrain.on_update
        def _(_) -> None:
            if 'terrain' in self._mesh_handles:
                self._mesh_handles['terrain'].visible = self.show_terrain.value

        @self.show_target_keypoints.on_update
        def _(_) -> None:
            if hasattr(self, '_target_keypoints'):
                self._target_keypoints.visible = self.show_target_keypoints.value

        @self.show_current_keypoints.on_update
        def _(_) -> None:
            if hasattr(self, '_current_keypoints'):
                self._current_keypoints.visible = self.show_current_keypoints.value

        @self.show_velocity_keypoints.on_update
        def _(_) -> None:
            if hasattr(self, '_velocity_keypoints'):
                self._velocity_keypoints.visible = self.show_velocity_keypoints.value

        @self.step_button.on_click
        def _(_):
            if not self.play_pause.value:  # Only allow stepping when paused
                self.step_requested = True

        @self.reset_button.on_click
        def _(_):
            """Reset both simulators when reset button is clicked"""
            if hasattr(self, 'robot'):
                # Reset IsaacGym environment
                self.robot.reset_idx(torch.tensor([0], device=self.robot.device))
                print(f'[PLAY] Resetting Isaac')
                
        # Add ray visualization controls
        with self.server.gui.add_folder("Ray Visualization"):
            self.show_ray_hits = self.server.gui.add_checkbox(
                "Show Ray Hit Points",
                initial_value=False,
                hint="Toggle visualization of ray hit points"
            )
            self.show_ray_starts = self.server.gui.add_checkbox(
                "Show Ray Start Points",
                initial_value=False,
                hint="Toggle visualization of ray start points"
            )
            self.show_ray_directions = self.server.gui.add_checkbox(
                "Show Ray Directions",
                initial_value=False,
                hint="Toggle visualization of ray directions"
            )
            self.show_camera_frustum = self.server.gui.add_checkbox(
                "Show Camera Frustum",
                initial_value=False,
                hint="Toggle visualization of camera frustum"
            )
            self.ray_point_size = self.server.gui.add_slider(
                "Ray Point Size",
                min=0.01,
                max=1,
                step=0.01,
                initial_value=0.01,
                hint="Size of ray points visualization"
            )
            self.frustum_scale = self.server.gui.add_slider(
                "Frustum Scale",
                min=0.1,
                max=5.0,
                step=0.1,
                initial_value=0.3,  # A larger default value to make frustums more visible
                hint="Scale of camera frustum visualization"
            )

        @self.show_ray_hits.on_update
        def _(event) -> None:
            if hasattr(self, '_heightfield_points'):
                self._heightfield_points.visible = self.show_ray_hits.value
        
        @self.show_ray_starts.on_update
        def _(event) -> None:
            if hasattr(self, '_heightfield_points'):
                self._heightfield_points.visible = self.show_ray_starts.value
        
        @self.show_ray_directions.on_update
        def _(event) -> None:
            if hasattr(self, '_heightfield_points'):
                self._heightfield_points.visible = self.show_ray_directions.value
        
        @self.show_ray_hits.on_update
        def _(event) -> None:
            if hasattr(self, '_heightfield_points'):
                self._heightfield_points.visible = self.show_ray_hits.value
        

    def init_isaacgym_robot(self, robot):
        """Setup IsaacGym robot instance"""
        self.robot = robot

    def set_viewer_camera(self, position: Union[np.ndarray, List[float]], lookat: Union[np.ndarray, List[float]]):
        """
        Set the camera position and look-at point.
        
        Args:
            position: Camera position in world coordinates
            lookat: Point to look at in world coordinates
        """
        clients = self.server.get_clients()
        for id, client in clients.items():
            client.camera.position = position
            client.camera.look_at = lookat

    def get_camera_position_for_robot(self, env_offset, root_pos):
        """
        Calculate camera position to look at the robot from 3m away.
        
        Args:
            env_offset: The environment offset
            root_pos: Current root position of the robot
        
        Returns:
            camera_pos: Position for the camera
            lookat_pos: Position to look at (robot position)
        """
        # Get actual robot position in world frame (root_pos is in env frame, need to transform to world frame)
        world_pos = root_pos + env_offset
        
        # Position to look at (robot position plus small height offset)
        lookat_pos = world_pos + np.array([0.0, 0.0, 0.5])
        
        # Calculate camera position 3m away at 45 degree angle
        distance = 3.0
        # camera_offset = np.array([distance/np.sqrt(2), -distance/np.sqrt(2), 1.5])  # 45 degrees in xy-plane
        camera_offset = np.array([0, -distance/np.sqrt(2), 1.5])  # 45 degrees in xy-plane
        camera_pos = world_pos + camera_offset
        
        return camera_pos, lookat_pos
    
    def setup_clip_selection(self, available_clips: List[str]):
        """
        Set up the clip selection dropdown with the available clips.
        
        Args:
            robot: The robot instance (should be a RobotDeepMimic instance)
            available_clips: List of available clip paths
        """
        # Get just the filenames without paths for display
        # clip_names = [os.path.basename(clip) for clip in available_clips]
        clip_names = available_clips
        self.available_clips = available_clips  # Store available clips for client connections
        
        # Create the dropdown if it doesn't exist
        if self.clip_selection is None:
            with self.server.gui.add_folder("Clip Playback"):
                self.clip_selection = self.server.gui.add_dropdown(
                    "Select Clip",
                    options=clip_names,
                    initial_value=clip_names[0] if clip_names else None,
                    hint="Select which motion clip to visualize"
                )
                self.clip_start_offset = self.server.gui.add_slider(
                    "Clip Start Frame",
                    min=0,
                    max=10000,  # Will be updated when clips are loaded
                    step=1,
                    initial_value=0,
                    hint="Select starting frame in the clip"
                )

                self.phase_start_offset = self.server.gui.add_slider(
                    "Phase Start Frame",
                    min=0,
                    max=1000,  # Will be updated when clips are loaded
                    step=1,
                    initial_value=0,
                    hint="Select starting frame in the clip"
                )

                self.use_kinematic_replay = self.server.gui.add_checkbox(
                    "Use Kinematic Replay", initial_value=False,
                    hint="Toggle between kinematic replay and physics simulation"
                )

            # Add callback for start offset
            @self.clip_start_offset.on_update
            def _(event) -> None:
                self.select_clip(self.clip_selection.value)
            
            @self.phase_start_offset.on_update
            def _(event) -> None:
                self.robot.phase_offset = self.phase_start_offset.value

            @self.use_kinematic_replay.on_update
            def _(_) -> None:
                if hasattr(self, 'robot'):
                    self.robot.viz_replay_sync_robot = self.use_kinematic_replay.value

            # Add callback for clip selection
            @self.clip_selection.on_update
            def _(event) -> None:
                self.select_clip(self.clip_selection.value)

            # Set up client connection handler
            @self.server.on_client_connect
            def _(client: viser.ClientHandle) -> None:
                """Handle new client connections by setting their camera to the current clip."""
                if hasattr(self.robot, 'env_offsets'):
                    # Get current robot state
                    current_state = self.robot.replay_data_loader.get_current_data()
                    env_offset = self.robot.env_offsets[0].cpu().numpy()
                    root_pos = current_state.root_pos[0].cpu().numpy()  # Get position for env 0
                    
                    camera_pos, lookat_pos = self.get_camera_position_for_robot(env_offset, root_pos)
                    client.camera.position = camera_pos
                    client.camera.look_at = lookat_pos
                    self.robot.set_viewer_camera(camera_pos, lookat_pos)

            # Initialize with the first clip
            if clip_names:
                self.select_clip(clip_names[0])

    def select_clip(self, clip_name: str):
        """
        Select a clip by its filename and update the visualization.
        
        Args:
            clip_name: Filename of the clip to select
        """
        for i, clip_path in enumerate(self.available_clips):
            if clip_path == clip_name:
                # Set the clip in the robot
                self.robot.set_visualization_episode(i, self.clip_start_offset.value)
                
                # Get current robot state after setting the episode
                current_state = self.robot.replay_data_loader.get_current_data()
                env_offset = self.robot.env_offsets[0].cpu().numpy()
                root_pos = current_state.root_pos[0].cpu().numpy()  # Get position for env 0
                
                # Update camera position to look at robot
                camera_pos, lookat_pos = self.get_camera_position_for_robot(env_offset, root_pos)
                self.set_viewer_camera(position=camera_pos, lookat=lookat_pos)
                break

    def setup_style_transfer_clip_selection(self, content_clips: List[Union[str, Tuple[str, int]]],
                                            style_clips: List[Union[str, Tuple[str, int]]]):
        """
        Set up paired content/style clip selectors for stage-2 style-transfer playback.

        Each option may be a display string or a (display string, replay index)
        tuple. The replay index is passed back to RobotDeepMimic unchanged.
        """
        def _normalize_options(options):
            labels = []
            index_by_label = {}
            for fallback_idx, option in enumerate(options):
                if isinstance(option, tuple):
                    label, replay_idx = option
                else:
                    label, replay_idx = option, fallback_idx
                labels.append(label)
                index_by_label[label] = int(replay_idx)
            return labels, index_by_label

        content_labels, self.content_clip_index_by_label = _normalize_options(content_clips)
        style_labels, self.style_clip_index_by_label = _normalize_options(style_clips)
        if not content_labels or not style_labels:
            return

        if self.content_clip_selection is None:
            with self.server.gui.add_folder("Style Transfer Clips"):
                self.content_clip_selection = self.server.gui.add_dropdown(
                    "Content Clip",
                    options=content_labels,
                    initial_value=content_labels[0],
                    hint="Select the motion clip used for the content stream"
                )
                self.style_clip_selection = self.server.gui.add_dropdown(
                    "Style Clip",
                    options=style_labels,
                    initial_value=style_labels[0],
                    hint="Select the motion clip used for the style stream"
                )
                self.content_clip_start_offset = self.server.gui.add_slider(
                    "Content Start Frame",
                    min=0,
                    max=10000,
                    step=1,
                    initial_value=0,
                    hint="Select the content clip starting frame"
                )
                self.style_clip_start_offset = self.server.gui.add_slider(
                    "Style Start Frame",
                    min=0,
                    max=10000,
                    step=1,
                    initial_value=0,
                    hint="Select the style clip starting frame"
                )
                self.phase_start_offset = self.server.gui.add_slider(
                    "Phase Start Frame",
                    min=0,
                    max=1000,
                    step=1,
                    initial_value=0,
                    hint="Select starting frame in the clip"
                )
                self.use_kinematic_replay = self.server.gui.add_checkbox(
                    "Use Kinematic Replay", initial_value=False,
                    hint="Toggle between kinematic replay and physics simulation"
                )

            @self.content_clip_selection.on_update
            def _(event) -> None:
                self.select_style_transfer_pair()

            @self.style_clip_selection.on_update
            def _(event) -> None:
                self.select_style_transfer_pair()

            @self.content_clip_start_offset.on_update
            def _(event) -> None:
                self.select_style_transfer_pair()

            @self.style_clip_start_offset.on_update
            def _(event) -> None:
                self.select_style_transfer_pair()

            @self.phase_start_offset.on_update
            def _(event) -> None:
                self.robot.phase_offset = self.phase_start_offset.value

            @self.use_kinematic_replay.on_update
            def _(_) -> None:
                if hasattr(self, 'robot'):
                    self.robot.viz_replay_sync_robot = self.use_kinematic_replay.value

            @self.server.on_client_connect
            def _(client: viser.ClientHandle) -> None:
                if hasattr(self.robot, 'env_offsets'):
                    current_state = self.robot.replay_data_loader.get_current_data()
                    env_offset = self.robot.env_offsets[0].cpu().numpy()
                    root_pos = current_state.root_pos[0].cpu().numpy()

                    camera_pos, lookat_pos = self.get_camera_position_for_robot(env_offset, root_pos)
                    client.camera.position = camera_pos
                    client.camera.look_at = lookat_pos
                    self.robot.set_viewer_camera(camera_pos, lookat_pos)

            self.select_style_transfer_pair()
        else:
            self.content_clip_selection.options = content_labels
            self.style_clip_selection.options = style_labels
            if self.content_clip_selection.value not in content_labels:
                self.content_clip_selection.value = content_labels[0]
            if self.style_clip_selection.value not in style_labels:
                self.style_clip_selection.value = style_labels[0]

    def select_style_transfer_pair(self):
        """Apply the selected content/style pair to the visualization environment."""
        if self.content_clip_selection is None or self.style_clip_selection is None:
            return
        content_label = self.content_clip_selection.value
        style_label = self.style_clip_selection.value
        if content_label not in self.content_clip_index_by_label or style_label not in self.style_clip_index_by_label:
            return

        content_idx = self.content_clip_index_by_label[content_label]
        style_idx = self.style_clip_index_by_label[style_label]
        self.robot.set_visualization_pair(
            content_idx,
            style_idx,
            self.content_clip_start_offset.value,
            self.style_clip_start_offset.value,
        )

        current_state = self.robot.replay_data_loader.get_current_data()
        env_offset = self.robot.env_offsets[0].cpu().numpy()
        root_pos = current_state.root_pos[0].cpu().numpy()
        camera_pos, lookat_pos = self.get_camera_position_for_robot(env_offset, root_pos)
        self.set_viewer_camera(position=camera_pos, lookat=lookat_pos)

    def add_mesh(self, 
                 name: str,
                 vertices: np.ndarray,
                 faces: np.ndarray,
                 color: Union[Tuple[float, float, float], List[float]] = (1.0, 0.5, 0.5),
                 transform: Optional[np.ndarray] = None):
        """
        Add a mesh to the scene under root "/".  (You can also attach to "/world" if you wish.)
        """
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        if transform is not None:
            mesh.apply_transform(transform)
        
        handle = self.server.scene.add_mesh_simple(
            name,
            mesh.vertices,
            mesh.faces,
            color=color,
            side='double',
            # light blue colour
        )
        self._mesh_handles[name.lstrip('/')] = handle
        return handle

    def update(self, root_pos: np.ndarray, root_quat: np.ndarray, dof_pos: np.ndarray):
        """
        Update the *global* pose by moving the entire /world node,
        then let yourdfpy handle the joint transforms via update_cfg.
        
        Args:
            root_pos: (x, y, z) of the robot base in the world
            root_quat: (w, x, y, z) quaternion for the robot base
            dof_pos: joint configuration array for the DOFs that yourdfpy expects
        """
        # Move the entire robot by setting the /world node's transform
        self._isaac_world_node.position = root_pos
        self._isaac_world_node.wxyz = root_quat

        if self.force_dt:
            if not hasattr(self, 'last_time'):
                self.last_time = time.monotonic()
            else:
                dt = time.monotonic() - self.last_time
                self.last_time = time.monotonic()
                if dt < self.dt:
                    # print(f'sleeping for {desired_dt - dt} seconds')
                    time.sleep(self.dt - dt)

        # Now let yourdfpy update the relative link transforms based on dof_pos
        with self.server.atomic():
            self.isaac_urdf.update_cfg(dof_pos)
    
    def _update_contact_force_visualization(self, rigid_body_pos, contact_forces, env_idx=0):
        if hasattr(self, '_contact_force_arrows'):
            self._contact_force_arrows.visible = self.show_contact_forces.value

        if not self.show_contact_forces.value:
            return

        # --- Existing Contact Force Visualization ---
        # (Keep the original contact force arrow logic below)
        # Convert to numpy for visualization
        positions = rigid_body_pos.cpu().numpy()
        forces = contact_forces.cpu().numpy()

        # Calculate arrow end points
        force_magnitudes = np.linalg.norm(forces, axis=1, keepdims=True)
        force_directions = np.zeros_like(forces)
        nonzero_mask = force_magnitudes > 1e-6
        if nonzero_mask.any():
            force_directions[nonzero_mask[:, 0]] = forces[nonzero_mask[:, 0]] / force_magnitudes[nonzero_mask.squeeze(1)]

        # Scale the arrows by force magnitude and user-defined scale
        arrow_ends = positions + force_directions * force_magnitudes * self.contact_force_scale.value

        # Create points array for line segments (start and end points for each arrow)
        points = np.stack([positions, arrow_ends], axis=1)
        
        # Create colors array (red for all arrows)
        num_forces = len(positions)
        colors = np.full((num_forces, 2, 3), [255, 0, 0])  # Red color for all arrows

        # Update or create the arrows visualization
        if hasattr(self, '_contact_force_arrows'):
            self._contact_force_arrows.points = points
        else:
            self._contact_force_arrows = self.server.scene.add_line_segments(
                "/contact_force_arrows",
                points=points,
                colors=colors,
                line_width=2.0,
            )
        

    def _update_target_contact_visualization(self, env_idx=0):
        # Ensure visualization handle exists and is visible if checkbox is checked
        if not hasattr(self, '_target_contact_points'):
            self._target_contact_points = self.server.scene.add_point_cloud(
                "/target_contact_points",
                points=np.zeros((0, 3)), # Start with no points
                colors=np.array([255, 255, 0]), # Yellow
                point_size=0.05,
                visible=False
            )
        self._target_contact_points.visible = self.show_target_contacts.value

        if not self.show_target_contacts.value:
            return # Don't proceed if visualization is disabled

        # --- Target Contact Visualization ---
        # Assuming robot object provides target_contacts and feet_indices
        if hasattr(self.robot, 'target_contacts') and hasattr(self.robot, 'feet_indices'):
            target_contacts = self.robot.target_contacts[env_idx].cpu().numpy() # Shape (num_feet,)
            feet_indices = self.robot.feet_indices.cpu().numpy()
            feet_positions = self.robot.rigid_body_pos[env_idx, feet_indices].cpu().numpy() # Shape (num_feet, 3)

            # Identify positions where target contact is True
            contact_positions = feet_positions[target_contacts > 0.5] # Use a threshold for boolean conversion

            # Update the point cloud
            if contact_positions.shape[0] > 0:
                self._target_contact_points.points = contact_positions
                self._target_contact_points.colors = np.array([[255, 255, 0]] * contact_positions.shape[0]) # Yellow
            else:
                # If no contacts, clear the points
                self._target_contact_points.points = np.zeros((0, 3))


    def update_contact_force_visualization(self, rigid_body_pos, contact_forces, env_idx=0):
        """
        Update the visualization of contact forces.
        
        Args:
            rigid_body_pos: Position of each rigid body (N, 3)
            contact_forces: Contact forces for each body (N, 3)
        """
        self._update_contact_force_visualization(rigid_body_pos, contact_forces, env_idx)
        self._update_target_contact_visualization(env_idx)

    def update_ray_visualization(self, env_idx=0):
        """Update the visualization of all sensors, including ray hits, start points, and directions.
        
        Args:
            env_idx: Which environment to visualize (default: 0)
        """
        if not hasattr(self, 'robot') or not hasattr(self.robot, 'sensors'):
            print("Warning: Robot or sensors not initialized for visualization")
            return
        
        # Check if visualization is enabled for any sensor
        visualization_enabled = (self.show_ray_hits.value or 
                                self.show_ray_starts.value or 
                                self.show_ray_directions.value or 
                                self.show_camera_frustum.value)
        
        # Initialize dictionaries to store visualization handles if they don't exist
        if not hasattr(self, '_sensor_viz_handles'):
            self._sensor_viz_handles = {
                'heightfield_points': {},
                'ray_start_points': {},
                'ray_direction_lines': {},
                'camera_frustums': {}
            }
        
        # Hide all visualizations if disabled
        if not visualization_enabled:
            for viz_type in self._sensor_viz_handles:
                for handle in self._sensor_viz_handles[viz_type].values():
                    if handle is not None:
                        handle.visible = False
            return
        
        # Count sensors by type for debugging
        found_heightfields = 0
        found_cameras = 0
        other_sensors = 0
        
        # Process each sensor based on its type
        for sensor_name, sensor in self.robot.sensors.items():
            from legged_gym.utils.raycaster import HeightfieldSensor, DepthCameraSensor
            # Check sensor type - heightfield sensor
            if isinstance(sensor, HeightfieldSensor):
                found_heightfields += 1
                self._update_heightfield_visualization(sensor_name, sensor, env_idx)
                
            # Check sensor type - depth camera sensor
            elif isinstance(sensor, DepthCameraSensor):
                found_cameras += 1
                self._update_camera_visualization(sensor_name, sensor, env_idx)
            else:
                other_sensors += 1
            
    def _update_heightfield_visualization(self, sensor_name, sensor, env_idx=0):
        """Update visualization for a heightfield sensor.
        
        Args:
            sensor_name: Name of the sensor
            sensor: The sensor object
            env_idx: Which environment to visualize
        """
        # Get heightfield data for visualization
        ray_starts = sensor.ray_starts_world[env_idx]
        ray_directions = sensor.ray_directions_world[env_idx]
        ray_hits = sensor.ray_hits[env_idx]
        
        # Update ray hit points visualization
        if self.show_ray_hits.value:
            # Create visualization if it doesn't exist
            if sensor_name not in self._sensor_viz_handles['heightfield_points']:
                try:

                    color = [255, 255, 0]
                    if sensor_name == "terrain_height":
                        color = [255, 255, 0]
                    elif sensor_name == "terrain_height_noisy":
                        color = [255, 0, 0]

                    self._sensor_viz_handles['heightfield_points'][sensor_name] = self.server.scene.add_point_cloud(
                        f"/heightfield_points_{sensor_name}",
                        points=ray_hits.cpu().numpy(),
                        colors=np.array([color] * ray_hits.shape[0]),
                    point_size=self.ray_point_size.value,
                        precision="float32"
                    )
                except TypeError:
                    print("Using old viser version, using default float16 precision for pointcloud")
                    self._sensor_viz_handles['heightfield_points'][sensor_name] = self.server.scene.add_point_cloud(
                        f"/heightfield_points_{sensor_name}",
                        points=ray_hits.cpu().numpy(),
                        colors=np.array([[255, 255, 0]] * ray_hits.shape[0]),  # Yellow for hits
                    )
            # Update existing visualization
            else:
                self._sensor_viz_handles['heightfield_points'][sensor_name].points = ray_hits.cpu().numpy()
                self._sensor_viz_handles['heightfield_points'][sensor_name].point_size = self.ray_point_size.value
            self._sensor_viz_handles['heightfield_points'][sensor_name].visible = True
            
        # Update ray start points visualization
        if self.show_ray_starts.value:
            # Create visualization if it doesn't exist
            if sensor_name not in self._sensor_viz_handles['ray_start_points']:
                try:
                    self._sensor_viz_handles['ray_start_points'][sensor_name] = self.server.scene.add_point_cloud(
                        f"/ray_start_points_{sensor_name}",
                        points=ray_starts.cpu().numpy(),
                        colors=np.array([[0, 255, 255]] * ray_starts.shape[0]),  # Cyan for starts
                    point_size=self.ray_point_size.value,
                        precision="float32"
                    )
                except TypeError:
                    print("Using old viser version, using default float16 precision for pointcloud")
                    self._sensor_viz_handles['ray_start_points'][sensor_name] = self.server.scene.add_point_cloud(
                        f"/ray_start_points_{sensor_name}",
                        points=ray_starts.cpu().numpy(),
                        colors=np.array([[0, 255, 255]] * ray_starts.shape[0]),  # Cyan for starts
                    )
            # Update existing visualization
            else:
                self._sensor_viz_handles['ray_start_points'][sensor_name].points = ray_starts.cpu().numpy()
                self._sensor_viz_handles['ray_start_points'][sensor_name].point_size = self.ray_point_size.value
            self._sensor_viz_handles['ray_start_points'][sensor_name].visible = True
            
        # Update ray direction lines visualization
        if self.show_ray_directions.value:
            # Calculate end points of direction lines
            ray_ends = ray_starts + ray_directions * 0.1  # Short rays for visualization
            
            # Create points array for line segments (start and end points for each ray)
            points = np.stack([ray_starts.cpu().numpy(), ray_ends.cpu().numpy()], axis=1)
            
            # Create colors array (green for all direction lines)
            num_rays = len(ray_starts)
            colors = np.full((num_rays, 2, 3), [0, 255, 0])  # Green for directions
            
            # Create or update visualization
            if sensor_name not in self._sensor_viz_handles['ray_direction_lines']:
                self._sensor_viz_handles['ray_direction_lines'][sensor_name] = self.server.scene.add_line_segments(
                    f"/ray_direction_lines_{sensor_name}",
                    points=points,
                    colors=colors,
                    line_width=1.0,
                )
            else:
                self._sensor_viz_handles['ray_direction_lines'][sensor_name].points = points
            self._sensor_viz_handles['ray_direction_lines'][sensor_name].visible = True

    def _update_camera_visualization(self, sensor_name, sensor, env_idx=0):
        """Update visualization for a depth camera sensor.
        
        Args:
            sensor_name: Name of the sensor
            sensor: The sensor object
            env_idx: Which environment to visualize
        """
        # Only proceed if camera frustum visualization is enabled
        if not self.show_camera_frustum.value:
            # Hide if already created
            if sensor_name in self._sensor_viz_handles['camera_frustums']:
                self._sensor_viz_handles['camera_frustums'][sensor_name].visible = False
            return
            
        # Get camera parameters
        fov = sensor.fov
        aspect = sensor.aspect
        scale = self.frustum_scale.value
        
        # Get camera position and orientation
        body_name = sensor.cfg.body_name
        try:
            body_idx = self.robot.body_names.index(body_name)
        except ValueError:
            print(f"Warning: Body name '{body_name}' for sensor '{sensor_name}' not found in robot body names")
            print(f"Available body names: {self.robot.body_names}")
            return
        
        body_pos = self.robot.rigid_body_pos[env_idx, body_idx].cpu().numpy()
        body_quat = self.robot.rigid_body_quat[env_idx, body_idx].cpu().numpy()
        
        # Convert body quaternion from xyzw to scipy format
        from scipy.spatial.transform import Rotation as R
        r_body = R.from_quat([body_quat[0], body_quat[1], body_quat[2], body_quat[3]])  # xyzw
        
        # Create rotation to match the raycaster transform
        # This rotation aligns the camera view direction with the forward direction
        r_transform = R.from_euler('xyz', [90, 180, 90], degrees=True)
        
        # Combine the rotations: first apply body rotation, then coordinate transform
        r_final = r_body * r_transform
        
        # Convert to quaternion (xyzw) and then to wxyz for viser
        final_quat_xyzw = r_final.as_quat()
        final_quat = np.array([final_quat_xyzw[3], final_quat_xyzw[0], final_quat_xyzw[1], final_quat_xyzw[2]])
        
        # Get depth image from the depth camera sensor
        try:
            depth_rgb = sensor.depth_map[env_idx].cpu().numpy()
            if depth_rgb is None or depth_rgb.size == 0:
                print(f"Warning: Depth map for sensor {sensor_name} is empty or None")
                depth_rgb = np.zeros((64, 64), dtype=np.uint8)  # Default empty image
        except (AttributeError, IndexError) as e:
            print(f"Error getting depth map for sensor {sensor_name}: {e}")
            depth_rgb = np.zeros((64, 64), dtype=np.uint8)  # Default empty image
        
        # Create a colormap (viridis is a good choice for depth visualization)
        # It goes from dark purple (far) to yellow (close)
        from matplotlib import cm
        normalized_depths = depth_rgb.astype(float) / 255.0 # Avoid division by zero
        colored_depths = cm.turbo(normalized_depths)
        # Convert to uint8 RGB
        depth_rgb = (colored_depths[:, :, :3] * 255).astype(np.uint8)
        
        # Create or update camera frustum visualization
        if sensor_name not in self._sensor_viz_handles['camera_frustums']:
            self._sensor_viz_handles['camera_frustums'][sensor_name] = self.server.scene.add_camera_frustum(
                f"/camera_frustum_{sensor_name}",
                fov=fov,
                aspect=aspect,
                scale=scale,
                line_width=3.0,  # Thicker lines for better visibility
                color=(0, 0, 0),  # Bright magenta color for visibility
                wxyz=final_quat,
                position=body_pos,
                image=depth_rgb,
                # format='rgb',  # Use RGB format which is supported by Viser
                format='jpeg',
                jpeg_quality=90,
            )
            # print(f"Created camera frustum for {sensor_name} at position {body_pos} with FOV {fov} and aspect {aspect}")
        else:
            self._sensor_viz_handles['camera_frustums'][sensor_name].wxyz = final_quat
            self._sensor_viz_handles['camera_frustums'][sensor_name].position = body_pos
            self._sensor_viz_handles['camera_frustums'][sensor_name].scale = scale
            self._sensor_viz_handles['camera_frustums'][sensor_name].image = depth_rgb
            self._sensor_viz_handles['camera_frustums'][sensor_name].line_width = 3.0
            self._sensor_viz_handles['camera_frustums'][sensor_name].color = (0, 0, 0)  # Update color for visibility
        
        # Make sure the frustum is visible
        self._sensor_viz_handles['camera_frustums'][sensor_name].visible = True

    def update_from_torch(self, root_states: torch.Tensor, dof_pos: torch.Tensor, env_idx: int = 0):
        """
        Update both IsaacGym visualizations.
        
        Args:
            root_states: (num_envs, 13) -> pos(3), quat(4), linvel(3), angvel(3)
            dof_pos: (num_envs, num_dofs)
            env_idx: Which environment to read from
        """
        # Block until either play is true or step is requested
        while not (self.play_pause.value or self.step_requested):
            time.sleep(0.01)  # Small sleep to prevent CPU spinning
            
        root_pos = root_states[env_idx, :3].cpu().numpy()
        root_quat = root_states[env_idx, 3:7].cpu().numpy()
        dof_pos_np = dof_pos[env_idx].cpu().numpy()
        
        # Convert quaternion from (x,y,z,w) to (w,x,y,z) for Viser
        viser_quat = np.array([root_quat[3], root_quat[0], root_quat[1], root_quat[2]])

        # Update IsaacGym visualization
        self._isaac_world_node.position = root_pos
        self._isaac_world_node.wxyz = viser_quat

        if self.force_dt:
            if not hasattr(self, 'last_time'):
                self.last_time = time.monotonic()
            else:
                dt = time.monotonic() - self.last_time
                self.last_time = time.monotonic()
                if dt < self.dt:
                    # print(f'sleeping for {desired_dt - dt} seconds')
                    time.sleep(self.dt - dt)

        # Now let yourdfpy update the relative link transforms based on dof_pos
        with self.server.atomic():
            self.isaac_urdf.update_cfg(dof_pos_np)

        # Update contact force visualization if we have a robot reference
        if hasattr(self, 'robot'):
            self.update_contact_force_visualization(
                self.robot.rigid_body_pos[env_idx],
                self.robot.contact_forces[env_idx]
            )
            
            # Update ray visualization if we have a robot reference with sensors
            if hasattr(self.robot, 'sensors'):
                self.update_ray_visualization(env_idx)
            
            # Update target contact visualization
            self.update_target_contact_visualization(env_idx=env_idx)
            
            if not hasattr(self, 'current_time'):
                self.current_time = 0.0
                self.time_history = []

            # Update action history and plot
            if hasattr(self.robot, 'actions'):
                self.current_time += self.dt
                self.time_history.append(self.current_time)
                self.action_history.append(self.robot.actions[env_idx].cpu().numpy())
                
            if hasattr(self.robot, 'target_root_pos'):
                # Calculate orientation errors
                root_quat = self.robot.root_states[env_idx, 3:7]
                target_root_quat = self.robot.target_root_quat[env_idx]
                
                # Calculate quaternion difference for root
                quat_diff_root = quat_mul(root_quat, quat_conjugate(target_root_quat))
                root_orientation_error = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff_root[:3], p=2), max=1.0))

                # Calculate torso orientation error
                torso_idx = self.robot.body_names.index('torso_link')
                torso_quat = self.robot.rigid_body_quat[env_idx, torso_idx]
                target_torso_quat = self.robot.target_extra_link_quat[env_idx, 0]
                
                # Calculate quaternion difference for torso
                quat_diff_torso = quat_mul(torso_quat, quat_conjugate(target_torso_quat))
                torso_orientation_error = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff_torso[:3], p=2), max=1.0))
                
                # Store orientation errors
                self.orientation_error_history.append([
                    root_orientation_error.cpu().item(),
                    torso_orientation_error.cpu().item()
                ])

                # Calculate and store joint position errors
                joint_errors = (self.robot.dof_pos[env_idx] - self.robot.target_motors[env_idx]).cpu().numpy()
                self.joint_error_history.append(joint_errors)
                
            if hasattr(self, 'robot'):
                # Store contact forces
                contact_forces = self.robot.contact_forces[env_idx, self.robot.feet_indices].cpu().numpy()
                self.contact_forces_history.append(contact_forces)
                
            if hasattr(self, 'reward_names'):

                # Store rewards
                # parse the reward value into a list of floats
                self.rewards_history.append([v[0].item() for v in self.robot.current_reward_value.values()])
                    # list(self.robot.current_reward_value.values()))
                # self.rewards_history = self.rewards_history[-self.history_length:]

                if len(self.reward_names) == 0:
                    self.reward_names = list(self.robot.current_reward_value.keys())
                else:
                    # sanity check
                    assert [self.reward_names[i] == list(self.robot.current_reward_value.keys())[i] for i in range(len(self.reward_names))]

                # Keep only the last history_length points
                if len(self.action_history) > self.history_length:
                    self.time_history = self.time_history[-self.history_length:]
                    self.action_history = self.action_history[-self.history_length:]
                    self.orientation_error_history = self.orientation_error_history[-self.history_length:]
                    self.joint_error_history = self.joint_error_history[-self.history_length:]
                    self.contact_forces_history = self.contact_forces_history[-self.history_length:]
                    self.rewards_history = self.rewards_history[-self.history_length:]
                
                self.update_action_plot()
                self.update_orientation_plot()
                self.update_joint_error_plot()
                self.update_contact_forces_plot()
                self.update_rewards_plot()

        # Reset step request after processing
        self.step_requested = False

    def update_action_plot(self):
        """Update the action plot with the current history."""
        if self.action_plot is None:
            return
            
        # Convert histories to numpy arrays for easier slicing
        times = np.array(self.time_history)
        actions = np.array(self.action_history)
        
        # Make times relative to current time
        current_time = self.current_time
        relative_times = times - current_time
        
        # Create a new figure
        fig = go.Figure()
        
        # Define colors for the traces
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
        
        # Plot each joint's actions
        if hasattr(self, 'robot'):
            for i, joint_name in enumerate(self.robot.dof_names):
                fig.add_trace(go.Scatter(
                    x=relative_times,
                    y=actions[:, i],
                    name=joint_name,
                    line=dict(color=colors[i % len(colors)]),
                    showlegend=False
                ))
        
        # Update layout
        fig.update_layout(
            title="Joint Actions",
            xaxis_title="Time (seconds ago)",
            yaxis_title="Action Value",
            xaxis=dict(
                range=[-5, 0],  # Fixed window of last 5 seconds
                autorange=False  # Disable autoranging
            ),
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
        )
        
        # Update the plot
        self.action_plot.figure = fig

    def update_orientation_plot(self):
        """Update the orientation error plot with the current history."""
        if self.orientation_plot is None:
            return
            
        # Convert histories to numpy arrays for easier slicing
        times = np.array(self.time_history)
        orientation_errors = np.array(self.orientation_error_history)
        
        # Make times relative to current time
        current_time = self.current_time
        relative_times = times - current_time
        
        # Create a new figure
        fig = go.Figure()
        
        # Plot root orientation error
        fig.add_trace(go.Scatter(
            x=relative_times,
            y=orientation_errors[:, 0],
            name="Root",
            line=dict(color='#e41a1c'),
            showlegend=True
        ))

        # Plot torso orientation error
        fig.add_trace(go.Scatter(
            x=relative_times,
            y=orientation_errors[:, 1],
            name="Torso",
            line=dict(color='#377eb8'),
            showlegend=True
        ))
        
        # Update layout
        fig.update_layout(
            title="Ori. Errors",
            xaxis_title="Time (seconds ago)",
            yaxis_title="Error (radians)",
            xaxis=dict(
                range=[-5, 0],  # Fixed window of last 5 seconds
                autorange=False  # Disable autoranging
            ),
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update the plot
        self.orientation_plot.figure = fig

    def update_joint_error_plot(self):
        """Update the joint position error plot with the current history."""
        if self.joint_error_plot is None:
            return
            
        # Convert histories to numpy arrays for easier slicing
        times = np.array(self.time_history)
        joint_errors = np.array(self.joint_error_history)
        
        # Make times relative to current time
        current_time = self.current_time
        relative_times = times - current_time
        
        # Create a new figure
        fig = go.Figure()
        
        # Plot joint position errors
        for i, joint_name in enumerate(self.robot.dof_names):
            fig.add_trace(go.Scatter(
                x=relative_times,
                y=joint_errors[:, i],
                name=f"{joint_name}",
                line=dict(color=f'rgba(0, 0, 255, {1 - (i / len(self.robot.dof_names))})'),
                showlegend=False
            ))
        
        # Update layout
        fig.update_layout(
            title="Joint Position Errors",
            xaxis_title="Time (seconds ago)",
            yaxis_title="Error (radians)",
            xaxis=dict(
                range=[-5, 0],  # Fixed window of last 5 seconds
                autorange=False  # Disable autoranging
            ),
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False
        )
        
        # Update the plot
        self.joint_error_plot.figure = fig

    def update_contact_forces_plot(self):
        """Update the contact forces plot with the current history."""
        if self.contact_forces_plot is None:
            return
            
        # Convert histories to numpy arrays for easier slicing
        times = np.array(self.time_history)
        contact_forces = np.array(self.contact_forces_history)
        
        # Make times relative to current time
        current_time = self.current_time
        relative_times = times - current_time
        
        # Create a new figure
        fig = go.Figure()
        
        # Colors for each component
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']  # Red for X, Blue for Y, Green for Z
        components = ['X', 'Y', 'Z']
        feet = [s for s in self.robot.body_names if self.robot.cfg.asset.foot_name in s]
        
        # Plot contact forces for each foot and component
        for foot_idx in range(len(self.robot.feet_names)):  # Two feet
            for comp_idx in range(3):  # X, Y, Z components
                # try:
                fig.add_trace(go.Scatter(
                    x=relative_times,
                    y=contact_forces[:, foot_idx, comp_idx],
                    name=f"{feet[foot_idx]} {components[comp_idx]}",
                    line=dict(
                        color=colors[foot_idx],
                        dash='dash' if comp_idx < 2 else 'solid'
                    ),
                    showlegend=True
                ))
                # except Exception as e:
                #     # print(f"Error adding trace for {feet[foot_idx]} {components[comp_idx]}: {e}")
                #     import pdb; pdb.set_trace()
        
        # Update layout
        fig.update_layout(
            title="Contact Forces",
            xaxis_title="Time (seconds ago)",
            yaxis_title="Force (N)",
            xaxis=dict(
                range=[-5, 0],  # Fixed window of last 5 seconds
                autorange=False  # Disable autoranging
            ),
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update the plot
        self.contact_forces_plot.figure = fig

    def update_rewards_plot(self):
        if self.rewards_plot is None or not self.rewards_history:
            return

        fig = go.Figure()
        # time_points = np.array(range(len(self.rewards_history))) * self.dt
        time_points = np.array(self.time_history)
        current_time = self.current_time
        relative_times = time_points - current_time

        rewards_data = np.array(self.rewards_history)
        
        for i, name in enumerate(self.reward_names):
            fig.add_trace(go.Scatter(
                x=relative_times,
                y=rewards_data[:, i],
                name=name,
                line=dict(width=2)
            ))

        fig.update_layout(
            title='Reward Terms',
            xaxis_title="Time (seconds ago)",
            yaxis_title='Reward Value',
            showlegend=False,
            height=400
        )

        self.rewards_plot.figure = fig

    def stop(self):
        """Stop the visualization server."""
        if self.server.port in LeggedRobotViser.global_servers:
            LeggedRobotViser.global_servers.pop(self.server.port).stop()

    def update_keypoints(self, target_points, current_points, velocity_points=None):
        """
        Update the keypoint visualizations.
        
        Args:
            target_points: Target keypoint positions (N, 3)
            current_points: Current keypoint positions (N, 3)
            velocity_points: Optional velocity visualization points (N, 3)
        """

        # Update target keypoints
        if not hasattr(self, '_target_keypoints'):
            try:
                self._target_keypoints = self.server.scene.add_point_cloud(
                    "/target_keypoints",
                    points=target_points,
                    colors=np.array([[255., 0, 0]] * target_points.shape[0]),
                    point_size=0.03,
                    precision="float32",
                    point_shape="circle"
                )
            except TypeError:
                print("Using old viser version, using default float16 precision for pointcloud")
                self._target_keypoints = self.server.scene.add_point_cloud(
                    "/target_keypoints",
                    points=target_points,
                    colors=np.array([[255., 0, 0]] * target_points.shape[0]),
                    point_size=0.03,
                    point_shape="circle"
                )
        else:
            self._target_keypoints.points = target_points
        self._target_keypoints.visible = self.show_target_keypoints.value

        # Update current keypoints
        if not hasattr(self, '_current_keypoints'):
            try:
                self._current_keypoints = self.server.scene.add_point_cloud(
                    "/current_keypoints",
                    points=current_points,
                    colors=np.array([[0., 255., 0]] * current_points.shape[0]),
                    point_size=0.03,
                    precision="float32",
                    point_shape="circle"
                )
            except TypeError:
                print("Using old viser version, using default float16 precision for pointcloud")
                self._current_keypoints = self.server.scene.add_point_cloud(
                    "/current_keypoints",
                    points=current_points,
                    colors=np.array([[0., 255., 0]] * current_points.shape[0]),
                    point_size=0.03,
                    point_shape="circle"
                )
        else:
            self._current_keypoints.points = current_points
        self._current_keypoints.visible = self.show_current_keypoints.value

        # Update velocity keypoints if provided
        if velocity_points is not None:
            if not hasattr(self, '_velocity_keypoints'):
                try:
                    self._velocity_keypoints = self.server.scene.add_point_cloud(
                        "/velocity_keypoints",
                        points=velocity_points,
                        colors=np.array([[0., 0., 255.]] * velocity_points.shape[0]),
                        point_size=0.03,
                        precision="float32"
                    )
                except TypeError:
                    print("Using old viser version, using default float16 precision for pointcloud")
                    self._velocity_keypoints = self.server.scene.add_point_cloud(
                        "/velocity_keypoints",
                        points=velocity_points,
                        colors=np.array([[0., 0., 255.]] * velocity_points.shape[0]),
                        point_size=0.03
                    )
            else:
                self._velocity_keypoints.points = velocity_points
            self._velocity_keypoints.visible = self.show_velocity_keypoints.value

    def setup_checkpoint_selection(self, log_root: str, on_checkpoint_selected):
        """
        Set up the checkpoint selection dropdown with available checkpoint directories.
        
        Args:
            log_root: Root directory containing checkpoint directories
            on_checkpoint_selected: Callback function when a checkpoint is selected
        """
        self.log_root = log_root  # Store for refresh functionality
        self.on_checkpoint_selected = on_checkpoint_selected  # Store for refresh functionality
        
        def refresh_checkpoint_list():
            try:
                # Get local runs
                runs = os.listdir(self.log_root)
                # Separate runs into dated and non-dated
                dated_runs = [r for r in runs if len(r) >= 15 and r[:15].replace('_','').isdigit()]
                other_runs = [r for r in runs if r not in dated_runs]
                
                # Sort dated runs by date (format YYYYMMDD_HHMMSS_*)
                dated_runs.sort(reverse=True)  # Most recent first
                
                # Sort other runs and combine lists
                other_runs.sort()
                runs = dated_runs + other_runs
                if 'exported' in runs: runs.remove('exported')

                self.wandb_run_info = {}
                wandb_options = []
                # # Get wandb runs
                # try:
                #     print("Fetching latest wandb runs...")
                #     import wandb
                #     api = wandb.Api()
                #     wandb_runs = list(api.runs("rsl_rl", order="-created_at", per_page=30))[:30]
                #     print(f"Found {len(wandb_runs)} wandb runs")
                #     # Store both name and ID for wandb runs
                #     self.wandb_run_info = {f"wandb_{run.name}": run.id for run in wandb_runs}
                #     wandb_options = list(self.wandb_run_info.keys())
                # except Exception as e:
                #     print(f"Warning: Could not fetch wandb runs: {e}")
                #     wandb_options = []
                #     self.wandb_run_info = {}
                
                # Update dropdown options
                if self.checkpoint_selection is not None:
                    # Include the initial checkpoint and wandb runs in the list
                    self.checkpoint_selection.options = ["Initial Checkpoint"] + wandb_options + runs
                    if runs or wandb_options:
                        # Keep the current value if it exists in the new options, otherwise use Initial Checkpoint
                        current_value = self.checkpoint_selection.value
                        if current_value not in self.checkpoint_selection.options:
                            self.checkpoint_selection.value = "Initial Checkpoint"
                    
            except Exception as e:
                print(f"Error refreshing checkpoint list: {e}")

        # Set up the refresh button callback
        @self.refresh_checkpoints.on_click
        def _(_):
            refresh_checkpoint_list()
            
        # Set up the checkpoint selection callback
        @self.checkpoint_selection.on_update
        def _(event) -> None:
            if self.checkpoint_selection.value and self.checkpoint_selection.value != "Initial Checkpoint":
                # If it's a wandb run, use the stored ID
                if self.checkpoint_selection.value in self.wandb_run_info:
                    self.on_checkpoint_selected("wandb_id_" + self.wandb_run_info[self.checkpoint_selection.value])
                else:
                    self.on_checkpoint_selected(self.checkpoint_selection.value)
        
        # Do initial refresh
        refresh_checkpoint_list()

    def update_visualization(self):
        """Update the visualization elements that are not part of the robot model"""
        # Update ray visualization if enabled
        if hasattr(self, 'robot') and hasattr(self.robot, 'sensors'):
            self.update_ray_visualization(env_idx=0)

    def update_target_contact_visualization(self, env_idx: int = 0):
        """
        Update the visualization of target contact states.

        Args:
            env_idx: Which environment to visualize (default: 0)
        """
        # Ensure visualization handle exists and set visibility based on checkbox
        if not hasattr(self, '_target_contact_points'):
            self._target_contact_points = self.server.scene.add_point_cloud(
                "/target_contact_points",
                points=np.zeros((0, 3)), # Start with no points
                colors=np.zeros((0, 3)), # Yellow
                point_size=0.05,
                visible=False
            )
        self._target_contact_points.visible = self.show_target_contacts.value

        if not self.show_target_contacts.value:
            return # Don't proceed if visualization is disabled

        # Check if necessary robot attributes exist
        if not hasattr(self.robot, 'target_contacts') or \
           not hasattr(self.robot, 'feet_indices') or \
           not hasattr(self.robot, 'rigid_body_pos'):
            print("Warning: Missing required attributes for target contact visualization.")
            return

        # --- Target Contact Visualization ---
        try:
            target_contacts = self.robot.target_contacts[env_idx].cpu().numpy() # Shape (num_feet,)
            feet_indices = self.robot.feet_indices.cpu().numpy()
            feet_positions = self.robot.rigid_body_pos[env_idx, feet_indices].cpu().numpy() # Shape (num_feet, 3)

            # Identify positions where target contact is True
            contact_mask = target_contacts > 0.5 # Use a threshold
            contact_positions = feet_positions[contact_mask]

            # Update the point cloud
            if contact_positions.shape[0] > 0:
                self._target_contact_points.points = contact_positions
                self._target_contact_points.colors = np.array([[255, 255, 0]] * contact_positions.shape[0]) # Yellow
            else:
                # If no contacts, clear the points
                self._target_contact_points.points = np.zeros((0, 3))

        except Exception as e:
            print(f"Error during target contact visualization update: {e}")
            # Ensure points are cleared on error
            self._target_contact_points.points = np.zeros((0, 3))
