from legged_gym.utils.deepmimic_terrain import DeepMimicTerrain
import numpy as np
from dataclasses import MISSING
import torch
from typing import Union, List

from legged_gym.utils.configclass import configclass

from legged_gym.envs.base.legged_robot_config import (
    LeggedRobotTerrainCfg,
    LeggedRobotInitStateCfg,
    LeggedRobotEnvCfg,
    LeggedRobotDomainRandCfg,
    LeggedRobotControlCfg,
    LeggedRobotAssetCfg,
    LeggedRobotRewardsCfg,
    LeggedRobotNormalizationCfg,
    LeggedRobotNoiseCfg,
    LeggedRobotSimCfg,
    LeggedRobotCommandsCfg,
    LeggedRobotCfg,
    LeggedRobotPolicyCfg,
    LeggedRobotAlgorithmCfg,
    LeggedRobotRunnerCfg,
    LeggedRobotCfgPPO,
    LeggedRobotSensorsCfg,
    DepthCameraCfg,
    HeightfieldCfg,
    MultiLinkHeightCfg,
)

@configclass
class LeggedRobotDeepMimicCfg:
    init_velocities = True
    randomize_start_offset = True
    # init_velocities = True
    # randomize_start_offset = True
    n_prepend = 0
    n_append = 0
    respawn_z_offset = 0.0
    height_direct_offset = 0.0
    link_pos_error_threshold = 0.3
    viz_replay = False
    viz_replay_sync_robot = False
    num_next_obs = 1

    # length to truncate rollout to, used for very long sequences
    truncate_rollout_length = -1

    upsample_data = True
    default_data_fps = -1

    contact_names = ['left_foot', 'right_foot']
    # contact_names = None # ['left_foot', 'right_foot']

    tracked_body_names = [
        'pelvis',

        'left_hip_pitch_link',
        'left_knee_link',
        'left_ankle_roll_link',

        'right_hip_pitch_link',
        'right_knee_link',
        'right_ankle_roll_link',

        'left_shoulder_pitch_link',

        'right_shoulder_pitch_link',

    ]

    randomize_terrain_offset = False
    randomize_terrain_offset_range = 1.0

    # Choose how to weight clips for sampling start states:
    # 'uniform_step': Each step across all clips has equal probability.
    # 'uniform_clip': Each clip has equal total probability, distributed uniformly among its steps.
    # 'success_rate_adaptive': Each clip's probability is inversely proportional to its success rate (within bounds).
    # clip_weighting_strategy: str = 'uniform_step'
    clip_weighting_strategy: str = 'success_rate_adaptive'
    # Factors for clamping adaptive weights (relative to uniform clip weight)
    min_success_rate_weight_factor: float = 1.0 / 3.0
    max_success_rate_weight_factor: float = 3.0
    # How often (in simulation steps) to update adaptive weights. Should be >= log_success_rate_interval in G1DeepMimic.
    adaptive_weight_update_interval: int = 5000 # Steps

    weighting_strategy = 'uniform' # uniform or linear. Weighting *within* an episode if randomize_start_offset is True.

    num_tracked_links = len(tracked_body_names)

    extra_link_names = [
        'torso_link'
    ]


    # init_positions_mode = 'replay_data'
    # init_positions_mode = 'default_pos'
    # init_default_frac = 0.05
    init_default_frac = 0.0

    use_amass = True
    amass_replay_data_path = 'lafan_walk_and_dance/*.pkl'

    # human_video_data_pattern = 'env_0_retarget_poses_g1_fit.pkl'
    # human_video_terrain_pattern = 'background_mesh_fit.obj'
    human_video_data_pattern = 'retarget_poses_g1.h5'
    human_video_terrain_pattern = 'background_mesh.obj'

    # Teacher checkpoint to use for AMASS data
    amass_teacher_checkpoint_run_name: str = "20250410_063030_g1_deepmimic"
    amass_terrain_difficulty = 2

    # data_root = 'demo_data/output_postprocessed'
    # data_root = 'demo_data/output_postprocessed_align3r'
    # alt_data_root= 'demo_data/output_postprocessed'
    data_root = '../data/videomimic_captures'
    alt_data_root= '' # somewhere else to look for data
    amass_data_root = '../data/unitree_lafan'

    is_csv_joint_only=False
    default_joint_order_type="g1"
    cut_off_import_length=-1

    zero_torso_xy = False
    zero_torso_yaw = False

    use_human_videos = False
    # Path to a YAML file listing human motion clips (relative to LEGGED_GYM_ROOT_DIR)
    # OR the name of a single human motion data folder (relative to data_root in g1_deepmimic.py)
    human_motion_source: str = 'resources/data_config/human_motion_list.yaml'
    # Allow directly providing a single .h5 path or glob pattern (e.g. ../data/mocap_xia/*.h5).
    allow_h5_direct_source = False
    # human_video_folders = [ # OLD LIST - REMOVED
    #     # ... old list content ...
    # ]

    human_video_oversample_factor = 1

@configclass
class LeggedRobotDeepMimicTerrainCfg(LeggedRobotTerrainCfg):
    terrain_class = 'DeepMimicTerrain'
    mesh_type = 'trimesh'
    n_rows = 1


    terrain_noise = None
    # terrain_noise = {
    #     # 'base_frequency': 0.5,
    #     # 'amplitude': 0.1,
    #     # 'octaves': 3,
    #     # 'persistence': 0.5
    #     # 'base_frequency': 0.5,
    #     'base_frequency': 5.0,
    #     'amplitude': 0.3,
    #     'octaves': 1,
    #     'persistence': 0.5,
    #     'random_z_scaling_enable': False,
    #     'random_z_scaling_scale': 0.02
    # }

    alternate_cast_to_heightfield = False
    cast_mesh_to_heightfield = False
    # If True, missing terrain meshes are replaced by a generated flat plane.
    auto_flat_if_mesh_missing = False
    flat_terrain_size = 20.0
    flat_terrain_z = 0.0


@configclass
class G1DeepMimicRewardScalesCfg:
    # defaults
    tracking_lin_vel = 0.0
    tracking_ang_vel = 0.0

    lin_vel_z = 0.0
    ang_vel_xy = 0.00
    orientation = 0.0
    base_height = 0.0
    
    # regularisation terms
    dof_acc = 0.0 #-1e-6 / 3.0 
    dof_vel = 0.0 # -5e-4 / 3.0 
    torques = 0.0 
    energy = 0.0 #-0.000005
    action_rate = -0.2
    action_accel = 0.0 # -0.01

    ankle_action = 0.0 

    no_fly = 0.0#-50.0

    collision = -15.0

    dof_pos_limits = -50.0
    alive = 0.0
    hip_pos = 0.0 #-1.0
    # contact_no_vel = -5.0
    contact_no_vel = -100.0
    feet_swing_height = 0.0
    contact = 0.0

    feet_orientation = 0.0

    root_vel_tracking = 0.0
    root_ang_vel_tracking = 0.0

    joint_pos_tracking = 120.0
    link_pos_tracking = 30.0
    root_pos_tracking = 0.0
    torso_pos_tracking = 15.0
    root_orientation_tracking = 15.0
    torso_orientation_tracking = 15.0

    link_vel_tracking = 5.0
    joint_vel_tracking = 24.0


    feet_contact_matching = 1.0 
    contact_smoothness = 0.0 

    feet_max_height_for_this_air = 0.0

    termination= -500.0
    # termination= -0.0
    
    feet_air_time = 2000.0


@configclass
class G1DeepMimicRewardsCfg(LeggedRobotRewardsCfg):
    # soft_dof_pos_limit = 1.0
    soft_dof_pos_limit = 0.98
    base_height_target = 0.78
   
    scales = G1DeepMimicRewardScalesCfg()

    joint_pos_tracking_k = 2.0
    joint_vel_tracking_k = 0.01
    torso_pos_tracking_k = 50.0
    torso_orientation_tracking_k = 3.0
    # link_pos_tracking_k = 10.0
    link_pos_tracking_k = 5.0
    link_vel_tracking_k = 0.1

    # these are disabled
    root_pos_tracking_k = 20.0
    root_orientation_tracking_k = 3.0
    root_vel_tracking_k = 10.0
    root_ang_vel_tracking_k = 0.01

    only_positive_rewards = False


@configclass
class G1DeepMimicNormalizationCfg(LeggedRobotNormalizationCfg):
    @configclass
    class ObsScales:
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
        height_measurements = 5.0
    obs_scales = ObsScales()
    clip_observations = 100.
    clip_actions = 8.0

@configclass
class G1DeepMimicNoiseCfg(LeggedRobotNoiseCfg):
    add_noise = True
    noise_level = 1.0 # scales other values

    # noise levels (uncorrelated noise)
    @configclass
    class NoiseScales:
        dof_pos = 0.01
        dof_vel = 1.5
        lin_vel = 0.1
        ang_vel = 0.2
        gravity = 0.05

        rel_xy = 0.01
        rel_yaw = 0.01
    noise_scales = NoiseScales()

    # fixed offsets (correlated noise)
    @configclass
    class OffsetScales:
        action = 0.0
        dof_pos = 0.00
        gravity = 0.0

    offset_scales = OffsetScales()

    # noise levels initialising the dof positions of the robot
    @configclass
    class InitNoiseScales:
        dof_pos = 0.0
        dof_vel = 0.0

        root_xy = 0.0
        root_z = 0.0
        root_quat = 0.0

    init_noise_scales = InitNoiseScales()

    @configclass
    class PlayBackNoiseScales:

        freeze_env_prob = 0.0
        unfreeze_env_prob = 0.0
    
    playback_noise_scales = PlayBackNoiseScales()

@configclass
class G1BaseAsset(LeggedRobotAssetCfg):
    file: str = MISSING#'{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf',
    num_dofs: int = MISSING

    foot_name = "ankle_roll"
    penalize_contacts_on = []
    terminate_after_contacts_on = []

    terminate_after_large_feet_contact_forces = False
    large_feet_contact_force_threshold = 1000.0

    # terminate_after_contacts_on = [],
    self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
    flip_visual_attachments = False
    collapse_fixed_joints = False
    armature = 0.001

@configclass
class G129Anneal23DofAsset(G1BaseAsset):
    file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_29dof_anneal_23dof.urdf'
    # used on subsequent gpus to randomise the feet contacts
    use_alt_files = True
    alt_files = [
        '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_29dof_anneal_23dof_spheres.urdf'
    ]

    num_dofs = 23

    dont_collide_groups = {
        0: {
            "left_ankle_roll_link",
            "left_ankle_pitch_link",
            "left_knee_link"
        },
        1: {
            "right_ankle_roll_link",
            "right_ankle_pitch_link",
            "right_knee_link"
        }
    }


    tracked_body_names = [
        'pelvis',

        'left_hip_pitch_link',
        # 'left_hip_roll_link',
        # 'left_hip_yaw_link',
        'left_knee_link',
        # 'left_ankle_pitch_link',
        'left_ankle_roll_link',

        'right_hip_pitch_link',
        # 'right_hip_roll_link',
        # 'right_hip_yaw_link',
        'right_knee_link',
        # 'right_ankle_pitch_link',
        'right_ankle_roll_link',

        # 'waist_yaw_link',
        # 'waist_roll_link',

        # 'torso_link',

        'left_shoulder_pitch_link',
        # # 'left_shoulder_roll_link',
        # # 'left_shoulder_yaw_link',
        'left_elbow_link',
        # # 'left_wrist_roll_link',
        # # 'left_wrist_pitch_link',
        'left_wrist_yaw_link',

        'right_shoulder_pitch_link',
        # # 'right_shoulder_roll_link',
        # # 'right_shoulder_yaw_link',
        'right_elbow_link',
        # # 'right_wrist_roll_link',
        # # 'right_wrist_pitch_link',
        'right_wrist_yaw_link',

    ]

    num_tracked_links = len(tracked_body_names)

    terminate_after_contacts_on = [
        # "left_elbow_link",
        # "right_elbow_link",
        # "left_shoulder_pitch_link",
        # "right_shoulder_pitch_link",
        # "left_shoulder_roll_link",
        # "right_shoulder_roll_link",
        # "left_shoulder_yaw_link",
        # "right_shoulder_yaw_link",
        # # didn't have this in the last training
        # "left_rubber_hand",
        # "right_rubber_hand",
    ]

    penalize_contacts_on = [

        # "left_elbow_link",
        # "right_elbow_link",
        # "left_shoulder_pitch_link",
        # "right_shoulder_pitch_link",
        # "left_shoulder_roll_link",
        # "right_shoulder_roll_link",
        # "left_shoulder_yaw_link",
        # "right_shoulder_yaw_link",
        # didn't have this in the last training
        "left_rubber_hand",
        "right_rubber_hand",
    ]

    upper_body_dof_names = [
        'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint'
        ]




tracked_body_names = [
    'pelvis',
    'left_hip_pitch_link',
    'left_knee_link',
    'left_ankle_roll_link',
    'right_hip_pitch_link',
    'right_knee_link',
    'right_ankle_roll_link',
    'left_shoulder_pitch_link',
    'left_elbow_link',
    'left_wrist_yaw_link',
    'right_shoulder_pitch_link',
    'right_elbow_link',
    'right_wrist_yaw_link',
]

sensor_cfgs = [
    # Terrain heightfield sensor
    HeightfieldCfg(
        name="terrain_height",
        body_name="torso_link",
        size=(1.0, 1.0),
        resolution=0.1,
        max_distance=5.0,
        use_float=True,  # Use float32 instead of uint8
        white_noise_scale=0.0,
        offset_noise_scale=0.0,
        roll_noise_scale=0.0,
        pitch_noise_scale=0.0,
        yaw_noise_scale=0.0,
    ),
    HeightfieldCfg(
        name="terrain_height_noisy",
        body_name="torso_link",
        size=(1.0, 1.0),
        resolution=0.1,
        max_distance=5.0,
        use_float=True,  # Use float32 instead of uint8
        white_noise_scale=0.02,
        offset_noise_scale=0.02,
        roll_noise_scale=0.04,
        pitch_noise_scale=0.04,
        yaw_noise_scale=0.08,
        max_delay=3,
        update_frequency_min=1,
        update_frequency_max=5,
        bad_distance_prob=0.01,
    ),

    # Root height sensor (single point)
    HeightfieldCfg(
        name="root_height",
        body_name="pelvis",
        size=(0.0, 0.0),
        resolution=0.1,
        max_distance=5.0,
        use_float=True,  # Use float32 instead of uint8
    ),
    
    # Multi-link height sensor for all tracked bodies
    MultiLinkHeightCfg(
        name="link_heights",
        body_name="pelvis",  # Reference body
        max_distance=5.0,
        link_names=tracked_body_names,
        use_float=True,  # Use float32 instead of uint8

    ),
    # Uncomment to use depth camera, then add `depth_camera` to obs list
    # DepthCameraCfg(
    #     name="depth_camera",
    #     body_name="d435_link",
    #     # change downsample factor and divide width and height by corresponding factor
    #     downsample_factor=2,
    #     width=320 // 2,
    #     height=240 // 2,
    #     max_distance=5.0,
    # ),
]


low_stiffness_cfg = {
    'stiffness': {'hip_yaw': 75,
                     'hip_roll': 75,
                     'hip_pitch': 75,
                     'knee': 75,
                     'ankle_pitch': 20,
                     'ankle_roll': 20,
                     'waist_yaw': 75,
                     'waist_roll': 75,
                     'waist_pitch': 75,
                     'shoulder_pitch': 75,
                     'shoulder_roll': 75,
                     'shoulder_yaw': 75,
                     'elbow': 75,
                     },  # [N*m/rad]

    'damping': {'hip_yaw': 2.,
                     'hip_roll': 2.,
                     'hip_pitch': 2.,
                     'knee': 2.,
                     'ankle_pitch': 0.2,
                     'ankle_roll': 0.1,
                     'waist_yaw': 2.0,
                     'waist_roll': 2.0,
                     'waist_pitch': 2.0,
                     'shoulder_pitch': 2.0,
                     'shoulder_roll': 2.0,
                     'shoulder_yaw': 2.0,
                     'elbow': 2.0,
                     },  # [N*m/rad]
}

    
@configclass
class G1DeepMimicCfg(LeggedRobotCfg):

    asset = G129Anneal23DofAsset()
  
    terrain = LeggedRobotDeepMimicTerrainCfg()

    # Example sensor configuration using the new format
    sensors = LeggedRobotSensorsCfg(
        sensor_cfgs = sensor_cfgs,
    )

    init_state = LeggedRobotInitStateCfg(
        pos = [0.0, 0.0, 0.78], # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,       
           'torso_joint' : 0.,
           'waist_yaw_joint' : 0.,
           'waist_pitch_joint' : 0.,
           'waist_roll_joint' : 0.,
           'left_shoulder_pitch_joint' : 0.,
           'left_shoulder_roll_joint' : 0.,
           'left_shoulder_yaw_joint' : 0.,
           'right_shoulder_pitch_joint' : 0.,
           'right_shoulder_roll_joint' : 0.,
           'right_shoulder_yaw_joint' : 0.,
           'left_elbow_joint' : 0.,
           'left_wrist_joint' : 0.,
           'right_elbow_joint' : 0.,
           'right_wrist_joint' : 0.,
           'left_wrist_roll_joint' : 0.,
           'left_wrist_pitch_joint' : 0.,
           'left_wrist_yaw_joint' : 0.,
           'right_wrist_roll_joint' : 0.,
           'right_wrist_pitch_joint' : 0.,
           'right_wrist_yaw_joint' : 0.,

        }
    )

    deepmimic = LeggedRobotDeepMimicCfg(
        num_tracked_links = asset.num_tracked_links,
        tracked_body_names = asset.tracked_body_names,
    )

    num_actions = asset.num_dofs  # G1 has 29 DOF



    env = LeggedRobotEnvCfg(
        num_actions = num_actions,
        obs = ['torso', 'torso_real', 'deepmimic', 'teacher', 'deepmimic_lin_ang_vel', 'terrain_height', 'terrain_height_noisy', 'root_height', 'phase', 'torso_xy_rel', 'torso_yaw_rel', 'torso_xy', 'torso_yaw', 'target_joints', 'target_root_roll', 'target_root_pitch', 'target_root_yaw', 'upper_body_joint_targets', 'teacher_checkpoint_index'],#, 'depth_camera'],
        obs_history = {
            'torso_real': 5,
            'torso_xy_rel': 5,
            'torso_yaw_rel': 5,
            'torso_xy': 5,
            'torso_yaw': 5,
            'deepmimic_lin_ang_vel': 5,
        }
    )

    rewards = G1DeepMimicRewardsCfg()
    normalization = G1DeepMimicNormalizationCfg()
    noise = G1DeepMimicNoiseCfg()

    
    domain_rand = LeggedRobotDomainRandCfg(
        randomize_friction = True,
        friction_range = [0.1, 1.25],
        randomize_base_mass = False,
        added_mass_range = [-1., 3.],
        push_robots = False,
        push_interval_s = 10,
        max_push_vel_xy = 0.25,
        torque_rfi_rand = False,
        torque_rfi_rand_scale = 0.04,
        p_gain_rand = False,
        p_gain_rand_scale = 0.03,
        d_gain_rand = False,
        d_gain_rand_scale = 0.03,
        # crucial for sim2sim at least
        randomize_dof_friction = False,
        max_dof_friction = 0.05,
        dof_friction_buckets = 64
    )

    control = LeggedRobotControlCfg(
   
        beta = 1.0,
        action_scale = 0.25,
        decimation = 4,
        control_type = 'P',
        **low_stiffness_cfg,
    )


    sim = LeggedRobotSimCfg(
        # dt =  0.005
        # dt =  1 / 240.,
        dt =  1 / 200.,
        # substeps = 4
        substeps = 1,
        gravity = [0., 0. ,-9.81],  # [m/s^2]
        up_axis = 1,  # 0 is y, 1 is z

        physx = LeggedRobotSimCfg.Physx(
            num_threads = 10,
            solver_type = 1,  # 0: pgs, 1: tgs
            num_position_iterations = 4,
            num_velocity_iterations = 0,
            contact_offset = 0.01,  # [m]
            rest_offset = 0.0,   # [m]
            bounce_threshold_velocity = 0.5, #0.5 [m/s]
            # max_depenetration_velocity = 1.0
            max_depenetration_velocity = 0.1,
            # max_gpu_contact_pairs = 2**23, #2**24 -> needed for 8000 envs and more
            max_gpu_contact_pairs = 2**24, #2**24 -> needed for 8000 envs and more
            # max_gpu_contact_pairs = 2**25, #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5,
            contact_collection = 2, # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
        )
    )


@configclass
class G1DeepMimicMocapCfg(G1DeepMimicCfg):

    asset = G129Anneal23DofAsset(
        terminate_after_large_feet_contact_forces=False
        )

    deepmimic = LeggedRobotDeepMimicCfg(
        use_amass=True,
        use_human_videos=False,
        link_pos_error_threshold=0.5,
        amass_terrain_difficulty=1,
        default_data_fps=30, # correct for lafan dataset
        cut_off_import_length=1000, # set to -1 to not cut off and use whole dataset
        amass_replay_data_path="lafan_walk_and_dance/*.pkl",
        num_tracked_links = asset.num_tracked_links,
        tracked_body_names = asset.tracked_body_names,
    )

    rewards = G1DeepMimicRewardsCfg(
        scales = G1DeepMimicRewardScalesCfg(
            contact_no_vel=5.0,
            feet_air_time=0.0,
            dof_pos_limits=-5.0,
            action_rate=-0.1,
        )
    )

    noise = G1DeepMimicNoiseCfg(add_noise=False)


@configclass
class G1DeepMimicPolicyCfg(LeggedRobotPolicyCfg):
    init_noise_std = 0.8

    @configclass
    class ObsProcActor:
        # torso = { 'type': 'identity' }
        # deepmimic = { 'type': 'identity' }
        history_torso_real = {'type': 'flatten'}
        # # history_deepmimic_lin_ang_vel = {'type': 'flatten'}
        history_torso_xy_rel = {'type': 'flatten'}
        history_torso_yaw_rel = {'type': 'flatten'}

        target_joints = {'type': 'identity'}
        target_root_roll = {'type': 'identity'}
        target_root_pitch = {'type': 'identity'}
        # target_root_yaw = {'type': 'flatten'}

        # history_torso_xy = {'type': 'flatten'}
        # history_torso_yaw = {'type': 'flatten'}

        # phase = {'type': 'identity'}
        # terrain_height = { 'type': 'flatten' }
        # terrain_height = { 'type': 'flatten_then_embed' , 'output_dim': 918}
        # terrain_height = { 'type': 'flatten_then_embed_with_attention' , 'output_dim': 918}
        
        # # Include sensor observations with the new names
        # front_camera = { 'type': 'flatten' }  # Flatten the front camera depth image
        # down_camera = { 'type': 'flatten' }   # Flatten the downward camera depth image
        # terrain_height = { 'type': 'downsample', 'factor': 2 }  # Downsample the heightfield
    
    @configclass
    class ObsProcCritic:
        torso = { 'type': 'identity' }
        deepmimic = { 'type': 'identity' }

        history_torso_real = {'type': 'flatten'}
        history_torso_xy_rel = {'type': 'flatten'}
        history_torso_yaw_rel = {'type': 'flatten'}

        target_joints = {'type': 'identity'}
        target_root_roll = {'type': 'identity'}
        target_root_pitch = {'type': 'identity'}

        # target_root_yaw = {'type': 'flatten'}
        # history_torso_xy = {'type': 'flatten'}
        # history_torso_yaw = {'type': 'flatten'}

        # phase = {'type': 'identity'}

        # # history_deepmimic_lin_ang_vel = {'type': 'flatten'}

        # terrain_height = { 'type': 'flatten_then_embed' , 'output_dim': 918}
        # terrain_height = { 'type': 'flatten_then_embed_with_attention' , 'output_dim': 918}
        
        # # Include sensor observations for critic too
        # front_camera = { 'type': 'flatten' }
        # down_camera = { 'type': 'flatten' }
        # terrain_height = { 'type': 'downsample', 'factor': 2 }

    obs_proc_actor = ObsProcActor()
    obs_proc_critic = ObsProcCritic()

@configclass
class G1DeepmimicHeightFieldPolicyCfg(G1DeepMimicPolicyCfg):

    @configclass
    class ObsProcActor:
        # torso = { 'type': 'identity' }
        # deepmimic = { 'type': 'identity' }
        history_torso_real = {'type': 'flatten'}
        # # history_deepmimic_lin_ang_vel = {'type': 'flatten'}
        history_torso_xy_rel = {'type': 'flatten'}
        history_torso_yaw_rel = {'type': 'flatten'}

        target_joints = {'type': 'identity'}
        target_root_roll = {'type': 'identity'}
        target_root_pitch = {'type': 'identity'}

        # history_torso_xy = {'type': 'flatten'}
        # history_torso_yaw = {'type': 'flatten'}

        # phase = {'type': 'identity'}
        # terrain_height = { 'type': 'flatten' }
        # terrain_height = { 'type': 'flatten_then_embed' , 'output_dim': 918}
        terrain_height = { 'type': 'flatten_then_embed_with_attention' , 'output_dim': 415}
        # terrain_height = { 'type': 'flatten_then_embed_with_attention_to_hidden' }
        
        # # Include sensor observations with the new names
        # front_camera = { 'type': 'flatten' }  # Flatten the front camera depth image
        # down_camera = { 'type': 'flatten' }   # Flatten the downward camera depth image
        # terrain_height = { 'type': 'downsample', 'factor': 2 }  # Downsample the heightfield
    
    @configclass
    class ObsProcCritic:
        torso = { 'type': 'identity' }
        deepmimic = { 'type': 'identity' }

        history_torso_real = {'type': 'flatten'}
        history_torso_xy_rel = {'type': 'flatten'}
        history_torso_yaw_rel = {'type': 'flatten'}

        target_joints = {'type': 'identity'}
        target_root_roll = {'type': 'identity'}
        target_root_pitch = {'type': 'identity'}

        # target_root_yaw = {'type': 'flatten'}
        # history_torso_xy = {'type': 'flatten'}
        # history_torso_yaw = {'type': 'flatten'}

        # phase = {'type': 'identity'}

        terrain_height = { 'type': 'flatten_then_embed_with_attention' , 'output_dim': 623}
        # terrain_height = { 'type': 'flatten_then_embed_with_attention_to_hidden' }

        # # history_deepmimic_lin_ang_vel = {'type': 'flatten'}

        # terrain_height = { 'type': 'flatten_then_embed' , 'output_dim': 918}
        # terrain_height = { 'type': 'flatten_then_embed_with_attention' , 'output_dim': 918}
        
        # # Include sensor observations for critic too
        # front_camera = { 'type': 'flatten' }
        # down_camera = { 'type': 'flatten' }
        # terrain_height = { 'type': 'downsample', 'factor': 2 }

    obs_proc_actor = ObsProcActor()
    obs_proc_critic = ObsProcCritic()

@configclass
class G1DeepMimicCfgProjHeightfieldPolicyCfg(G1DeepMimicPolicyCfg):

    @configclass
    class ObsProcActor:
        # torso = { 'type': 'identity' }
        # deepmimic = { 'type': 'identity' }
        # torso = {'type': 'identity'}
        # torso_real = {'type': 'flatten'}
        # deepmimic_lin_ang_vel = {'type': 'identity'}
        history_torso_real = {'type': 'flatten'}
        # history_deepmimic_lin_ang_vel = {'type': 'flatten'}
        # phase = {'type': 'identity'}
        # deepmimic_lin_ang_vel = {'type': 'identity'}
        # target_actions = {'type': 'identity'}

        # history_torso_real = {'type': 'flatten'}
        history_torso_xy_rel = {'type': 'flatten'}
        history_torso_yaw_rel = {'type': 'flatten'}

        # upper_body_joint_targets = {'type': 'identity'}

        target_joints = {'type': 'identity'}
        target_root_roll = {'type': 'identity'}
        target_root_pitch = {'type': 'identity'}

        # terrain_height = { 'type': 'flatten_then_embed_with_attention' , 'output_dim': 415}
        # terrain_height = { 'type': 'flatten_then_embed_with_attention' , 'output_dim': 401}
        terrain_height = { 'type': 'flatten_then_embed_with_attention_to_hidden' }

    @configclass
    class ObsProcCritic:
        torso = { 'type': 'identity' }
        deepmimic = { 'type': 'identity' }

        history_torso_real = {'type': 'flatten'}
        history_torso_xy_rel = {'type': 'flatten'}
        history_torso_yaw_rel = {'type': 'flatten'}

        target_joints = {'type': 'identity'}
        target_root_roll = {'type': 'identity'}
        target_root_pitch = {'type': 'identity'}

        # target_root_yaw = {'type': 'flatten'}
        # history_torso_xy = {'type': 'flatten'}
        # history_torso_yaw = {'type': 'flatten'}

        # phase = {'type': 'identity'}

        terrain_height = { 'type': 'flatten_then_embed_with_attention_to_hidden' }
        # terrain_height = { 'type': 'flatten_then_embed_with_attention_to_hidden' }

        # # history_deepmimic_lin_ang_vel = {'type': 'flatten'}

        # terrain_height = { 'type': 'flatten_then_embed' , 'output_dim': 918}
        # terrain_height = { 'type': 'flatten_then_embed_with_attention' , 'output_dim': 918}
        
        # # Include sensor observations for critic too
        # front_camera = { 'type': 'flatten' }
        # down_camera = { 'type': 'flatten' }
        # terrain_height = { 'type': 'downsample', 'factor': 2 }

    
    obs_proc_actor = ObsProcActor()
    obs_proc_critic = ObsProcCritic()

@configclass
class G1DeepMimicCfgRootHeightfieldPolicyCfg(G1DeepMimicPolicyCfg):

    @configclass
    class ObsProcActor:
        # torso = { 'type': 'identity' }
        # deepmimic = { 'type': 'identity' }
        # torso = {'type': 'identity'}
        # torso_real = {'type': 'flatten'}
        # deepmimic_lin_ang_vel = {'type': 'identity'}
        history_torso_real = {'type': 'flatten'}
        # history_deepmimic_lin_ang_vel = {'type': 'flatten'}
        # phase = {'type': 'identity'}
        # deepmimic_lin_ang_vel = {'type': 'identity'}
        # target_actions = {'type': 'identity'}

        # history_torso_real = {'type': 'flatten'}
        history_torso_xy_rel = {'type': 'flatten'}
        history_torso_yaw_rel = {'type': 'flatten'}

        # upper_body_joint_targets = {'type': 'identity'}

        # target_joints = {'type': 'identity'}
        # target_root_roll = {'type': 'identity'}
        # target_root_pitch = {'type': 'identity'}

        # terrain_height = { 'type': 'flatten_then_embed_with_attention' , 'output_dim': 415}
        # terrain_height = { 'type': 'flatten_then_embed_with_attention' , 'output_dim': 401}
        terrain_height = { 'type': 'flatten_then_embed_with_attention_to_hidden' }

    @configclass
    class ObsProcCritic:
        torso = { 'type': 'identity' }
        deepmimic = { 'type': 'identity' }

        history_torso_real = {'type': 'flatten'}
        history_torso_xy_rel = {'type': 'flatten'}
        history_torso_yaw_rel = {'type': 'flatten'}

        target_joints = {'type': 'identity'}
        target_root_roll = {'type': 'identity'}
        target_root_pitch = {'type': 'identity'}

        # target_root_yaw = {'type': 'flatten'}
        # history_torso_xy = {'type': 'flatten'}
        # history_torso_yaw = {'type': 'flatten'}

        # phase = {'type': 'identity'}

        terrain_height = { 'type': 'flatten_then_embed_with_attention_to_hidden' }
        # terrain_height = { 'type': 'flatten_then_embed_with_attention_to_hidden' }

        # # history_deepmimic_lin_ang_vel = {'type': 'flatten'}

        # terrain_height = { 'type': 'flatten_then_embed' , 'output_dim': 918}
        # terrain_height = { 'type': 'flatten_then_embed_with_attention' , 'output_dim': 918}
        
        # # Include sensor observations for critic too
        # front_camera = { 'type': 'flatten' }
        # down_camera = { 'type': 'flatten' }
        # terrain_height = { 'type': 'downsample', 'factor': 2 }

    
    obs_proc_actor = ObsProcActor()
    obs_proc_critic = ObsProcCritic()


@configclass
class G1DeepMimicCfgRootHeightfieldNoHistoryPolicyCfg(G1DeepMimicPolicyCfg):

    @configclass
    class ObsProcActor:
        # torso = { 'type': 'identity' }
        # deepmimic = { 'type': 'identity' }
        # torso = {'type': 'identity'}
        # torso_real = {'type': 'flatten'}
        # deepmimic_lin_ang_vel = {'type': 'identity'}
        history_torso_real = {'type': 'flatten'}
        # history_deepmimic_lin_ang_vel = {'type': 'flatten'}
        # phase = {'type': 'identity'}
        # deepmimic_lin_ang_vel = {'type': 'identity'}
        # target_actions = {'type': 'identity'}

        # history_torso_real = {'type': 'flatten'}
        torso_xy_rel = {'type': 'identity'}
        torso_yaw_rel = {'type': 'identity'}

        # upper_body_joint_targets = {'type': 'identity'}

        # target_joints = {'type': 'identity'}
        # target_root_roll = {'type': 'identity'}
        # target_root_pitch = {'type': 'identity'}

        # terrain_height = { 'type': 'flatten_then_embed_with_attention' , 'output_dim': 415}
        # terrain_height = { 'type': 'flatten_then_embed_with_attention' , 'output_dim': 401}
        # terrain_height = { 'type': 'flatten_then_embed_with_attention_to_hidden' }
        terrain_height_noisy = { 'type': 'flatten_then_embed_with_attention_to_hidden' }

    @configclass
    class ObsProcCritic:
        torso = { 'type': 'identity' }
        deepmimic = { 'type': 'identity' }

        history_torso_real = {'type': 'flatten'}
        history_torso_xy_rel = {'type': 'flatten'}
        history_torso_yaw_rel = {'type': 'flatten'}

        target_joints = {'type': 'identity'}
        target_root_roll = {'type': 'identity'}
        target_root_pitch = {'type': 'identity'}

        # target_root_yaw = {'type': 'flatten'}
        # history_torso_xy = {'type': 'flatten'}
        # history_torso_yaw = {'type': 'flatten'}

        # phase = {'type': 'identity'}

        # terrain_height = { 'type': 'flatten_then_embed_with_attention_to_hidden' }
        terrain_height_noisy = { 'type': 'flatten_then_embed_with_attention_to_hidden' }
        # terrain_height = { 'type': 'flatten_then_embed_with_attention_to_hidden' }

        # # history_deepmimic_lin_ang_vel = {'type': 'flatten'}

        # terrain_height = { 'type': 'flatten_then_embed' , 'output_dim': 918}
        # terrain_height = { 'type': 'flatten_then_embed_with_attention' , 'output_dim': 918}
        
        # # Include sensor observations for critic too
        # front_camera = { 'type': 'flatten' }
        # down_camera = { 'type': 'flatten' }
        # terrain_height = { 'type': 'downsample', 'factor': 2 }

    
    obs_proc_actor = ObsProcActor()
    obs_proc_critic = ObsProcCritic()

    # big deep helps (somewhat)
    actor_hidden_dims = [1024, 512, 256, 128]
    critic_hidden_dims = [1024, 512, 256, 128]

@configclass
class G1DeepMimicCfgRootPolicyCfg(G1DeepMimicPolicyCfg):

    @configclass
    class ObsProcActor:
        # torso = { 'type': 'identity' }
        # deepmimic = { 'type': 'identity' }
        # torso = {'type': 'identity'}
        # torso_real = {'type': 'flatten'}
        # deepmimic_lin_ang_vel = {'type': 'identity'}
        history_torso_real = {'type': 'flatten'}
        # history_deepmimic_lin_ang_vel = {'type': 'flatten'}
        # phase = {'type': 'identity'}
        # deepmimic_lin_ang_vel = {'type': 'identity'}
        # target_actions = {'type': 'identity'}

        # history_torso_real = {'type': 'flatten'}
        history_torso_xy_rel = {'type': 'flatten'}
        history_torso_yaw_rel = {'type': 'flatten'}

        # upper_body_joint_targets = {'type': 'identity'}

        # target_joints = {'type': 'identity'}
        # target_root_roll = {'type': 'identity'}
        # target_root_pitch = {'type': 'identity'}

        # terrain_height = { 'type': 'flatten_then_embed_with_attention' , 'output_dim': 415}
        # terrain_height = { 'type': 'flatten_then_embed_with_attention' , 'output_dim': 401}
        # terrain_height = { 'type': 'flatten_then_embed_with_attention_to_hidden' }
        # terrain_height_noisy = { 'type': 'flatten_then_embed_with_attention_to_hidden' }

    @configclass
    class ObsProcCritic:
        torso = { 'type': 'identity' }
        deepmimic = { 'type': 'identity' }

        history_torso_real = {'type': 'flatten'}
        history_torso_xy_rel = {'type': 'flatten'}
        history_torso_yaw_rel = {'type': 'flatten'}

        target_joints = {'type': 'identity'}
        target_root_roll = {'type': 'identity'}
        target_root_pitch = {'type': 'identity'}

        # target_root_yaw = {'type': 'flatten'}
        # history_torso_xy = {'type': 'flatten'}
        # history_torso_yaw = {'type': 'flatten'}

        # phase = {'type': 'identity'}

        # terrain_height = { 'type': 'flatten_then_embed_with_attention_to_hidden' }
        terrain_height_noisy = { 'type': 'flatten_then_embed_with_attention_to_hidden' }
        # terrain_height = { 'type': 'flatten_then_embed_with_attention_to_hidden' }

        # # history_deepmimic_lin_ang_vel = {'type': 'flatten'}

        # terrain_height = { 'type': 'flatten_then_embed' , 'output_dim': 918}
        # terrain_height = { 'type': 'flatten_then_embed_with_attention' , 'output_dim': 918}
        
        # # Include sensor observations for critic too
        # front_camera = { 'type': 'flatten' }
        # down_camera = { 'type': 'flatten' }
        # terrain_height = { 'type': 'downsample', 'factor': 2 }

    
    obs_proc_actor = ObsProcActor()
    obs_proc_critic = ObsProcCritic()

    # big deep helps (somewhat)
    actor_hidden_dims = [1024, 512, 256, 128]
    critic_hidden_dims = [1024, 512, 256, 128]

@configclass
class G1DeepMimicCfgRootHeightfieldNoHistoryWithProjJointsPolicyCfg(G1DeepMimicPolicyCfg):

    @configclass
    class ObsProcActor:
        # torso = { 'type': 'identity' }
        # deepmimic = { 'type': 'identity' }
        # torso = {'type': 'identity'}
        # torso_real = {'type': 'flatten'}
        # deepmimic_lin_ang_vel = {'type': 'identity'}
        history_torso_real = {'type': 'flatten'}
        # history_deepmimic_lin_ang_vel = {'type': 'flatten'}
        # phase = {'type': 'identity'}
        # deepmimic_lin_ang_vel = {'type': 'identity'}
        # target_actions = {'type': 'identity'}

        # history_torso_real = {'type': 'flatten'}
        torso_xy_rel = {'type': 'identity'}
        torso_yaw_rel = {'type': 'identity'}

        # upper_body_joint_targets = {'type': 'identity'}

        target_joints = {'type': 'embed_with_attention_to_hidden'}
        target_root_roll = {'type': 'embed_with_attention_to_hidden'}
        target_root_pitch = {'type': 'embed_with_attention_to_hidden'}

        # terrain_height = { 'type': 'flatten_then_embed_with_attention' , 'output_dim': 415}
        # terrain_height = { 'type': 'flatten_then_embed_with_attention' , 'output_dim': 401}
        # terrain_height = { 'type': 'flatten_then_embed_with_attention_to_hidden' }
        terrain_height_noisy = { 'type': 'flatten_then_embed_with_attention_to_hidden' }

    @configclass
    class ObsProcCritic:
        torso = { 'type': 'identity' }
        deepmimic = { 'type': 'identity' }

        history_torso_real = {'type': 'flatten'}
        history_torso_xy_rel = {'type': 'flatten'}
        history_torso_yaw_rel = {'type': 'flatten'}

        target_joints = {'type': 'identity'}
        target_root_roll = {'type': 'identity'}
        target_root_pitch = {'type': 'identity'}

        # target_root_yaw = {'type': 'flatten'}
        # history_torso_xy = {'type': 'flatten'}
        # history_torso_yaw = {'type': 'flatten'}

        # phase = {'type': 'identity'}

        # terrain_height = { 'type': 'flatten_then_embed_with_attention_to_hidden' }
        terrain_height_noisy = { 'type': 'flatten_then_embed_with_attention_to_hidden' }
        # terrain_height = { 'type': 'flatten_then_embed_with_attention_to_hidden' }

        # # history_deepmimic_lin_ang_vel = {'type': 'flatten'}

        # terrain_height = { 'type': 'flatten_then_embed' , 'output_dim': 918}
        # terrain_height = { 'type': 'flatten_then_embed_with_attention' , 'output_dim': 918}
        
        # # Include sensor observations for critic too
        # front_camera = { 'type': 'flatten' }
        # down_camera = { 'type': 'flatten' }
        # terrain_height = { 'type': 'downsample', 'factor': 2 }

    
    obs_proc_actor = ObsProcActor()
    obs_proc_critic = ObsProcCritic()

    # big deep helps (somewhat)
    actor_hidden_dims = [1024, 512, 256, 128]
    critic_hidden_dims = [1024, 512, 256, 128]



@configclass
class G1DeepMimicCfgRecurrent(G1DeepMimicPolicyCfg):
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = 'elu'

    rnn_type = 'lstm'
    rnn_hidden_size = 512
    rnn_num_layers = 1

@configclass
class G1DeepMimicCfgBigDeep(G1DeepMimicPolicyCfg):
    actor_hidden_dims = [1024, 512, 256, 128]
    critic_hidden_dims = [1024, 512, 256, 128]
    activation = 'elu'

@configclass
class G1DeepMimicCfgBigDeepRecurrent(G1DeepMimicCfgBigDeep):
    rnn_type = 'lstm'
    rnn_hidden_size = 1024
    rnn_num_layers = 2

@configclass
class G1DeepMimicAlgorithmCfg(LeggedRobotAlgorithmCfg):
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.0025
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1.e-3
    schedule = 'adaptive'
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.02
    max_grad_norm = 1.0


    bounds_loss_coef = 0.0005
    # used in bounds loss computation and teacher action clipping
    clip_actions_threshold = 8.0

    # for Dagger, to override below
    bc_loss_coef = 0.0
    # either a single policy path or a list of policy paths
    policy_to_clone: Union[str, List[str]] = None
    clip_teacher_actions: bool = True
    take_teacher_actions: bool = False

    use_multi_teacher: bool = False
    multi_teacher_select_obs_var: str = 'teacher_checkpoint_index'

@configclass
class G1DeepMimicCfgPPO(LeggedRobotCfgPPO):
    policy = G1DeepMimicPolicyCfg()
    algorithm = G1DeepMimicAlgorithmCfg()
    runner = LeggedRobotRunnerCfg(
        max_iterations = 100000,
        experiment_name = 'g1_deepmimic',
        save_interval = 10000,
        # policy_class_name = 'ActorCriticRecurrent',
    )

@configclass
class G1DeepmimicHeightFieldCfgPPO(G1DeepMimicCfgPPO):
    policy = G1DeepmimicHeightFieldPolicyCfg()

@configclass
class G1DeepMimicCfgProjHeightfieldPPO(G1DeepMimicCfgPPO):
    policy = G1DeepMimicCfgProjHeightfieldPolicyCfg()

@configclass
class G1DeepMimicCfgRootHeightfieldPPO(G1DeepMimicCfgPPO):
    policy = G1DeepMimicCfgRootHeightfieldPolicyCfg()

@configclass
class G1DeepMimicCfgRootHeightfieldNoHistoryWithProjJointsPPO(G1DeepMimicCfgPPO):
    # policy = G1DeepMimicCfgRootHeightfieldWithJointsPolicyCfg()
    policy = G1DeepMimicCfgRootHeightfieldNoHistoryWithProjJointsPolicyCfg()

@configclass
class G1DeepMimicCfgDagger(G1DeepMimicCfgPPO):

    @configclass
    class ObsProcActor:
        # torso = { 'type': 'identity' }
        # deepmimic = { 'type': 'identity' }
        # torso = {'type': 'identity'}
        # torso_real = {'type': 'flatten'}
        # deepmimic_lin_ang_vel = {'type': 'identity'}
        history_torso_real = {'type': 'flatten'}
        # history_deepmimic_lin_ang_vel = {'type': 'flatten'}
        # phase = {'type': 'identity'}
        # deepmimic_lin_ang_vel = {'type': 'identity'}
        # target_actions = {'type': 'identity'}

        # history_torso_real = {'type': 'flatten'}
        history_torso_xy_rel = {'type': 'flatten'}
        history_torso_yaw_rel = {'type': 'flatten'}

        upper_body_joint_targets = {'type': 'identity'}

        # target_joints = {'type': 'identity'}
        # target_root_roll = {'type': 'identity'}
        # target_root_pitch = {'type': 'identity'}

        # terrain_height = { 'type': 'flatten_then_embed_with_attention' , 'output_dim': 415}
        # terrain_height = { 'type': 'flatten_then_embed_with_attention' , 'output_dim': 401}
        # terrain_height = { 'type': 'flatten_then_embed_with_attention_to_hidden' }

    # @configclass
    # class ObsProcCritic:
    #     torso = { 'type': 'identity' }
    #     deepmimic = { 'type': 'identity' }
    #     history_torso_real = {'type': 'flatten'}
    #     phase = {'type': 'identity'}

    policy = G1DeepMimicPolicyCfg(obs_proc_actor = ObsProcActor())#, obs_proc_critic = ObsProcCritic())
    # policy = G1DeepMimicCfgRecurrent(obs_proc_actor = ObsProcActor())
    # policy = G1DeepMimicCfgBigDeep(obs_proc_actor = ObsProcActor())
    # policy = G1DeepMimicCfgBigDeepRecurrent(obs_proc_actor = ObsProcActor())

    algorithm = G1DeepMimicAlgorithmCfg(
        bc_loss_coef = 1.0,
        policy_to_clone = './logs/jit/policy.pt',
        learning_rate = 3.e-4,
        bounds_loss_coef = 0.0005,
    )

@configclass
class G1DeepmimicRootHeightfieldDagger(G1DeepMimicCfgDagger):
    policy = G1DeepMimicCfgRootHeightfieldPolicyCfg()

@configclass
class G1DeepmimicRootDagger(G1DeepMimicCfgDagger):
    policy = G1DeepMimicCfgRootPolicyCfg()

@configclass
class G1DeepmimicRootHeightfieldNoHistoryDagger(G1DeepMimicCfgDagger):
    policy = G1DeepMimicCfgRootHeightfieldNoHistoryPolicyCfg()

from legged_gym.utils.task_registry import task_registry
from legged_gym.envs.g1.g1_deepmimic import G1DeepMimic

task_registry.register( "g1_deepmimic", G1DeepMimic, G1DeepMimicCfg(), G1DeepMimicCfgPPO())
task_registry.register( "g1_deepmimic_mocap", G1DeepMimic, G1DeepMimicMocapCfg(), G1DeepMimicCfgPPO())
task_registry.register( "g1_deepmimic_dagger", G1DeepMimic, G1DeepMimicCfg(), G1DeepMimicCfgDagger())
task_registry.register( "g1_deepmimic_heightfield", G1DeepMimic, G1DeepMimicCfg(), G1DeepmimicHeightFieldCfgPPO())
task_registry.register( "g1_deepmimic_proj_heightfield", G1DeepMimic, G1DeepMimicCfg(), G1DeepMimicCfgProjHeightfieldPPO())
task_registry.register( "g1_deepmimic_root_heightfield", G1DeepMimic, G1DeepMimicCfg(), G1DeepMimicCfgRootHeightfieldPPO())
task_registry.register( "g1_deepmimic_root_heightfield_dagger", G1DeepMimic, G1DeepMimicCfg(), G1DeepmimicRootHeightfieldDagger())
task_registry.register( "g1_deepmimic_root_heightfield_no_history_dagger", G1DeepMimic, G1DeepMimicCfg(), G1DeepmimicRootHeightfieldNoHistoryDagger())
task_registry.register( "g1_deepmimic_root_heightfield_no_history_with_proj_joints_ppo", G1DeepMimic, G1DeepMimicCfg(), G1DeepMimicCfgRootHeightfieldNoHistoryWithProjJointsPPO())
task_registry.register( "g1_deepmimic_root_dagger", G1DeepMimic, G1DeepMimicCfg(), G1DeepmimicRootDagger())