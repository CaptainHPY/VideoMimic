#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR/../.."

# Stage 2 starts from a stage-1 checkpoint and enables content/style conditioning.
# Replace this with the stage-1 run directory you want to initialize from.
LOAD_RUN=20260630_225252_g1_deepmimic

torchrun --nproc-per-node 2 legged_gym/scripts/train.py \
--task=g1_deepmimic --multi_gpu --headless \
--load_run ${LOAD_RUN} --resume \
--train.runner.load_model_strict=False \
--env.deepmimic.allow_h5_direct_source=True \
--env.terrain.auto_flat_if_mesh_missing=True \
--env.terrain.flat_terrain_size=20.0 \
--env.terrain.flat_terrain_z=0 \
--env.deepmimic.human_motion_source="../data/mocap_xia/*.h5" \
--env.deepmimic.use_amass=False \
--env.deepmimic.upsample_data=True \
--env.deepmimic.use_human_videos=True \
--env.deepmimic.stage=2 \
--env.deepmimic.use_style_conditioning=True \
--env.deepmimic.style_pair_relation=style \
--train.policy.stage=2 \
--train.policy.freeze_style_branch=False \
--train.policy.content_lr_scale=0.2 \
--train.policy.style_lr_scale=1.0 \
--train.policy.head_lr_scale=0.2 \
--train.policy.base_lr_scale=0.2 \
--train.policy.style_lr_warmup_steps=5000 \
--env.terrain.n_rows=16 \
--num_envs=1024 \
--max_iterations=300000 \
--train.runner.save_interval=5000 \
--wandb_note "videomimic_new_arch_stage_2_style_transfer" \
--env.deepmimic.truncate_rollout_length=500 \
--env.noise.add_noise=True \
--env.deepmimic.init_velocities=False \
--env.deepmimic.link_pos_error_threshold=0.5 \
--env.rewards.scales.action_rate=-25.0 \
--env.deepmimic.amass_terrain_difficulty=2 \
--env.domain_rand.p_gain_rand=False \
--env.domain_rand.d_gain_rand=False \
--env.domain_rand.push_robots=False \
--env.domain_rand.control_delays=False \
--env.domain_rand.control_delay_min=0 \
--env.domain_rand.control_delay_max=5 \
--env.noise.offset_scales.gravity=0.02 \
--env.noise.offset_scales.dof_pos=0.005 \
--env.deepmimic.randomize_terrain_offset=False \
--env.asset.use_alt_files=True \
--env.noise.init_noise_scales.root_xy=0.0 \
--env.noise.init_noise_scales.root_z=0.0 \
--env.domain_rand.randomize_base_mass=False \
--env.asset.terminate_after_large_feet_contact_forces=True \
--env.noise.init_noise_scales.dof_pos=0.0 \
--env.sim.substeps=2 \
--env.sim.physx.num_position_iterations=8 \
--env.sim.physx.num_velocity_iterations=1 \
--env.sim.physx.max_gpu_contact_pairs=33554432 \
--env.sim.physx.default_buffer_size_multiplier=10 \
--train.algorithm.learning_rate=2e-5 \
--train.algorithm.schedule=fixed \
--train.algorithm.use_discriminator=True \
--train.algorithm.discriminator_sequence_length=8 \
--train.algorithm.discriminator_learning_rate=1e-4 \
--train.algorithm.discriminator_updates_per_iter=1 \
--train.algorithm.discriminator_r1_coef=10.0 \
--train.algorithm.discriminator_reward_coef=0.0 \
--train.algorithm.discriminator_recon_coef=0.0 \
--train.algorithm.discriminator_cycle_content_coef=0.0 \
--train.algorithm.discriminator_cycle_style_coef=0.0 \
--train.algorithm.auxiliary_recon_loss_coef=0.0006 \
--train.algorithm.auxiliary_cycle_content_loss_coef=0.0004 \
--train.algorithm.auxiliary_cycle_style_loss_coef=0.0001
