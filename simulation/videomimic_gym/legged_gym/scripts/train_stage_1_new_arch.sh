#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR/../.."

torchrun --nproc-per-node 2 legged_gym/scripts/train.py \
--task=g1_deepmimic --multi_gpu --headless \
--env.deepmimic.use_amass=True \
--env.deepmimic.stage=1 \
--env.deepmimic.use_style_conditioning=False \
--train.policy.stage=1 \
--train.policy.freeze_style_branch=False \
--env.terrain.n_rows=16 \
--num_envs=2048 \
--wandb_note "videomimic_new_arch_stage_1" \
--env.deepmimic.truncate_rollout_length=500 \
--env.noise.add_noise=True \
--env.deepmimic.link_pos_error_threshold=0.7 \
--env.rewards.scales.action_rate=-1.0 \
--env.rewards.scales.dof_pos_limits=-5.0 \
--env.deepmimic.amass_terrain_difficulty=1 \
--env.domain_rand.p_gain_rand=True \
--env.domain_rand.d_gain_rand=True \
--env.domain_rand.push_robots=False \
--env.domain_rand.control_delays=False \
--env.noise.offset_scales.gravity=0.02 \
--env.noise.offset_scales.dof_pos=0.005 \
--env.deepmimic.randomize_terrain_offset=True \
--env.asset.use_alt_files=True \
--env.noise.init_noise_scales.root_xy=0.1 \
--env.noise.init_noise_scales.root_z=0.02 \
--env.domain_rand.randomize_base_mass=False \
--env.asset.terminate_after_large_feet_contact_forces=False \
--env.noise.init_noise_scales.dof_pos=0.01
