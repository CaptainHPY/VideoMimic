#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR/../.."

LOAD_RUN=20250410_063030_g1_deepmimic
POLICY_TO_CLONE=logs/g1_deepmimic/${LOAD_RUN}

torchrun --nproc-per-node 2 legged_gym/scripts/train.py \
--multi_gpu \
--task=g1_deepmimic_root_heightfield_no_history_dagger --headless --env.terrain.n_rows=1 --num_envs=4096 --wandb_note "videomimic_stage_3" \
--env.deepmimic.human_motion_source=resources/data_config/human_motion_list_123_motions.yaml --train.algorithm.learning_rate=1e-3 --train.algorithm.schedule=fixed --env.deepmimic.upsample_data=True --env.deepmimic.use_human_videos=True --env.deepmimic.link_pos_error_threshold=0.3 --train.runner.save_interval=500 --train.runner.max_iterations=5000 --env.deepmimic.respawn_z_offset=0.1 --env.terrain.cast_mesh_to_heightfield=False --env.deepmimic.truncate_rollout_length=500 --train.runner.load_model_strict=False --env.deepmimic.use_amass=False --train.algorithm.policy_to_clone=${POLICY_TO_CLONE} \
--env.rewards.scales.contact_no_vel=0.0