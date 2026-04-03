#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR/../.."

python legged_gym/scripts/play.py --task=g1_deepmimic_dagger --env.deepmimic.use_amass=True --env.deepmimic.amass_terrain_difficulty=1 --env.deepmimic.amass_replay_data_path="lafan_walk_and_dance/*.pkl" \
--load_run 20250414_170842_g1_deepmimic --num_envs 1 --env.deepmimic.use_human_videos=False --headless  --env.deepmimic.respawn_z_offset=0.1 --env.deepmimic.link_pos_error_threshold=10.0 --train.algorithm.bc_loss_coef=0.0 --env.viser.enable=True