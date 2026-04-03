#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR/../.."

python legged_gym/scripts/play.py --task=g1_deepmimic --env.deepmimic.use_amass=True --env.deepmimic.amass_terrain_difficulty=1 --env.deepmimic.amass_replay_data_path="lafan_single_walk/*.pkl" --load_run 20250410_063030_g1_deepmimic --num_envs 1 --env.deepmimic.use_human_videos=False --headless  --env.deepmimic.respawn_z_offset=0.1 --env.deepmimic.link_pos_error_threshold=10.0 --env.viser.enable=True