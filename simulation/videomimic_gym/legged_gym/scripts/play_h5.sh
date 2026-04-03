#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR/../.."

python legged_gym/scripts/play.py --task=g1_deepmimic_root_heightfield_no_history_dagger \
--load_run 20250502_124756_g1_deepmimic --num_envs 1 --headless --env.viser.enable=True --env.deepmimic.use_amass=False --env.deepmimic.use_human_videos=True \
--env.deepmimic.allow_h5_direct_source=True --env.deepmimic.human_motion_source="../data/mocap_xia/*.h5" --env.terrain.auto_flat_if_mesh_missing=True --env.terrain.flat_terrain_size=20.0 --env.terrain.flat_terrain_z=-0.05 --env.terrain.cast_mesh_to_heightfield=False --env.deepmimic.n_append=0 --env.deepmimic.respawn_z_offset=0.1 --env.deepmimic.link_pos_error_threshold=10.0 --train.algorithm.bc_loss_coef=0.0 --train.runner.load_model_strict=True