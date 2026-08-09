#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR/../.."

# Replace this with the stage-2 run directory you want to inspect.
LOAD_RUN=20260630_225252_g1_deepmimic

python legged_gym/scripts/play.py \
--task=g1_deepmimic \
--load_run ${LOAD_RUN} \
--num_envs 1 \
--headless \
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
--env.terrain.n_rows=1 \
--env.deepmimic.truncate_rollout_length=500 \
--env.noise.add_noise=False \
--env.deepmimic.init_velocities=False \
--env.deepmimic.link_pos_error_threshold=10.0 \
--env.domain_rand.push_robots=False \
--env.domain_rand.control_delays=False \
--env.viser.enable=True
