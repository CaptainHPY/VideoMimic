import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import parse_unknown_args
import torch

def train(args, unknown):
    # Parse unknown args into env and train override dicts
    env_overrides, train_overrides = parse_unknown_args(unknown)

    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_overrides=env_overrides)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_overrides=train_overrides)
    print(f"Train config: {train_cfg}")

    if hasattr(train_cfg.policy, 're_init_std') and train_cfg.policy.re_init_std:
        print(f"After loading checkpoint, re-initializing std with value {train_cfg.policy.init_noise_std}")
        ppo_runner.alg.actor_critic.re_init_std(train_cfg.policy.init_noise_std)

    if hasattr(train_cfg.policy, 're_init_style_decoder_gate') and train_cfg.policy.re_init_style_decoder_gate:
        gate_logit = train_cfg.policy.style_decoder_gate_init
        print(f"After loading checkpoint, re-initializing style decoder gate logits with value {gate_logit}")
        ppo_runner.alg.actor_critic.re_init_style_decoder_gates(gate_logit)

    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=False)

if __name__ == '__main__':
    args, unknown = get_args()
    train(args, unknown)
