# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv
import torch.distributed as dist


class OnPolicyRunner:
    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu',
                 multi_gpu=False):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # flag to load model with strict=False
        # useful for model surgery (e.g. adding a new head)
        self.load_model_strict = self.cfg.get("load_model_strict", False)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.multi_gpu = multi_gpu

        if self.multi_gpu:

            # os.environ["MASTER_ADDR"] = "127.0.0.1"
            # os.environ["MASTER_PORT"] = "29500"

            self.multi_gpu_rank = int(os.getenv("LOCAL_RANK", "0"))
            self.global_gpu_rank = int(os.getenv("RANK", "0"))
            self.multi_gpu_size = int(os.getenv("WORLD_SIZE", "1"))
            dist.init_process_group("nccl", rank=self.global_gpu_rank, world_size=self.multi_gpu_size)

            assert self.device == 'cuda:' + str(self.multi_gpu_rank) # check it was set correctly
        else:
            # for default args
            self.multi_gpu_rank = 0
            self.global_gpu_rank = 0
            self.multi_gpu_size = -1

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        actor_critic: ActorCritic = actor_critic_class( self.env.get_obs_shapes(),
                                                        self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, multi_gpu=self.multi_gpu, multi_gpu_rank=self.multi_gpu_rank, multi_gpu_size=self.multi_gpu_size, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        env_obs_shapes = self.env.get_obs_shapes()
        used_obs_keys = set(self.policy_cfg["obs_proc_actor"].keys()) | set(self.policy_cfg["obs_proc_critic"].keys()) | set((self.alg.multi_teacher_select_obs_var,))
        # Keep index-like debug signals in rollout storage so actor-side NaN diagnostics
        # can map a bad sample back to the source replay clip.
        for debug_key in ("clip_index", "episode_indices", "episode_index", "index_within_episode"):
            if debug_key in env_obs_shapes:
                used_obs_keys.add(debug_key)
        used_obs_shapes = {k: env_obs_shapes[k] for k in used_obs_keys}

        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, used_obs_shapes, [self.env.num_actions])

        # Log
        # only log on the first GPU
        if self.multi_gpu and self.global_gpu_rank != 0:
            self.disable_logs = True
        else:
            self.disable_logs = False

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        # Console episode log controls (TensorBoard logging keeps all keys)
        self.console_ep_log = self.cfg.get("console_ep_log", True)
        self.console_ep_log_keys = self.cfg.get("console_ep_log_keys", None)
        self.console_ep_log_prefixes = self.cfg.get("console_ep_log_prefixes", None)
        self.console_ep_log_exclude = set(self.cfg.get("console_ep_log_exclude", []))
        self.console_ep_log_max_items = int(self.cfg.get("console_ep_log_max_items", 6))
        self.console_ep_log_sort = self.cfg.get("console_ep_log_sort", "name")

        _ = self.env.reset()


    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.cfg['use_wandb'] and not self.disable_logs:
            import wandb
            wandb.init(project="rsl_rl", config=self.cfg, name=self.cfg['run_name'], notes=self.cfg['wandb_note'] if 'wandb_note' in self.cfg else '', entity=self.cfg['wandb_entity'] if 'wandb_entity' in self.cfg else None)
            # Start tensorboard logging

        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        # privileged_obs = self.env.get_privileged_observations()
        # critic_obs = privileged_obs if privileged_obs is not None else obs
        # obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        obs = {k: v.to(self.device) for k, v in obs.items()}

        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.multi_gpu: 
            # ensure all parameters are in sync -- broadcast from seed 0
            torch.cuda.set_device(self.multi_gpu_rank)
            print(f"Broadcasting initial parameters from GPU {self.multi_gpu_rank}")
            model_params = [self.alg.actor_critic.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.alg.actor_critic.load_state_dict(model_params[0])
            iteration = torch.tensor(
                [self.current_learning_iteration],
                dtype=torch.long,
                device=self.device,
            )
            dist.broadcast(iteration, 0)
            self.current_learning_iteration = int(iteration.item())


        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, infos = self.env.step(actions)
                    obs = {k: v.to(self.device) for k, v in obs.items()}
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

            # Learning step
            start = stop
            self.alg.compute_returns(obs, current_learning_iteration=it)
            
            mean_value_loss, mean_surrogate_loss, mean_bc_loss, mean_bounds_loss = self.alg.update(it)
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0 and not self.disable_logs:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):

        if self.disable_logs:
            return

        collection_size = self.num_steps_per_env * self.env.num_envs

        if self.multi_gpu:
            collection_size *= self.multi_gpu_size

        self.tot_timesteps += collection_size
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']
        wandb_metrics = {}

        def add_scalar(key, value, step):
            self.writer.add_scalar(key, value, step)
            if self.cfg['use_wandb']:
                if isinstance(value, torch.Tensor):
                    value = value.detach().mean().item()
                wandb_metrics[key] = value

        ep_string = f''
        if locs['ep_infos']:
            all_keys = set()
            for ep_info in locs['ep_infos']:
                all_keys.update(ep_info.keys())
            ep_means = {}
            for key in all_keys:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if key in ep_info:
                        if not isinstance(ep_info[key], torch.Tensor):
                            ep_info[key] = torch.Tensor([ep_info[key]])
                        if len(ep_info[key].shape) == 0:
                            ep_info[key] = ep_info[key].unsqueeze(0)
                        infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                add_scalar(key, value, locs['it'])
                ep_means[key] = value.item()

            if self.console_ep_log:
                if isinstance(self.console_ep_log_keys, (list, tuple)) and len(self.console_ep_log_keys) > 0:
                    selected_items = [(k, ep_means[k]) for k in self.console_ep_log_keys if k in ep_means]
                else:
                    selected_items = list(ep_means.items())

                if isinstance(self.console_ep_log_prefixes, (list, tuple)) and len(self.console_ep_log_prefixes) > 0:
                    selected_items = [(k, v) for k, v in selected_items if any(k.startswith(p) for p in self.console_ep_log_prefixes)]

                if self.console_ep_log_exclude:
                    selected_items = [(k, v) for k, v in selected_items if k not in self.console_ep_log_exclude]

                if self.console_ep_log_sort == 'value_desc':
                    selected_items.sort(key=lambda x: x[1], reverse=True)
                elif self.console_ep_log_sort == 'value_asc':
                    selected_items.sort(key=lambda x: x[1])
                else:
                    selected_items.sort(key=lambda x: x[0])

                hidden_count = max(0, len(selected_items) - self.console_ep_log_max_items)
                selected_items = selected_items[:self.console_ep_log_max_items]

                for key, value in selected_items:
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
                if hidden_count > 0:
                    ep_string += f"""{f'Mean episode ...':>{pad}} +{hidden_count} more metrics\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        add_scalar('learning_iteration', locs['it'], locs['it'])
        add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        add_scalar('Loss/bc', locs['mean_bc_loss'], locs['it'])
        add_scalar('Loss/bounds', locs['mean_bounds_loss'], locs['it'])
        add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        discriminator_stats = getattr(self.alg, "last_discriminator_stats", {})
        for key, value in discriminator_stats.items():
            add_scalar(f'Discriminator/{key}', value, locs['it'])
        latent_discriminator_stats = getattr(self.alg, "last_latent_discriminator_stats", {})
        for key, value in latent_discriminator_stats.items():
            add_scalar(f'LatentDiscriminator/{key}', value, locs['it'])
        auxiliary_loss_stats = getattr(self.alg, "last_auxiliary_loss_stats", {})
        for key, value in auxiliary_loss_stats.items():
            add_scalar(f'Auxiliary/{key}', value, locs['it'])
        style_gradient_stats = getattr(self.alg, "last_style_gradient_stats", {})
        for key, value in style_gradient_stats.items():
            add_scalar(f'GradientNorm/style/{key}', value, locs['it'])
        add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        add_scalar('Perf/total_fps', fps, locs['it'])
        add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if hasattr(self.alg.actor_critic, "get_and_reset_generated_feature_contribution_stats"):
            contribution_stats = self.alg.actor_critic.get_and_reset_generated_feature_contribution_stats()
            contribution_values = []
            mixture_share_values = []
            for module_name, stats in contribution_stats.items():
                contribution_values.append(stats['contribution_ratio'])
                mixture_share_values.append(stats['mixture_share'])
                add_scalar(f'Network/generated_feature_contribution/{module_name}', stats['contribution_ratio'], locs['it'])
                add_scalar(f'Network/generated_feature_mixture_share/{module_name}', stats['mixture_share'], locs['it'])
                add_scalar(f'Network/generated_feature_contribution_forwards/{module_name}', stats['num_forwards'], locs['it'])
            if len(contribution_values) > 0:
                add_scalar('Network/generated_feature_contribution_mean', sum(contribution_values) / len(contribution_values), locs['it'])
                add_scalar('Network/generated_feature_mixture_share_mean', sum(mixture_share_values) / len(mixture_share_values), locs['it'])
        style_sensitivity_stats = getattr(self.alg, "last_style_sensitivity_stats", {})
        for key, value in style_sensitivity_stats.items():
            add_scalar(f'Network/{key}', value, locs['it'])
        if hasattr(self.alg.actor_critic, "get_decoder_gate_stats"):
            for module_name, stats in self.alg.actor_critic.get_decoder_gate_stats().items():
                add_scalar(f'Network/content_decoder_gate/{module_name}', stats['content'], locs['it'])
                add_scalar(f'Network/style_decoder_gate/{module_name}', stats['style'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)


        actor_attention = None
        critic_attention = None

        def _get_attention(input_nets, obs_name):
            for input_net in input_nets:
                if input_net is None:
                    continue
                if obs_name in input_net.heads:
                    return input_net.heads[obs_name].state_dict()['attention']
                if obs_name in input_net.extra_proj_heads:
                    return input_net.extra_proj_heads[obs_name].state_dict()['attention']
            return None

        actor_input_nets = [
            getattr(self.alg.actor_critic, name, None)
            for name in ('actor_shared_input_net', 'actor_content_input_net', 'actor_style_input_net', 'actor_input_net')
        ]
        critic_input_nets = [
            getattr(self.alg.actor_critic, name, None)
            for name in ('critic_shared_input_net', 'critic_content_input_net', 'critic_style_input_net', 'critic_input_net')
        ]

        actor_attention = _get_attention(actor_input_nets, 'terrain_height')
        if actor_attention is not None:
            add_scalar('Network/attention_terrain_height_actor', torch.abs(actor_attention).mean(), locs['it'])
            add_scalar('Network/max_attention_terrain_height_actor', actor_attention.max(), locs['it'])

        critic_attention = _get_attention(critic_input_nets, 'terrain_height')
        if critic_attention is not None:
            add_scalar('Network/attention_terrain_height_critic', torch.abs(critic_attention).mean(), locs['it'])
            add_scalar('Network/max_attention_terrain_height_critic', critic_attention.max(), locs['it'])

        if self.cfg['use_wandb'] and len(wandb_metrics) > 0:
            import wandb
            wandb.log(wandb_metrics, step=locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Bounds loss:':>{pad}} {locs['mean_bounds_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Mean bc loss:':>{pad}} {locs['mean_bc_loss']:.4f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        checkpoint = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'policy_cfg': self.policy_cfg,
            'infos': infos,
            }
        if getattr(self.alg, "discriminator", None) is not None:
            checkpoint['discriminator_state_dict'] = self.alg.discriminator.state_dict()
        if getattr(self.alg, "discriminator_optimizer", None) is not None:
            checkpoint['discriminator_optimizer_state_dict'] = self.alg.discriminator_optimizer.state_dict()
        if getattr(self.alg, "latent_discriminator", None) is not None:
            checkpoint['latent_discriminator_state_dict'] = self.alg.latent_discriminator.state_dict()
        if getattr(self.alg, "latent_discriminator_optimizer", None) is not None:
            checkpoint['latent_discriminator_optimizer_state_dict'] = self.alg.latent_discriminator_optimizer.state_dict()
        checkpoint['latent_stage2_start_iteration'] = getattr(
            self.alg, "_latent_stage2_start_iteration", None
        )
        checkpoint['latent_stage2_iteration'] = getattr(
            self.alg, "_latent_stage2_iteration", 0
        )
        torch.save(checkpoint, path)
        if self.cfg['use_wandb'] and not self.disable_logs:
            import wandb
            wandb.save(path)

    def load(self, path, load_optimizer=True):
        if self.multi_gpu and self.multi_gpu_rank != 0:
            print(f"Skipping loading model from {path} on GPU {self.multi_gpu_rank}")
            return

        print(f"\n\n\n\n\n\nLoading model from {path}\n\n\n\n\n\n")
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'], strict=self.load_model_strict)
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        if 'discriminator_state_dict' in loaded_dict and hasattr(self.alg, "load_discriminator_state_dict"):
            discriminator_optimizer_state_dict = (
                loaded_dict.get('discriminator_optimizer_state_dict', None) if load_optimizer else None
            )
            self.alg.load_discriminator_state_dict(
                loaded_dict['discriminator_state_dict'],
                discriminator_optimizer_state_dict,
            )
        if (
            'latent_discriminator_state_dict' in loaded_dict
            and hasattr(self.alg, "load_latent_discriminator_state_dict")
        ):
            latent_discriminator_optimizer_state_dict = (
                loaded_dict.get('latent_discriminator_optimizer_state_dict', None)
                if load_optimizer else None
            )
            self.alg.load_latent_discriminator_state_dict(
                loaded_dict['latent_discriminator_state_dict'],
                latent_discriminator_optimizer_state_dict,
            )
        if 'latent_stage2_start_iteration' in loaded_dict:
            self.alg._latent_stage2_start_iteration = loaded_dict[
                'latent_stage2_start_iteration'
            ]
        if 'latent_stage2_iteration' in loaded_dict:
            self.alg._latent_stage2_iteration = loaded_dict['latent_stage2_iteration']
        elif self.alg._latent_stage2_start_iteration is not None:
            self.alg._latent_stage2_iteration = max(
                0,
                int(loaded_dict['iter'])
                - int(self.alg._latent_stage2_start_iteration),
            )
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
