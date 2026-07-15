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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import os

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils.jit import try_load_jit_model

class PPO:
    actor_critic: ActorCritic

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        multi_gpu=False,
        multi_gpu_rank=0,
        multi_gpu_size=-1,
        bc_loss_coef=0.0,
        bounds_loss_coef=0.0,
        clip_actions_threshold=100.0, # N.B. clipping isnt actually applied but used for bounds loss and teacher action clipping
        policy_to_clone=None,
        clip_teacher_actions=False,
        take_teacher_actions=False,
        use_multi_teacher=False,
        multi_teacher_select_obs_var='teacher_checkpoint_index',
        switch_to_rl_after=-1,
    ):

        self.device = device
        self.multi_gpu = multi_gpu
        self.multi_gpu_rank = multi_gpu_rank
        self.multi_gpu_size = multi_gpu_size

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        if hasattr(self.actor_critic, "get_optimizer_param_groups"):
            self.optimizer = optim.Adam(self.actor_critic.get_optimizer_param_groups(learning_rate))
        else:
            self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.clip_actions_threshold = clip_actions_threshold
        self.bounds_loss_coef = bounds_loss_coef

        self.bc_loss_coef = bc_loss_coef
        self.bc_policy_loaded = False
        # set to true to prevent different generator loops
        self.has_teacher_actions = True #self.bc_loss_coef > 0.0
        self.actor_loss_mul = 1.0 - self.bc_loss_coef
        self.policy_to_clone = policy_to_clone
        self.clip_teacher_actions = clip_teacher_actions
        self.take_teacher_actions = take_teacher_actions
        self.use_multi_teacher = use_multi_teacher
        self.multi_teacher_select_obs_var = multi_teacher_select_obs_var
        self.switch_to_rl_after = switch_to_rl_after

        if self.bc_loss_coef > 0.0 or self.switch_to_rl_after > 0 and self.policy_to_clone is not None:
            if self.use_multi_teacher:
                self.load_teachers(self.policy_to_clone)
            else:
                self.bc_policy = self.load_policy_to_clone(self.policy_to_clone)
                self.bc_policy_loaded = True
        elif (self.bc_loss_coef > 0.0 or self.switch_to_rl_after > 0) and self.policy_to_clone is None:
            raise ValueError('policy_to_clone must be provided if bc_loss_coef > 0.0')

    def _sync_optimizer_learning_rates(self, current_learning_iteration):
        style_lr_scale = 1.0
        if hasattr(self.actor_critic, "get_style_lr_scale"):
            style_lr_scale = self.actor_critic.get_style_lr_scale(current_learning_iteration)

        for param_group in self.optimizer.param_groups:
            group_name = param_group.get("name", "")
            if group_name == "style":
                param_group["lr"] = self.learning_rate * style_lr_scale
            else:
                param_group["lr"] = self.learning_rate

    def init_storage(self, num_envs, num_transitions_per_env, obs_shapes, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, obs_shapes, action_shape, has_teacher_actions=self.has_teacher_actions, device=self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    # def act(self, obs, critic_obs):
    def act(self, obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs

        actions_to_take = self.transition.actions

        if self.bc_loss_coef > 0.0 and self.bc_policy_loaded:
            self.transition.teacher_actions = self.get_teacher_actions(obs).detach()
            # disabled teacher value calculation for now
            self.transition.teacher_values = torch.zeros_like(self.transition.values)
            if self.take_teacher_actions:
                actions_to_take = self.transition.teacher_actions
            # actions_to_take = self.transition.teacher_actions * self.teacher_actions_mask + actions_to_take * (1-self.teacher_actions_mask)
        elif self.has_teacher_actions:
            self.transition.teacher_actions = torch.zeros_like(self.transition.actions)
            self.transition.teacher_values = torch.zeros_like(self.transition.values)
    

        return actions_to_take
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
        if 'discriminator' in infos:
            self.transition.discriminator_observations = {
                key: value.to(self.device) for key, value in infos['discriminator'].items()
            }

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
        if self.bc_policy_loaded:
            self.reset_teacher(dones)

    
    def compute_returns(self, last_obs):
        last_values= self.actor_critic.evaluate(last_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)
    
    def switch_to_rl(self):
        self.bc_loss_coef = 0.0
        self.actor_loss_mul = 1.0

    def update(self, current_learning_iteration):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_bc_loss = 0
        mean_bounds_loss = 0
        self._sync_optimizer_learning_rates(current_learning_iteration)
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        
        if current_learning_iteration == self.switch_to_rl_after:
            self.switch_to_rl()

        # for obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
        #     old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:
        for obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, teacher_actions_batch, teacher_values_batch, \
            hid_states_batch, masks_batch in generator:


                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)
                    if self.multi_gpu:
                        # compute the average KL over all GPUs
                        dist.all_reduce(kl_mean, op=dist.ReduceOp.SUM)
                        kl_mean = kl_mean / self.multi_gpu_size

                    # only do LR updates on process 0 in multi GPU scenario
                    if not self.multi_gpu or self.multi_gpu_rank == 0:
                        # if kl_mean > self.desired_kl * 2.0:
                        #     self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        # elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        #     self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        factor = 1.2
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / factor)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * factor)

                        self._sync_optimizer_learning_rates(current_learning_iteration)

                    # broadcast computed learning rate from process 0 (where calculation took place) to the rest 
                    if self.multi_gpu:
                        learning_rate_tensor = torch.tensor([self.learning_rate], device=self.device)
                        dist.broadcast(learning_rate_tensor, 0)
                        self.learning_rate = learning_rate_tensor.item()
                        self._sync_optimizer_learning_rates(current_learning_iteration)


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # BC loss
                if self.bc_loss_coef > 0.0 and self.bc_policy_loaded:

                    mu_student = self.actor_critic.action_mean

                    # todo -- possibly sample from student so BC can set sigma student as well
                    sigma_student = self.actor_critic.action_std

                    # sigma_teacher = torch.ones_like(sigma_student) * self.bc_policy.std.detach()
                    sigma_teacher = self.get_teacher_std(obs_batch)

                    if self.clip_teacher_actions:
                        teacher_actions_batch = torch.clip(teacher_actions_batch, -self.clip_actions_threshold, self.clip_actions_threshold)

                    bc_loss = (teacher_actions_batch - mu_student).pow(2).sum(dim=-1).mean() + (sigma_student - sigma_teacher).pow(2).sum(dim=-1).mean()# + (teacher_values_batch - value_batch).pow(2).mean()

                else:
                    bc_loss = torch.zeros_like(surrogate_loss)


                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()
                
                # clip_value = 6.0
                if self.bounds_loss_coef > 0.0:
                    clip_actions_value = self.clip_actions_threshold
                    clipped_actions = torch.clamp(mu_batch, -clip_actions_value, clip_actions_value)
                    bounds_loss = torch.sum(torch.abs(clipped_actions - mu_batch), dim=-1).mean()
                else:
                    bounds_loss = torch.zeros_like(surrogate_loss)

                loss = self.actor_loss_mul * (surrogate_loss
                    - self.entropy_coef * entropy_batch.mean()) \
                    + self.value_loss_coef * value_loss \
                    + self.bc_loss_coef * bc_loss \
                    + self.bounds_loss_coef * bounds_loss

                if not torch.isfinite(loss):
                    print("[PPO] Non-finite loss detected; skipping optimizer step for this mini-batch.")
                    continue

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()

                if self.multi_gpu:
                    # from RL-Games
                    # batch allreduce ops: see https://github.com/entity-neural-network/incubator/pull/220
                    all_grads_list = []
                    for param in self.actor_critic.parameters():
                        if param.grad is not None:
                            all_grads_list.append(param.grad.view(-1))
                    all_grads = torch.cat(all_grads_list)
                    # sum grads on each gpu
                    dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
                    offset = 0
                    for param in self.actor_critic.parameters():
                        if param.grad is not None:
                            # copy data back from shared buffer
                            param.grad.data.copy_(
                                all_grads[offset : offset + param.numel()].view_as(param.grad.data) / self.multi_gpu_size
                            )
                            offset += param.numel()

                has_nonfinite_grad = False
                for param in self.actor_critic.parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        has_nonfinite_grad = True
                        break

                if has_nonfinite_grad:
                    print("[PPO] Non-finite gradient detected; skipping optimizer step for this mini-batch.")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Protect action std from NaN/Inf pollution in long runs.
                with torch.no_grad():
                    if hasattr(self.actor_critic, 'std'):
                        std = self.actor_critic.std.data
                        if not torch.isfinite(std).all():
                            print("[PPO] actor_critic.std became non-finite; resetting invalid entries to 1.0.")
                            std = torch.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
                        self.actor_critic.std.data.copy_(std.clamp_(1e-3, 10.0))

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_bc_loss += bc_loss.item()
                mean_bounds_loss += bounds_loss.item()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_bc_loss /= num_updates
        mean_bounds_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_bc_loss, mean_bounds_loss
    
    def load_policy_to_clone(self, file_path):
        """ Load the policy to clone (for Dagger) from a file path.

        Args:
            file_path (str): The path to the file containing the policy to clone.

        Basically, policy to clone is expected to be a jitted model.
        If we give it a regular model, we need to jit it first.
        For a jitted checkpoint, we only accept the direct path to the checkpoint file.
        For a regular checkpoint, we accept the log directory, and will automatically load the latest checkpoint (or else a path to a specific checkpoint).
        """
        num_attempts = 5
        print(f'Loading policy for BC from {file_path}...')
        for attempt in range(num_attempts):
            try:
                # basically, policy to clone is expected to be a jitted model.
                # if we give it a regular model, we need to jit it first.
                # is_jit, policy = try_load_jit_model(file_path)
                # if is_jit:
                #     policy = policy.to(self.device)
                # else:
                #     if not os.path.isfile(file_path):
                #         from rsl_rl.utils.utils import get_checkpoint_path
                #         file_path = get_checkpoint_path(file_path)
                #     loaded_dict = torch.load(file_path)
                #     policy = ActorCritic(
                #         self.actor_critic.env_obs_shapes,
                #         self.actor_critic.env_num_actions,
                #         **loaded_dict['policy_cfg']
                #     )
                #     from rsl_rl.utils.jit import get_torchscript_model
                #     policy.load_state_dict(loaded_dict['model_state_dict'])
                #     policy = get_torchscript_model(policy).to(self.device)

                if not os.path.isfile(file_path):
                    from rsl_rl.utils.utils import get_checkpoint_path
                    file_path = get_checkpoint_path(file_path, multi_gpu=self.multi_gpu, multi_gpu_rank=self.multi_gpu_rank)
                loaded_dict = torch.load(file_path)
                policy = ActorCritic(
                    self.actor_critic.env_obs_shapes,
                    self.actor_critic.env_num_actions,
                    **loaded_dict['policy_cfg']
                ).to(self.device)
                policy.load_state_dict(loaded_dict['model_state_dict'])
                print(f'Successfully loaded policy for BC from {file_path}!')
                return policy
            except Exception as exc:
                print(f'Exception {exc} when trying to load policy for BC from {file_path}...')
                wait_sec = 2 ** attempt
                print(f'Waiting {wait_sec} before trying again...')
                import time
                time.sleep(wait_sec)

    def load_teachers(self, teacher_checkpoints):
        self.teacher_checkpoints = teacher_checkpoints
        self.bc_policies = []
        for teacher_checkpoint in teacher_checkpoints:
            cheeckpoint_joined = os.path.join('logs/g1_deepmimic', teacher_checkpoint)
            self.bc_policies.append(self.load_policy_to_clone(cheeckpoint_joined))
    
    def get_teacher_actions(self, obs):
        if self.use_multi_teacher:
            selected_policy_index = obs[self.multi_teacher_select_obs_var]
            actions = torch.zeros(obs[self.multi_teacher_select_obs_var].shape[0], self.actor_critic.env_num_actions, device=self.device)
            for i in range(len(self.bc_policies)):
                actions[selected_policy_index == i] = self.bc_policies[i].act({k: v[selected_policy_index == i] for k, v in obs.items()})
            return actions
        else:
            return self.bc_policy.act(obs)
        
    def get_teacher_std(self, obs):
        if self.use_multi_teacher:
            selected_policy_index = obs[self.multi_teacher_select_obs_var]
            stds = torch.zeros(obs[self.multi_teacher_select_obs_var].shape[0], self.actor_critic.env_num_actions, device=self.device)
            for i in range(len(self.bc_policies)):
                stds[selected_policy_index == i] = self.bc_policies[i].std.detach()
            return stds
        else:
            return self.bc_policy.std.detach()
    
    def reset_teacher(self, dones):
        if self.bc_policy_loaded:
            if self.use_multi_teacher:
                for i in range(len(self.bc_policies)):
                    self.bc_policies[i].reset(dones)
            else:
                self.bc_policy.reset(dones)
