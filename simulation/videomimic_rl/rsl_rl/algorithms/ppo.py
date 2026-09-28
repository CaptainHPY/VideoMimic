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
import torch.nn.functional as F
from contextlib import nullcontext

from rsl_rl.modules import ActorCritic, LatentStyleDiscriminator, MotionStyleDiscriminator
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
        use_discriminator=False,
        discriminator_sequence_length=8,
        discriminator_hidden_dim=256,
        discriminator_num_heads=4,
        discriminator_num_layers=2,
        discriminator_learning_rate=None,
        discriminator_updates_per_iter=1,
        discriminator_update_interval=1,
        discriminator_r1_coef=10.0,
        discriminator_mismatched_style_coef=1.0,
        discriminator_reward_coef=0.0,
        discriminator_recon_coef=0.0,
        discriminator_cycle_content_coef=0.0,
        discriminator_cycle_style_coef=0.0,
        discriminator_max_sequence_length=64,
        auxiliary_recon_loss_coef=0.0,
        auxiliary_cycle_content_loss_coef=0.0,
        auxiliary_cycle_style_loss_coef=0.0,
        auxiliary_style_consistency_loss_coef=0.0,
        auxiliary_style_contrastive_loss_coef=0.0,
        auxiliary_style_contrastive_margin=0.2,
        auxiliary_style_norm_loss_coef=0.0,
        style_gradient_monitor_interval=0,
        use_latent_discriminator=False,
        latent_discriminator_hidden_dim=256,
        latent_discriminator_learning_rate=1e-5,
        latent_discriminator_update_interval=1,
        latent_discriminator_mismatched_style_coef=1.0,
        latent_discriminator_ce_coef=1.0,
        latent_aux_adversarial_coef=0.0,
        latent_aux_ce_coef=1.0,
        latent_discriminator_warmup_steps=500,
        latent_aux_adversarial_ramp_steps=500,
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

        self.use_discriminator = bool(use_discriminator)
        self.discriminator_sequence_length = int(discriminator_sequence_length)
        self.discriminator_hidden_dim = int(discriminator_hidden_dim)
        self.discriminator_num_heads = int(discriminator_num_heads)
        self.discriminator_num_layers = int(discriminator_num_layers)
        self.discriminator_learning_rate = learning_rate if discriminator_learning_rate is None else discriminator_learning_rate
        self.discriminator_updates_per_iter = int(discriminator_updates_per_iter)
        self.discriminator_update_interval = int(discriminator_update_interval)
        if self.discriminator_update_interval <= 0:
            raise ValueError("discriminator_update_interval must be positive")
        self.discriminator_r1_coef = float(discriminator_r1_coef)
        self.discriminator_mismatched_style_coef = float(discriminator_mismatched_style_coef)
        if self.discriminator_mismatched_style_coef < 0.0:
            raise ValueError("discriminator_mismatched_style_coef must be non-negative")
        self.discriminator_reward_coef = float(discriminator_reward_coef)
        self.discriminator_recon_coef = float(discriminator_recon_coef)
        self.discriminator_cycle_content_coef = float(discriminator_cycle_content_coef)
        self.discriminator_cycle_style_coef = float(discriminator_cycle_style_coef)
        self.discriminator_max_sequence_length = int(discriminator_max_sequence_length)
        self.discriminator = None
        self.discriminator_optimizer = None
        self._discriminator_has_been_updated = False
        self._discriminator_synced = False
        self.last_discriminator_stats = {}
        self._warned_missing_discriminator_reward_fields = set()
        self.auxiliary_recon_loss_coef = float(auxiliary_recon_loss_coef)
        self.auxiliary_cycle_content_loss_coef = float(auxiliary_cycle_content_loss_coef)
        self.auxiliary_cycle_style_loss_coef = float(auxiliary_cycle_style_loss_coef)
        self.auxiliary_style_consistency_loss_coef = float(auxiliary_style_consistency_loss_coef)
        self.auxiliary_style_contrastive_loss_coef = float(auxiliary_style_contrastive_loss_coef)
        self.auxiliary_style_contrastive_margin = float(auxiliary_style_contrastive_margin)
        self.auxiliary_style_norm_loss_coef = float(auxiliary_style_norm_loss_coef)
        self.style_gradient_monitor_interval = int(style_gradient_monitor_interval)
        if self.auxiliary_style_contrastive_margin < 0.0:
            raise ValueError("auxiliary_style_contrastive_margin must be non-negative")
        if self.style_gradient_monitor_interval < 0:
            raise ValueError("style_gradient_monitor_interval must be non-negative")
        self.last_auxiliary_loss_stats = {}
        self.use_latent_discriminator = bool(use_latent_discriminator)
        self.latent_discriminator_hidden_dim = int(latent_discriminator_hidden_dim)
        self.latent_discriminator_learning_rate = float(latent_discriminator_learning_rate)
        self.latent_discriminator_update_interval = int(latent_discriminator_update_interval)
        self.latent_discriminator_mismatched_style_coef = float(latent_discriminator_mismatched_style_coef)
        self.latent_discriminator_ce_coef = float(latent_discriminator_ce_coef)
        self.latent_aux_adversarial_coef = float(latent_aux_adversarial_coef)
        self.latent_aux_ce_coef = float(latent_aux_ce_coef)
        self.latent_discriminator_warmup_steps = int(latent_discriminator_warmup_steps)
        self.latent_aux_adversarial_ramp_steps = int(latent_aux_adversarial_ramp_steps)
        if self.latent_discriminator_mismatched_style_coef < 0.0:
            raise ValueError("latent_discriminator_mismatched_style_coef must be non-negative")
        if self.latent_discriminator_update_interval <= 0:
            raise ValueError("latent_discriminator_update_interval must be positive")
        if self.latent_discriminator_warmup_steps < 0:
            raise ValueError("latent_discriminator_warmup_steps must be non-negative")
        if self.latent_aux_adversarial_ramp_steps < 0:
            raise ValueError("latent_aux_adversarial_ramp_steps must be non-negative")
        self.latent_discriminator = None
        self.latent_discriminator_optimizer = None
        self._latent_discriminator_synced = False
        self._latent_discriminator_minibatch_step = 0
        self._latent_stage2_start_iteration = None
        self._latent_stage2_iteration = 0
        self.last_latent_discriminator_stats = {}
        self.last_style_sensitivity_stats = {}
        self.last_style_gradient_stats = {}

        if self.bc_loss_coef > 0.0 or self.switch_to_rl_after > 0 and self.policy_to_clone is not None:
            if self.use_multi_teacher:
                self.load_teachers(self.policy_to_clone)
            else:
                self.bc_policy = self.load_policy_to_clone(self.policy_to_clone)
                self.bc_policy_loaded = True
        elif (self.bc_loss_coef > 0.0 or self.switch_to_rl_after > 0) and self.policy_to_clone is None:
            raise ValueError('policy_to_clone must be provided if bc_loss_coef > 0.0')

    def _sync_optimizer_learning_rates(self, current_learning_iteration):
        lr_scales = {}
        if hasattr(self.actor_critic, "get_optimizer_lr_scales"):
            lr_scales = self.actor_critic.get_optimizer_lr_scales(current_learning_iteration)
        elif hasattr(self.actor_critic, "get_style_lr_scale"):
            lr_scales["style"] = self.actor_critic.get_style_lr_scale(current_learning_iteration)

        for param_group in self.optimizer.param_groups:
            group_name = param_group.get("name", "base")
            param_group["lr"] = self.learning_rate * lr_scales.get(group_name, 1.0)

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
            discriminator_observations = {
                key: value.to(self.device) for key, value in infos['discriminator'].items()
            }
            if hasattr(self.actor_critic, "get_discriminator_counterfactual_observations"):
                counterfactual_observations = self.actor_critic.get_discriminator_counterfactual_observations(
                    self.transition.observations
                )
                for key, value in counterfactual_observations.items():
                    discriminator_observations[key] = value.to(self.device)
            self.transition.discriminator_observations = discriminator_observations
        if 'auxiliary' in infos:
            self.transition.auxiliary_observations = {
                key: value.to(self.device) for key, value in infos['auxiliary'].items()
            }

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
        if self.bc_policy_loaded:
            self.reset_teacher(dones)

    
    def compute_returns(self, last_obs, current_learning_iteration=0):
        self._update_discriminator(current_learning_iteration)
        self._add_discriminator_rewards()
        with torch.no_grad():
            last_values= self.actor_critic.evaluate(last_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    @staticmethod
    def _adv_loss(logits, target):
        targets = torch.full_like(logits, fill_value=float(target))
        return F.binary_cross_entropy_with_logits(logits, targets)

    @staticmethod
    def _r1_reg(d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(),
            inputs=x_in,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return 0.5 * grad_dout.pow(2).view(batch_size, -1).sum(1).mean(0)

    def _discriminator_attention_context(self):
        if self.discriminator_r1_coef <= 0.0 or not torch.cuda.is_available():
            return nullcontext()

        cuda_backends = getattr(torch.backends, "cuda", None)
        sdp_kernel = getattr(cuda_backends, "sdp_kernel", None) if cuda_backends is not None else None
        if sdp_kernel is None:
            return nullcontext()

        return sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)

    def _warn_missing_discriminator_reward_field(self, field_group, field_names):
        if field_group in self._warned_missing_discriminator_reward_fields:
            return
        self._warned_missing_discriminator_reward_fields.add(field_group)
        print(
            f"[PPO] Skipping discriminator {field_group} reward because none of these "
            f"rollout fields are available: {list(field_names)}"
        )

    @staticmethod
    def _get_first_available(discriminator_sequences, field_names):
        for field_name in field_names:
            if field_name in discriminator_sequences:
                return discriminator_sequences[field_name]
        return None

    @staticmethod
    def _masked_sequence_l2_error(prediction, target, padding_mask=None, valid_mask=None):
        difference = prediction - target
        if difference.dim() < 3:
            raise ValueError(
                f"Expected sequence tensors with shape (B, T, ...), got {tuple(difference.shape)}"
            )

        reduce_dims = tuple(range(2, difference.dim()))
        token_error = torch.linalg.vector_norm(difference, dim=reduce_dims)
        token_mask = torch.ones_like(token_error)

        if padding_mask is not None:
            token_mask = token_mask * (~padding_mask.bool()).float()

        if valid_mask is not None:
            valid = valid_mask
            if valid.dim() > 2:
                valid = valid.bool().flatten(start_dim=2).all(dim=-1)
            token_mask = token_mask * valid.float()

        return (token_error * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1.0)

    def _ensure_discriminator(self, discriminator_sequences):
        if not self.use_discriminator or discriminator_sequences is None:
            return
        if self.discriminator is None:
            motion_dim = discriminator_sequences["generated_motion"].shape[-1]
            style_label_count = discriminator_sequences.get("style_label_count")
            if style_label_count is not None:
                num_style_labels = int(style_label_count.max().item())
            elif "style_label" in discriminator_sequences:
                num_style_labels = int(discriminator_sequences["style_label"].max().item()) + 1
            else:
                raise ValueError(
                    "Conditional discriminator requires filename-derived style_label observations"
                )
            if num_style_labels < 2:
                raise ValueError(
                    "Conditional discriminator mismatch training requires at least two "
                    "filename-derived style labels"
                )
            self.discriminator = MotionStyleDiscriminator(
                motion_dim=motion_dim,
                num_style_labels=num_style_labels,
                hidden_dim=self.discriminator_hidden_dim,
                num_heads=self.discriminator_num_heads,
                num_layers=self.discriminator_num_layers,
                max_sequence_length=self.discriminator_max_sequence_length,
            ).to(self.device)
            self.discriminator_optimizer = optim.Adam(
                self.discriminator.parameters(),
                lr=self.discriminator_learning_rate,
                betas=(0.9, 0.99),
                weight_decay=1e-4,
            )
        if self.multi_gpu and not self._discriminator_synced:
            discriminator_params = [self.discriminator.state_dict()]
            dist.broadcast_object_list(discriminator_params, 0)
            self.discriminator.load_state_dict(discriminator_params[0])
            update_state = torch.tensor(
                [int(self._discriminator_has_been_updated)],
                dtype=torch.long,
                device=self.device,
            )
            dist.broadcast(update_state, 0)
            self._discriminator_has_been_updated = bool(update_state.item())
            self._discriminator_synced = True

    def load_discriminator_state_dict(self, discriminator_state_dict, discriminator_optimizer_state_dict=None):
        if "condition_proj.weight" in discriminator_state_dict:
            raise ValueError(
                "This checkpoint contains the former continuous-condition discriminator. "
                "Restart stage2 from a stage1 checkpoint to train the discrete-label discriminator."
            )
        motion_dim = discriminator_state_dict["motion_proj.weight"].shape[1]
        max_sequence_length = discriminator_state_dict["pos_embedding"].shape[1] - 1
        num_style_labels = discriminator_state_dict["head.3.weight"].shape[0]
        if num_style_labels < 2:
            raise ValueError(
                "This checkpoint contains the old single-head discriminator. "
                "Restart stage2 from a stage1 checkpoint to train the conditional discriminator."
            )
        self.discriminator = MotionStyleDiscriminator(
            motion_dim=motion_dim,
            num_style_labels=num_style_labels,
            hidden_dim=self.discriminator_hidden_dim,
            num_heads=self.discriminator_num_heads,
            num_layers=self.discriminator_num_layers,
            max_sequence_length=max_sequence_length,
        ).to(self.device)
        self.discriminator.load_state_dict(discriminator_state_dict)
        self._discriminator_has_been_updated = True
        self._discriminator_synced = False
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=self.discriminator_learning_rate,
            betas=(0.9, 0.99),
            weight_decay=1e-4,
        )
        if discriminator_optimizer_state_dict is not None:
            self.discriminator_optimizer.load_state_dict(discriminator_optimizer_state_dict)

    @staticmethod
    def _get_style_label_count(observations, style_labels):
        style_label_count = observations.get("style_label_count")
        if style_label_count is not None:
            return int(style_label_count.max().item())
        return int(style_labels.max().item()) + 1

    def _ensure_latent_discriminator(self, generated_latent, style_labels, style_label_count=None):
        if not self.use_latent_discriminator:
            return
        if self.latent_discriminator is None:
            num_style_labels = (
                int(style_label_count)
                if style_label_count is not None
                else int(style_labels.max().item()) + 1
            )
            if num_style_labels < 2:
                raise ValueError("Latent discriminator requires at least two style labels")
            self.latent_discriminator = LatentStyleDiscriminator(
                latent_dim=generated_latent.shape[-1],
                num_style_labels=num_style_labels,
                hidden_dim=self.latent_discriminator_hidden_dim,
            ).to(self.device)
            self.latent_discriminator_optimizer = optim.Adam(
                self.latent_discriminator.parameters(),
                lr=self.latent_discriminator_learning_rate,
                betas=(0.9, 0.99),
                weight_decay=1e-4,
            )
        if self.multi_gpu and not self._latent_discriminator_synced:
            latent_discriminator_params = [self.latent_discriminator.state_dict()]
            dist.broadcast_object_list(latent_discriminator_params, 0)
            self.latent_discriminator.load_state_dict(latent_discriminator_params[0])
            self._latent_discriminator_synced = True

    def load_latent_discriminator_state_dict(
        self,
        latent_discriminator_state_dict,
        latent_discriminator_optimizer_state_dict=None,
    ):
        latent_dim = latent_discriminator_state_dict["input_norm.weight"].shape[0]
        num_style_labels = latent_discriminator_state_dict["head.weight"].shape[0]
        hidden_dim = latent_discriminator_state_dict["backbone.0.weight"].shape[0]
        self.latent_discriminator = LatentStyleDiscriminator(
            latent_dim=latent_dim,
            num_style_labels=num_style_labels,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.latent_discriminator.load_state_dict(latent_discriminator_state_dict)
        self._latent_discriminator_synced = False
        self.latent_discriminator_optimizer = optim.Adam(
            self.latent_discriminator.parameters(),
            lr=self.latent_discriminator_learning_rate,
            betas=(0.9, 0.99),
            weight_decay=1e-4,
        )
        if latent_discriminator_optimizer_state_dict is not None:
            self.latent_discriminator_optimizer.load_state_dict(
                latent_discriminator_optimizer_state_dict
            )

    def _latent_adversarial_coef(self, current_learning_iteration):
        if self._latent_stage2_start_iteration is None:
            self._latent_stage2_start_iteration = int(current_learning_iteration)
        relative_iteration = self._latent_stage2_iteration
        if relative_iteration < self.latent_discriminator_warmup_steps:
            return 0.0, relative_iteration
        if self.latent_aux_adversarial_ramp_steps == 0:
            return self.latent_aux_adversarial_coef, relative_iteration
        ramp_iteration = relative_iteration - self.latent_discriminator_warmup_steps
        ramp_fraction = min(
            max(ramp_iteration / self.latent_aux_adversarial_ramp_steps, 0.0), 1.0
        )
        return self.latent_aux_adversarial_coef * ramp_fraction, relative_iteration

    def _sync_latent_schedule_state(self):
        if not self.multi_gpu:
            return
        start_iteration = (
            -1
            if self._latent_stage2_start_iteration is None
            else self._latent_stage2_start_iteration
        )
        schedule_state = torch.tensor(
            [self._latent_stage2_iteration, start_iteration],
            dtype=torch.long,
            device=self.device,
        )
        dist.broadcast(schedule_state, 0)
        self._latent_stage2_iteration = int(schedule_state[0].item())
        self._latent_stage2_start_iteration = (
            None if schedule_state[1].item() < 0 else int(schedule_state[1].item())
        )

    def _update_latent_discriminator(self, generated_latent, real_latent, style_labels):
        real_all_logits = self.latent_discriminator.forward_all_logits(real_latent.detach())
        real_logits = real_all_logits.gather(1, style_labels.unsqueeze(1)).squeeze(1)
        wrong_head_mask = ~F.one_hot(
            style_labels,
            num_classes=self.latent_discriminator.num_style_labels,
        ).bool()
        wrong_real_logits = real_all_logits[wrong_head_mask]
        fake_logits = self.latent_discriminator(generated_latent.detach(), style_labels)

        loss_real = self._adv_loss(real_logits, 1)
        loss_fake = self._adv_loss(fake_logits, 0)
        loss_wrong = self._adv_loss(wrong_real_logits, 0)
        loss_ce = F.cross_entropy(real_all_logits, style_labels)
        negative_loss = (
            loss_fake
            + self.latent_discriminator_mismatched_style_coef * loss_wrong
        ) / (1.0 + self.latent_discriminator_mismatched_style_coef)
        discriminator_loss = (
            loss_real
            + negative_loss
            + self.latent_discriminator_ce_coef * loss_ce
        )

        if torch.isfinite(discriminator_loss):
            self.latent_discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self._all_reduce_module_grads(self.latent_discriminator)
            nn.utils.clip_grad_norm_(self.latent_discriminator.parameters(), self.max_grad_norm)
            self.latent_discriminator_optimizer.step()
            self.latent_discriminator_optimizer.zero_grad(set_to_none=True)

        return {
            "D_loss": discriminator_loss.detach(),
            "D_real": loss_real.detach(),
            "D_fake": loss_fake.detach(),
            "D_wrong_heads": loss_wrong.detach(),
            "D_style_ce": loss_ce.detach(),
            "real_logits": real_logits.detach().mean(),
            "fake_logits": fake_logits.detach().mean(),
            "wrong_head_logits": wrong_real_logits.detach().mean(),
            "real_accuracy": (real_logits.detach() > 0).float().mean(),
            "fake_accuracy": (fake_logits.detach() < 0).float().mean(),
            "wrong_head_accuracy": (wrong_real_logits.detach() < 0).float().mean(),
            "style_accuracy": (real_all_logits.detach().argmax(dim=1) == style_labels).float().mean(),
        }

    def _compute_latent_actor_losses(self, generated_latent, style_labels):
        parameter_states = [
            parameter.requires_grad for parameter in self.latent_discriminator.parameters()
        ]
        for parameter in self.latent_discriminator.parameters():
            parameter.requires_grad_(False)
        try:
            all_logits = self.latent_discriminator.forward_all_logits(generated_latent)
            target_logits = all_logits.gather(1, style_labels.unsqueeze(1)).squeeze(1)
            adversarial_loss = self._adv_loss(target_logits, 1)
            style_ce_loss = F.cross_entropy(all_logits, style_labels)
        finally:
            for parameter, requires_grad in zip(
                self.latent_discriminator.parameters(), parameter_states
            ):
                parameter.requires_grad_(requires_grad)
        return adversarial_loss, style_ce_loss, {
            "G_adv_loss": adversarial_loss.detach(),
            "G_style_ce": style_ce_loss.detach(),
            "G_target_logits": target_logits.detach().mean(),
            "G_style_accuracy": (
                all_logits.detach().argmax(dim=1) == style_labels
            ).float().mean(),
        }

    def _get_style_actor_parameters(self):
        if not hasattr(self.actor_critic, "_get_optimizer_group_name"):
            return []
        return [
            parameter
            for name, parameter in self.actor_critic.named_parameters()
            if parameter.requires_grad
            and name.startswith("actor_stream_modules.")
            and self.actor_critic._get_optimizer_group_name(name) == "style"
        ]

    def _compute_style_gradient_stats(self, objectives):
        parameters = self._get_style_actor_parameters()
        if not parameters:
            return {}

        stats = {"parameter_count": float(sum(p.numel() for p in parameters))}
        reference_grads = None
        total_grads = [None] * len(parameters)
        for name, (objective, coefficient) in objectives.items():
            if not objective.requires_grad:
                continue
            grads = torch.autograd.grad(
                objective,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            norm_sq = objective.new_zeros((), dtype=torch.float32)
            for grad in grads:
                if grad is not None:
                    norm_sq = norm_sq + grad.detach().float().pow(2).sum()
            grad_norm = norm_sq.sqrt()
            stats[f"{name}_raw"] = float(grad_norm.item())
            stats[f"{name}_weighted"] = float(abs(coefficient) * grad_norm.item())

            if name == "ppo_surrogate":
                reference_grads = [
                    None if grad is None else grad.detach()
                    for grad in grads
                ]
            elif reference_grads is not None:
                dot = objective.new_zeros((), dtype=torch.float32)
                reference_norm_sq = objective.new_zeros((), dtype=torch.float32)
                for reference_grad, grad in zip(reference_grads, grads):
                    if reference_grad is None or grad is None:
                        continue
                    dot = dot + (reference_grad.float() * grad.detach().float()).sum()
                    reference_norm_sq = reference_norm_sq + reference_grad.float().pow(2).sum()
                denominator = reference_norm_sq.sqrt() * grad_norm
                stats[f"{name}_cosine_to_ppo"] = float(
                    (dot / denominator.clamp_min(1e-12)).item()
                )

            if coefficient != 0.0:
                for index, grad in enumerate(grads):
                    if grad is None:
                        continue
                    weighted_grad = grad.detach() * coefficient
                    total_grads[index] = (
                        weighted_grad
                        if total_grads[index] is None
                        else total_grads[index] + weighted_grad
                    )

        total_norm_sq = sum(
            (grad.float().pow(2).sum() for grad in total_grads if grad is not None),
            torch.zeros((), device=self.device),
        )
        stats["total_weighted"] = float(total_norm_sq.sqrt().item())
        return stats

    def _select_valid_discriminator_sequences(self, discriminator_sequences):
        if "padding_mask" not in discriminator_sequences and "valid_mask" not in discriminator_sequences:
            return {key: value for key, value in discriminator_sequences.items() if key not in ("padding_mask", "valid_mask")}
        valid = self._get_discriminator_sequence_validity(discriminator_sequences)
        if not valid.any():
            return None
        return {
            key: value[valid] if isinstance(value, torch.Tensor) and value.shape[0] == valid.shape[0] else value
            for key, value in discriminator_sequences.items()
            if key not in ("padding_mask", "valid_mask")
        }

    @staticmethod
    def _get_discriminator_sequence_validity(discriminator_sequences):
        num_sequences = discriminator_sequences["generated_motion"].shape[0]
        device = discriminator_sequences["generated_motion"].device
        valid = torch.ones(num_sequences, dtype=torch.bool, device=device)
        padding_mask = discriminator_sequences.get("padding_mask", None)
        if padding_mask is not None:
            valid &= ~padding_mask.bool().any(dim=1)
        valid_mask = discriminator_sequences.get("valid_mask", None)
        if valid_mask is not None:
            token_valid = valid_mask.bool()
            if token_valid.dim() > 2:
                token_valid = token_valid.flatten(start_dim=2).all(dim=-1)
            valid &= token_valid.all(dim=1)
        style_labels = discriminator_sequences.get("style_label")
        if style_labels is not None:
            labels = style_labels.long()
            if labels.dim() > 1:
                labels = labels.flatten(start_dim=1)
                valid &= (labels >= 0).all(dim=1)
                valid &= (labels == labels[:, :1]).all(dim=1)
            else:
                valid &= labels >= 0
        return valid

    @staticmethod
    def _get_sequence_style_labels(discriminator_sequences):
        style_labels = discriminator_sequences.get("style_label")
        if style_labels is None:
            raise ValueError("Discriminator sequence is missing style_label")
        labels = style_labels.long()
        if labels.dim() > 1:
            labels = labels.flatten(start_dim=1)
            if not (labels == labels[:, :1]).all():
                raise ValueError("A discriminator sequence contains multiple style labels")
            labels = labels[:, 0]
        return labels.reshape(-1)

    def _all_reduce_module_grads(self, module):
        if not self.multi_gpu or module is None:
            return
        all_grads_list = []
        for param in module.parameters():
            if param.grad is not None:
                all_grads_list.append(param.grad.view(-1))
        if len(all_grads_list) == 0:
            return
        all_grads = torch.cat(all_grads_list)
        dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
        offset = 0
        for param in module.parameters():
            if param.grad is not None:
                param.grad.data.copy_(
                    all_grads[offset : offset + param.numel()].view_as(param.grad.data) / self.multi_gpu_size
                )
                offset += param.numel()

    def _update_discriminator(self, current_learning_iteration):
        if not self.use_discriminator or self.storage is None:
            return
        discriminator_sequences = self.storage.get_discriminator_sequences(self.discriminator_sequence_length)
        if discriminator_sequences is None:
            return
        self._ensure_discriminator(discriminator_sequences)
        sequence_valid = self._get_discriminator_sequence_validity(discriminator_sequences)
        validity_stats = {
            "valid_sequence_fraction": sequence_valid.float().mean().item(),
            "num_valid_sequences": float(sequence_valid.sum().item()),
        }
        if (
            self._discriminator_has_been_updated
            and current_learning_iteration % self.discriminator_update_interval != 0
        ):
            self.last_discriminator_stats = {
                **self.last_discriminator_stats,
                **validity_stats,
                "D_updated": 0.0,
            }
            return
        discriminator_sequences = self._select_valid_discriminator_sequences(discriminator_sequences)
        if self.discriminator is None or discriminator_sequences is None:
            self.last_discriminator_stats = {**validity_stats, "D_updated": 0.0}
            return

        stats = {**validity_stats, "D_updated": 0.0}
        for _ in range(max(1, self.discriminator_updates_per_iter)):
            generated_motion = discriminator_sequences["generated_motion"].detach()
            style_motion = discriminator_sequences["style_motion"].detach().clone()
            style_motion.requires_grad_(self.discriminator_r1_coef > 0.0)
            style_labels = self._get_sequence_style_labels(discriminator_sequences)
            if self.discriminator.num_style_labels < 2:
                raise ValueError(
                    "Mismatched-label discriminator training requires at least two style labels"
                )

            with self._discriminator_attention_context():
                real_all_logits = self.discriminator.forward_all_logits(style_motion)
                real_logits = real_all_logits.gather(
                    1, style_labels.unsqueeze(1)
                ).squeeze(1)
                wrong_head_mask = ~F.one_hot(
                    style_labels,
                    num_classes=self.discriminator.num_style_labels,
                ).bool()
                mismatched_style_logits = real_all_logits[wrong_head_mask]
                max_mismatched_style_logits = real_all_logits.masked_fill(
                    ~wrong_head_mask, -torch.inf
                ).max(dim=1).values
                fake_logits = self.discriminator(
                    generated_motion, style_labels
                )
                loss_real = self._adv_loss(real_logits, 1)
                loss_fake = self._adv_loss(fake_logits, 0)
                loss_mismatched_style = self._adv_loss(mismatched_style_logits, 0)
                loss_reg = self._r1_reg(real_logits, style_motion) if self.discriminator_r1_coef > 0.0 else torch.zeros_like(loss_real)
                negative_loss = (
                    loss_fake
                    + self.discriminator_mismatched_style_coef * loss_mismatched_style
                ) / (1.0 + self.discriminator_mismatched_style_coef)
                discriminator_loss = loss_real + negative_loss + self.discriminator_r1_coef * loss_reg

            if not torch.isfinite(discriminator_loss):
                print("[PPO] Non-finite discriminator loss detected; skipping discriminator step.")
                continue

            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self._all_reduce_module_grads(self.discriminator)
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            self.discriminator_optimizer.step()
            self._discriminator_has_been_updated = True

            stats = {
                **validity_stats,
                "D_updated": 1.0,
                "D_loss": discriminator_loss.item(),
                "D_real": loss_real.item(),
                "D_fake": loss_fake.item(),
                "D_mismatched_style": loss_mismatched_style.item(),
                "D_reg": loss_reg.item(),
                "real_logits": real_logits.mean().item(),
                "fake_logits": fake_logits.mean().item(),
                "mismatched_style_logits": mismatched_style_logits.mean().item(),
                "max_mismatched_style_logits": max_mismatched_style_logits.mean().item(),
                "real_accuracy": (real_logits > 0).float().mean().item(),
                "fake_accuracy": (fake_logits < 0).float().mean().item(),
                "mismatched_style_accuracy": (mismatched_style_logits < 0).float().mean().item(),
                "max_mismatched_style_accuracy": (max_mismatched_style_logits < 0).float().mean().item(),
                "logit_gap": (real_logits.mean() - fake_logits.mean()).item(),
            }
        self.last_discriminator_stats = stats

    def _add_discriminator_rewards(self):
        if not self.use_discriminator or self.storage is None:
            return
        if (
            self.discriminator_reward_coef == 0.0
            and self.discriminator_recon_coef == 0.0
            and self.discriminator_cycle_content_coef == 0.0
            and self.discriminator_cycle_style_coef == 0.0
        ):
            return

        all_discriminator_sequences = self.storage.get_discriminator_sequences(self.discriminator_sequence_length)
        if all_discriminator_sequences is None:
            return
        self._ensure_discriminator(all_discriminator_sequences)
        if self.discriminator is None:
            return

        sequence_valid = self._get_discriminator_sequence_validity(all_discriminator_sequences)
        discriminator_sequences = self._select_valid_discriminator_sequences(all_discriminator_sequences)
        if discriminator_sequences is None:
            return

        generated_motion = discriminator_sequences["generated_motion"]
        content_motion = discriminator_sequences.get("content_motion", None)
        style_motion = discriminator_sequences.get("style_motion", None)
        content_token_motion = discriminator_sequences.get("content_token_motion", None)
        style_token_motion = discriminator_sequences.get("style_token_motion", None)
        style_labels = self._get_sequence_style_labels(discriminator_sequences)

        sequence_rewards = torch.zeros(generated_motion.shape[0], device=self.device)
        reward_stats = {}
        was_training = self.discriminator.training
        self.discriminator.eval()
        with torch.no_grad():
            if self.discriminator_reward_coef != 0.0:
                fake_logits = self.discriminator(
                    generated_motion,
                    style_labels,
                )
                adv_reward = -F.binary_cross_entropy_with_logits(
                    fake_logits,
                    torch.ones_like(fake_logits),
                    reduction="none",
                )
                sequence_rewards += self.discriminator_reward_coef * adv_reward
                reward_stats["G_adv_reward"] = adv_reward.mean().item()
                reward_stats["G_fake_logits"] = fake_logits.mean().item()

            if self.discriminator_recon_coef != 0.0:
                recon_field_names = (
                    "reconstruction_motion",
                    "generated_reconstruction_motion",
                    "gen_recon_motion",
                )
                reconstruction_motion = self._get_first_available(discriminator_sequences, recon_field_names)
                if reconstruction_motion is None:
                    self._warn_missing_discriminator_reward_field("reconstruction", recon_field_names)
                elif content_token_motion is None or content_token_motion.shape != reconstruction_motion.shape:
                    self._warn_missing_discriminator_reward_field("reconstruction_target", ("content_token_motion",))
                else:
                    recon_error = self._masked_sequence_l2_error(
                        reconstruction_motion,
                        content_token_motion,
                    )
                    sequence_rewards -= self.discriminator_recon_coef * recon_error
                    reward_stats["G_recon"] = recon_error.mean().item()

            if self.discriminator_cycle_content_coef != 0.0:
                cycle_content_field_names = (
                    "cycle_content_motion",
                    "generated_cycle_content_motion",
                    "gen_cycle_content_motion",
                )
                cycle_content_motion = self._get_first_available(discriminator_sequences, cycle_content_field_names)
                if cycle_content_motion is None:
                    self._warn_missing_discriminator_reward_field("cycle_content", cycle_content_field_names)
                elif content_token_motion is None or content_token_motion.shape != cycle_content_motion.shape:
                    self._warn_missing_discriminator_reward_field("cycle_content_target", ("content_token_motion",))
                else:
                    cycle_content_error = self._masked_sequence_l2_error(
                        cycle_content_motion,
                        content_token_motion,
                    )
                    sequence_rewards -= self.discriminator_cycle_content_coef * cycle_content_error
                    reward_stats["G_cyc-c"] = cycle_content_error.mean().item()

            if self.discriminator_cycle_style_coef != 0.0:
                cycle_style_motion = discriminator_sequences.get("cycle_style_motion", None)
                if cycle_style_motion is None:
                    self._warn_missing_discriminator_reward_field("cycle_style", ("cycle_style_motion",))
                elif style_token_motion is None or style_token_motion.shape != cycle_style_motion.shape:
                    self._warn_missing_discriminator_reward_field("cycle_style_target", ("style_token_motion",))
                else:
                    style_error = self._masked_sequence_l2_error(
                        cycle_style_motion,
                        style_token_motion,
                    )
                    sequence_rewards -= self.discriminator_cycle_style_coef * style_error
                    reward_stats["G_cyc-s"] = style_error.mean().item()

        if was_training:
            self.discriminator.train()
        if len(reward_stats) > 0:
            reward_stats["G_sequence_reward"] = sequence_rewards.mean().item()
            reward_stats["reward_valid_sequence_fraction"] = sequence_valid.float().mean().item()
            self.last_discriminator_stats.update(reward_stats)
        all_sequence_rewards = sequence_rewards.new_zeros(sequence_valid.shape[0])
        all_sequence_rewards[sequence_valid] = sequence_rewards
        self.storage.add_discriminator_sequence_rewards(all_sequence_rewards, self.discriminator_sequence_length)
    
    def switch_to_rl(self):
        self.bc_loss_coef = 0.0
        self.actor_loss_mul = 1.0

    def update(self, current_learning_iteration):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_bc_loss = 0
        mean_bounds_loss = 0
        mean_auxiliary_recon_loss = 0
        mean_auxiliary_cycle_content_loss = 0
        mean_auxiliary_cycle_style_loss = 0
        mean_auxiliary_style_consistency_loss = 0
        mean_auxiliary_style_contrastive_loss = 0
        mean_auxiliary_style_norm_loss = 0
        mean_latent_discriminator_stats = {}
        latent_discriminator_stat_counts = {}
        num_latent_discriminator_updates = 0
        style_consistency_stat_names = (
            "style_consistency_mean",
            "style_consistency_median",
            "style_consistency_p90",
            "style_consistency_max",
            "style_consistency_normalized",
            "style_consistency_pair_fraction",
            "style_consistency_distinct_reference_fraction",
            "style_contrastive_loss",
            "style_contrastive_active_fraction",
            "style_contrastive_pair_fraction",
            "style_positive_cosine_distance",
            "style_negative_cosine_distance",
            "generated_latent_rms",
            "generated_latent_rms_p90",
            "generated_latent_rms_max",
            "content_latent_rms",
            "real_latent_rms",
            "generated_to_target_latent_rms_ratio",
            "generated_residual_rms",
            "style_norm_anchor_loss",
        )
        mean_style_consistency_stats = {
            name: 0.0 for name in style_consistency_stat_names
        }
        mean_auxiliary_total_loss = 0
        use_style_auxiliary_losses = (
            self.auxiliary_recon_loss_coef != 0.0
            or self.auxiliary_cycle_content_loss_coef != 0.0
            or self.auxiliary_cycle_style_loss_coef != 0.0
            or self.auxiliary_style_consistency_loss_coef != 0.0
            or self.auxiliary_style_contrastive_loss_coef != 0.0
            or self.auxiliary_style_norm_loss_coef != 0.0
            or self.use_latent_discriminator
        ) and hasattr(self.actor_critic, "compute_style_auxiliary_losses")
        self.last_auxiliary_loss_stats = {}
        self.last_latent_discriminator_stats = {}
        self.last_style_sensitivity_stats = {}
        self.last_style_gradient_stats = {}
        if self.use_latent_discriminator:
            self._sync_latent_schedule_state()
            latent_adversarial_coef, latent_stage2_iteration = self._latent_adversarial_coef(
                current_learning_iteration
            )
        else:
            latent_adversarial_coef, latent_stage2_iteration = 0.0, 0
        sensitivity_recorded = False
        gradient_stats_recorded = False
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
            hid_states_batch, masks_batch, auxiliary_obs_batch in generator:


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

                auxiliary_recon_loss = torch.zeros_like(surrogate_loss)
                auxiliary_cycle_content_loss = torch.zeros_like(surrogate_loss)
                auxiliary_cycle_style_loss = torch.zeros_like(surrogate_loss)
                auxiliary_style_consistency_loss = torch.zeros_like(surrogate_loss)
                auxiliary_style_contrastive_loss = torch.zeros_like(surrogate_loss)
                auxiliary_style_norm_loss = torch.zeros_like(surrogate_loss)
                latent_adversarial_loss = torch.zeros_like(surrogate_loss)
                latent_style_ce_loss = torch.zeros_like(surrogate_loss)
                style_consistency_stats = {
                    name: torch.zeros_like(surrogate_loss)
                    for name in style_consistency_stat_names
                }
                if use_style_auxiliary_losses:
                    style_auxiliary_obs = auxiliary_obs_batch if auxiliary_obs_batch is not None else obs_batch
                    auxiliary_losses = self.actor_critic.compute_style_auxiliary_losses(
                        style_auxiliary_obs,
                        contrastive_margin=self.auxiliary_style_contrastive_margin,
                    )
                    auxiliary_recon_loss = auxiliary_losses.get("recon", auxiliary_recon_loss)
                    auxiliary_cycle_content_loss = auxiliary_losses.get("cycle_content", auxiliary_cycle_content_loss)
                    auxiliary_cycle_style_loss = auxiliary_losses.get("cycle_style", auxiliary_cycle_style_loss)
                    auxiliary_style_consistency_loss = auxiliary_losses.get("style_consistency", auxiliary_style_consistency_loss)
                    auxiliary_style_contrastive_loss = auxiliary_losses.get(
                        "style_contrastive", auxiliary_style_contrastive_loss
                    )
                    auxiliary_style_norm_loss = auxiliary_losses.get(
                        "style_norm_anchor", auxiliary_style_norm_loss
                    )
                    style_consistency_stats = {
                        name: auxiliary_losses.get(name, default_value)
                        for name, default_value in style_consistency_stats.items()
                    }
                    if self.use_latent_discriminator:
                        generated_latent = auxiliary_losses.get("latent_generated_feature")
                        real_latent = auxiliary_losses.get("latent_real_style_feature")
                        style_labels = auxiliary_losses.get("latent_style_label")
                        if generated_latent is None or real_latent is None or style_labels is None:
                            raise ValueError(
                                "Latent discriminator requires generated/style latents and style_label"
                            )
                        style_labels = style_labels.long().reshape(-1)
                        num_style_labels = self._get_style_label_count(
                            style_auxiliary_obs, style_labels
                        )
                        self._ensure_latent_discriminator(
                            generated_latent,
                            style_labels,
                            style_label_count=num_style_labels,
                        )
                        if (
                            style_labels.min() < 0
                            or style_labels.max() >= self.latent_discriminator.num_style_labels
                        ):
                            raise ValueError("Invalid style_label for latent discriminator")
                        latent_d_stats = {}
                        if (
                            self._latent_discriminator_minibatch_step
                            % self.latent_discriminator_update_interval
                            == 0
                        ):
                            latent_d_stats = self._update_latent_discriminator(
                                generated_latent,
                                real_latent,
                                style_labels,
                            )
                            num_latent_discriminator_updates += 1
                        self._latent_discriminator_minibatch_step += 1
                        latent_adversarial_loss, latent_style_ce_loss, latent_g_stats = self._compute_latent_actor_losses(
                            generated_latent,
                            style_labels,
                        )
                        for name, value in {**latent_d_stats, **latent_g_stats}.items():
                            mean_latent_discriminator_stats[name] = (
                                mean_latent_discriminator_stats.get(name, 0.0)
                                + float(value.item())
                            )
                            latent_discriminator_stat_counts[name] = (
                                latent_discriminator_stat_counts.get(name, 0) + 1
                            )

                    if not sensitivity_recorded and hasattr(
                        self.actor_critic, "compute_style_action_sensitivity"
                    ):
                        sensitivity_observations = dict(obs_batch)
                        if auxiliary_obs_batch is not None:
                            sensitivity_observations.update(auxiliary_obs_batch)
                        sensitivity_stats = self.actor_critic.compute_style_action_sensitivity(
                            sensitivity_observations,
                            sensitivity_observations.get("style_label"),
                            sensitivity_observations.get("style_content_id"),
                        )
                        self.last_style_sensitivity_stats = {
                            name: float(value.item())
                            for name, value in sensitivity_stats.items()
                        }
                        sensitivity_recorded = True
                auxiliary_total_loss = (
                    self.auxiliary_recon_loss_coef * auxiliary_recon_loss
                    + self.auxiliary_cycle_content_loss_coef * auxiliary_cycle_content_loss
                    + self.auxiliary_cycle_style_loss_coef * auxiliary_cycle_style_loss
                    + self.auxiliary_style_consistency_loss_coef * auxiliary_style_consistency_loss
                    + self.auxiliary_style_contrastive_loss_coef * auxiliary_style_contrastive_loss
                    + self.auxiliary_style_norm_loss_coef * auxiliary_style_norm_loss
                    + latent_adversarial_coef * latent_adversarial_loss
                    + self.latent_aux_ce_coef * latent_style_ce_loss
                )

                should_record_gradients = (
                    self.style_gradient_monitor_interval > 0
                    and current_learning_iteration % self.style_gradient_monitor_interval == 0
                    and not gradient_stats_recorded
                )
                if should_record_gradients:
                    self.last_style_gradient_stats = self._compute_style_gradient_stats({
                        "ppo_surrogate": (surrogate_loss, self.actor_loss_mul),
                        "recon": (auxiliary_recon_loss, self.auxiliary_recon_loss_coef),
                        "cycle_content": (
                            auxiliary_cycle_content_loss,
                            self.auxiliary_cycle_content_loss_coef,
                        ),
                        "cycle_style": (
                            auxiliary_cycle_style_loss,
                            self.auxiliary_cycle_style_loss_coef,
                        ),
                        "style_consistency": (
                            auxiliary_style_consistency_loss,
                            self.auxiliary_style_consistency_loss_coef,
                        ),
                        "style_contrastive": (
                            auxiliary_style_contrastive_loss,
                            self.auxiliary_style_contrastive_loss_coef,
                        ),
                        "style_norm_anchor": (
                            auxiliary_style_norm_loss,
                            self.auxiliary_style_norm_loss_coef,
                        ),
                        "latent_adversarial": (
                            latent_adversarial_loss,
                            latent_adversarial_coef,
                        ),
                        "latent_style_ce": (
                            latent_style_ce_loss,
                            self.latent_aux_ce_coef,
                        ),
                    })
                    gradient_stats_recorded = True

                loss = self.actor_loss_mul * (surrogate_loss
                    - self.entropy_coef * entropy_batch.mean()) \
                    + self.value_loss_coef * value_loss \
                    + self.bc_loss_coef * bc_loss \
                    + self.bounds_loss_coef * bounds_loss \
                    + auxiliary_total_loss

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
                mean_auxiliary_recon_loss += auxiliary_recon_loss.item()
                mean_auxiliary_cycle_content_loss += auxiliary_cycle_content_loss.item()
                mean_auxiliary_cycle_style_loss += auxiliary_cycle_style_loss.item()
                mean_auxiliary_style_consistency_loss += auxiliary_style_consistency_loss.item()
                mean_auxiliary_style_contrastive_loss += auxiliary_style_contrastive_loss.item()
                mean_auxiliary_style_norm_loss += auxiliary_style_norm_loss.item()
                for name, value in style_consistency_stats.items():
                    mean_style_consistency_stats[name] += value.item()
                mean_auxiliary_total_loss += auxiliary_total_loss.item()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_bc_loss /= num_updates
        mean_bounds_loss /= num_updates
        mean_auxiliary_recon_loss /= num_updates
        mean_auxiliary_cycle_content_loss /= num_updates
        mean_auxiliary_cycle_style_loss /= num_updates
        mean_auxiliary_style_consistency_loss /= num_updates
        mean_auxiliary_style_contrastive_loss /= num_updates
        mean_auxiliary_style_norm_loss /= num_updates
        mean_style_consistency_stats = {
            name: value / num_updates
            for name, value in mean_style_consistency_stats.items()
        }
        mean_auxiliary_total_loss /= num_updates
        if mean_latent_discriminator_stats:
            self.last_latent_discriminator_stats = {
                name: value / latent_discriminator_stat_counts[name]
                for name, value in mean_latent_discriminator_stats.items()
            }
            self.last_latent_discriminator_stats.update({
                "G_effective_coef": latent_adversarial_coef,
                "G_style_ce_coef": self.latent_aux_ce_coef,
                "G_weighted_adv_loss": latent_adversarial_coef
                * self.last_latent_discriminator_stats["G_adv_loss"],
                "G_weighted_style_ce": self.latent_aux_ce_coef
                * self.last_latent_discriminator_stats["G_style_ce"],
                "G_weighted_loss": (
                    latent_adversarial_coef
                    * self.last_latent_discriminator_stats["G_adv_loss"]
                    + self.latent_aux_ce_coef
                    * self.last_latent_discriminator_stats["G_style_ce"]
                ),
                "stage2_iteration": float(latent_stage2_iteration),
                "D_update_fraction": num_latent_discriminator_updates / num_updates,
                "warmup_active": float(
                    latent_stage2_iteration < self.latent_discriminator_warmup_steps
                ),
            })
        if self.use_latent_discriminator:
            self._latent_stage2_iteration += 1
        if use_style_auxiliary_losses:
            self.last_auxiliary_loss_stats = {
                "recon": mean_auxiliary_recon_loss,
                "weighted_recon": self.auxiliary_recon_loss_coef * mean_auxiliary_recon_loss,
                "cycle_content": mean_auxiliary_cycle_content_loss,
                "weighted_cycle_content": self.auxiliary_cycle_content_loss_coef * mean_auxiliary_cycle_content_loss,
                "cycle_style": mean_auxiliary_cycle_style_loss,
                "weighted_cycle_style": self.auxiliary_cycle_style_loss_coef * mean_auxiliary_cycle_style_loss,
                "style_consistency": mean_auxiliary_style_consistency_loss,
                "weighted_style_consistency": self.auxiliary_style_consistency_loss_coef * mean_auxiliary_style_consistency_loss,
                "style_contrastive": mean_auxiliary_style_contrastive_loss,
                "weighted_style_contrastive": self.auxiliary_style_contrastive_loss_coef * mean_auxiliary_style_contrastive_loss,
                "style_norm_anchor": mean_auxiliary_style_norm_loss,
                "weighted_style_norm_anchor": self.auxiliary_style_norm_loss_coef * mean_auxiliary_style_norm_loss,
                **mean_style_consistency_stats,
                "total": mean_auxiliary_total_loss,
            }
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
