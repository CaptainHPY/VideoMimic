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

import numpy as np
import os
import json

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from typing import Dict, List, Optional, Tuple


class EmbedMLP(nn.Module):

    def __init__(self, input_size, output_size, bias=True):
        super(EmbedMLP, self).__init__()
        self.input_proc = nn.Linear(input_size, output_size, bias=bias)
    
    def forward(self, x):
        return self.input_proc(x)

class EmbedMLPWithAttention(nn.Module):
    def __init__(self, input_size, output_size):
        super(EmbedMLPWithAttention, self).__init__()
        self.attention = nn.Parameter(torch.zeros(output_size,))
        if isinstance(input_size, tuple):
            assert len(input_size) == 1, f"Can only embed 1d observation, but obs {input_size} has shape {input_size}"
            input_size = input_size[0]
        self.embed = EmbedMLP(input_size, output_size)

    def forward(self, x):
        out = self.embed(x)
        return out * self.attention

class FlattenThenEmbedMLP(nn.Module):
    def __init__(self, input_size, output_size, bias=True):
        super(FlattenThenEmbedMLP, self).__init__()
        self.flatten = nn.Flatten()
        # embed size is product of input size
        input_size_flattened = int(np.prod(input_size))
        self.embed = EmbedMLP(input_size_flattened, output_size, bias=bias)

    def forward(self, x):
        flattened = self.flatten(x)
        return self.embed(flattened)

class FlattenThenEmbedMLPWithAttention(nn.Module):
    def __init__(self, input_size, output_size):
        super(FlattenThenEmbedMLPWithAttention, self).__init__()
        self.attention = nn.Parameter(torch.zeros(output_size,))
        self.embed = FlattenThenEmbedMLP(input_size, output_size)

    def forward(self, x):
        out = self.embed(x)
        return out * self.attention

obs_proc_types = {
    "identity": nn.Identity,
    "flatten": nn.Flatten,
    "embed": EmbedMLP,
    'flatten_then_embed': FlattenThenEmbedMLP,
    'flatten_then_embed_with_attention': FlattenThenEmbedMLPWithAttention,
    'flatten_then_embed_with_attention_to_hidden': FlattenThenEmbedMLPWithAttention,
    'embed_with_attention_to_hidden': EmbedMLPWithAttention,
}

class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.LayerNorm(num_features, elementwise_affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        # x: (B, C)  s: (B, S)
        h = self.fc(s)
        gamma, beta = torch.chunk(h, 2, dim=1)
        out = self.norm(x)
        return (1 + gamma) * out + beta


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, q_in, k_in, v_in, mask=None):
        # q_in, k_in, v_in: (B, N, C)
        B, N, C = q_in.shape
        def reshape(x):
            return x.view(B, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)

        q = reshape(self.q(q_in))
        k = reshape(self.k(k_in))
        v = reshape(self.v(v_in))

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0,2,1,3).contiguous().view(B, N, C)
        out = self.proj(out)
        return out


class TransformerModulator(nn.Module):
    def __init__(self, dim, num_heads=4, num_parts=6):
        super().__init__()
        self.cross_attn = CrossAttention(dim, num_heads=num_heads)
        self.linear = nn.Linear(dim * num_parts, dim * num_parts)
        self.ff = nn.Sequential(nn.LayerNorm(dim * num_parts), nn.Linear(dim * num_parts, dim * num_parts), nn.GELU(), nn.Linear(dim * num_parts, dim * num_parts))

    def forward(self, query_feat, key_feat, value_feat):
        # query/key/value: (B, P, C)
        attended = self.cross_attn(query_feat, key_feat, value_feat)
        attended = attended.view(attended.size(0), -1)
        value_flat = value_feat.view(value_feat.size(0), -1)
        out = self.linear(value_flat) + attended
        out = out + self.ff(out)
        return out


class TransformerDecoder(nn.Module):
    def __init__(self, dim, num_heads=4, num_layers=2, num_parts=6):
        super().__init__()
        self.num_parts = num_parts
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        style_vec_dim = dim * num_parts
        self.adain = AdaIN(style_vec_dim, style_vec_dim)

    def forward(self, target_tokens, memory_tokens, style_signal: Optional[torch.Tensor] = None):
        # target_tokens: (B, P, C)
        # memory_tokens: (B, P, C)
        if style_signal is not None and self.adain is not None:
            B, P, C = target_tokens.shape
            flat = target_tokens.reshape(B, -1)  # (B, P*C)
            flat = self.adain(flat, style_signal)
            target_tokens = flat.reshape(B, P, C)
        return self.decoder(target_tokens, memory_tokens)


class StyleTransformer(nn.Module):
    """A compact Style-like backbone that embeds input into body-part tokens,
    applies encoder blocks + a PSM-style modulator + a TransformerDecoder, and
    returns a flattened feature vector of same dim as input.

    When both cnt and sty are provided, the two streams are embedded and encoded
    separately before modulation, matching the content/style split used in
    class.py.
    """
    def __init__(self, input_dim, num_parts=6, part_dim=None, num_enc_layers=2, num_dec_layers=3, num_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.num_parts = num_parts
        part_dim = max(part_dim or max(16, input_dim // max(1, num_parts)), num_heads)
        part_dim = ((part_dim + num_heads - 1) // num_heads) * num_heads
        self.part_dim = part_dim

        # project input to per-part embeddings
        self.part_proj = nn.ModuleList([nn.Linear(input_dim, part_dim) for _ in range(num_parts)])

        # small transformer encoder applied per-part sequence (we'll use a shared nn.TransformerEncoderLayer)
        encoder_layer = nn.TransformerEncoderLayer(d_model=part_dim, nhead=num_heads, dim_feedforward=part_dim*4, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_enc_layers)

        # last-block encoder (encoder_IN) similar to class.py's final IN-enabled block
        self.encoder_IN = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # decoder query and motion decoder used to generate the style-transferred motion tokens
        self.decoder_query = nn.Parameter(torch.randn(num_parts, part_dim))
        self.decoder = TransformerDecoder(part_dim, num_heads=num_heads, num_layers=num_dec_layers, num_parts=num_parts)
        self.content_decoder_gate = nn.Parameter(torch.tensor(-4.0))

        # learnable style tokens, one token per body part, matching class.py
        self.learnable_style_token = nn.Parameter(torch.randn(1, num_parts, part_dim))

        # PSM-like modulator
        self.modulator = TransformerModulator(part_dim, num_heads=num_heads, num_parts=num_parts)

        # project decoded motion tokens back into the feature space used by policy heads
        self.motion_proj = nn.Linear(part_dim * num_parts, input_dim)

        # final proj
        self.out_proj = nn.Linear(input_dim, input_dim)

    def forward(self, cnt, sty: Optional[torch.Tensor] = None, content_only: bool = False):
        # cnt/sty: (B, input_dim)
        B = cnt.shape[0]

        def encode_stream(x: torch.Tensor, add_style_token: bool = False):
            parts = [proj(x).unsqueeze(1) for proj in self.part_proj]
            parts = torch.cat(parts, dim=1)
            if add_style_token:
                style_tokens = self.learnable_style_token.expand(B, -1, -1)
                parts = torch.cat([style_tokens, parts], dim=1)
            enc = self.encoder(parts)
            return enc

        style_token_count = self.num_parts

        def decode_tokens(memory_tokens: torch.Tensor, style_signal: torch.Tensor):
            decoder_query = self.decoder_query.unsqueeze(0).expand(B, -1, -1)
            return self.decoder(decoder_query, memory_tokens, style_signal=style_signal)

        cnt_enc = encode_stream(cnt, add_style_token=True)
        cnt_of_content_motion = cnt_enc[:, style_token_count:, :]
        cnt_enc_IN = self.encoder_IN(cnt_of_content_motion)

        if content_only:
            content_style_signal = cnt_enc_IN.reshape(B, -1)
            generated_tokens = decode_tokens(cnt_enc_IN, content_style_signal)
            generated_motion = self.motion_proj(generated_tokens.reshape(B, -1))
            generated_feature = self.out_proj(generated_motion)
            decoder_gate = torch.sigmoid(self.content_decoder_gate)
            return cnt + decoder_gate * generated_feature

        sty_enc_IN = encode_stream(sty, add_style_token=True)
        sty_of_sty_enc = sty_enc_IN[:, :style_token_count, :]
        cnt_of_style_motion = sty_enc_IN[:, style_token_count:, :]
        sty_enc_IN = self.encoder_IN(cnt_of_style_motion)

        # style-conditioned token refinement
        modulated = self.modulator(cnt_enc_IN, sty_enc_IN, sty_of_sty_enc)
        modulated_tokens = modulated.view(B, self.num_parts, self.part_dim)

        # TransformerDecoder generates the motion tokens that will be tracked later
        generated_tokens = decode_tokens(cnt_enc_IN + modulated_tokens, modulated)
        generated_motion = self.motion_proj(generated_tokens.reshape(B, -1))

        out = self.out_proj(generated_motion)
        return out


class ForwardProcDict(nn.Module):

    def __init__(self, obs_shapes, obs_proc_spec, add_outputs=True, learn_weights=False, embed_dim=512, first_hidden_dim=256):

        super(ForwardProcDict, self).__init__()

        self.add_outputs = add_outputs
        self.learn_weights = learn_weights
        self.obs_proc_spec = obs_proc_spec
        self.first_hidden_dim = first_hidden_dim

        missing_obs_keys = [k for k in obs_proc_spec.keys() if k not in obs_shapes]
        if len(missing_obs_keys) > 0:
            print(f"ForwardProcDict ignoring obs keys missing from obs_shapes: {missing_obs_keys}")

        obs_proc_heads = {}
        extra_proj_heads = {}
        for k, v in obs_proc_spec.items():
            if k not in obs_shapes:
                continue
            if v["type"] in ["identity", "flatten",]:
                obs_proc_heads[k] = obs_proc_types[v["type"]]()
            # these project to the network input space and then add it on to the input
            elif v["type"] == "embed" or v["type"] == "flatten_then_embed" or v["type"] == "flatten_then_embed_with_attention":
                if v["type"] == "embed":
                    assert len(obs_shapes[k]) == 1, f"Can only embed 1d observation, but obs {k} has shape {obs_shapes[k]}"
                output_dim = v["output_dim"]
                obs_proc_heads[k] = obs_proc_types[v["type"]](obs_shapes[k], output_dim)
            elif v["type"] == "flatten_then_embed_with_attention_to_hidden" or v["type"] == "embed_with_attention_to_hidden":
                try:
                    extra_proj_heads[k] = obs_proc_types[v["type"]](obs_shapes[k], first_hidden_dim)
                except Exception as e:
                    print(f"Error creating extra proj head for {k}: {e}")
                    import pdb; pdb.set_trace()
            else:
                raise NotImplementedError(f"Obs proc type {v['type']} not implemented")

        self.heads = nn.ModuleDict(obs_proc_heads)
        self.extra_proj_heads = nn.ModuleDict(extra_proj_heads)

        output_shape = self.forward({k: torch.zeros(1, *obs_shapes[k]) for k in obs_shapes})[0].shape
        assert len(output_shape) == 2 # expect one batch dim and then latent dim
        self.output_shape = output_shape[1]
    
    def forward(self, input_dict: Dict[str, torch.Tensor]):
        outputs = []
        extra_add_outputs = []
        extra_proj_outputs = []
        for k, head in self.heads.items():
            head_output = head(input_dict[k])
            outputs.append(head_output)

        # extra proj heads, returned separately to be added later to the net
        for k, head in self.extra_proj_heads.items():
            head_output = head(input_dict[k])
            extra_proj_outputs.append(head_output)

        if self.add_outputs:
            ret = torch.stack(outputs, dim=0).sum(dim=0)
            if len(extra_add_outputs) > 0:
                ret = ret + torch.stack(extra_add_outputs, dim=0).sum(dim=0)
        else:
            ret = torch.cat(outputs, dim=-1)
            if len(extra_add_outputs) > 0:
                ret = ret + torch.stack(extra_add_outputs, dim=0).sum(dim=0)
        return ret, extra_proj_outputs


class SequentialWithExtraProj(nn.Sequential):

    def __init__(self, *args, **kwargs):
        super(SequentialWithExtraProj, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, extra_proj_outputs: Optional[List[torch.Tensor]] = None):
        for idx, module in enumerate(self):
            if idx == 0:
                x = module(x)
                if extra_proj_outputs is not None and len(extra_proj_outputs) > 0:
                    for extra_proj_output in extra_proj_outputs:
                        x = x + extra_proj_output
            else:
                x = module(x)
        return x


def _split_prefixed_mapping(mapping: Dict[str, object], prefix: str):
    return {k: v for k, v in mapping.items() if k.startswith(prefix)}


def _split_obs_streams(obs_shapes: Dict[str, Tuple[int, ...]], obs_proc_spec: Dict[str, Dict], prefixes=("content_", "style_")):
    shared_shapes = {k: v for k, v in obs_shapes.items() if not any(k.startswith(prefix) for prefix in prefixes)}
    shared_spec = {k: v for k, v in obs_proc_spec.items() if not any(k.startswith(prefix) for prefix in prefixes)}

    stream_shapes = {}
    stream_spec = {}
    for prefix in prefixes:
        prefix_shapes = {k: v for k, v in obs_shapes.items() if k.startswith(prefix) and k in obs_proc_spec}
        prefix_spec = {k: v for k, v in obs_proc_spec.items() if k.startswith(prefix)}
        stream_shapes[prefix] = prefix_shapes
        stream_spec[prefix] = prefix_spec

    return shared_shapes, shared_spec, stream_shapes, stream_spec

class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  obs_shapes,
                        num_actions,
                        obs_proc_actor,
                        obs_proc_critic,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        lstm_dim=0,
                        layer_norm=False,
                        # StyleBackbone options
                        stage=1,
                        freeze_style_branch=False,
                        style_lr_scale=0.1,
                        style_lr_warmup_steps=5000,
                        style_num_parts=6,
                        style_part_dim=None,
                        style_num_enc_layers=2,
                        style_num_dec_layers=3,
                        style_num_heads=4,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        activation_name = activation

        # For activation monitoring
        self.activation_hooks = []
        self.activation_values = {}

        self.env_obs_shapes = obs_shapes
        self.env_num_actions = num_actions
        self.stage = stage
        self._freeze_style_branch_override = bool(freeze_style_branch)
        self.freeze_style_branch = bool(freeze_style_branch) or self.stage == 1
        self.use_style_stream = self.stage >= 2 and not self.freeze_style_branch
        self.style_lr_scale = float(style_lr_scale)
        self.style_lr_warmup_steps = int(style_lr_warmup_steps)

        actor_shared_shapes, actor_shared_spec, actor_stream_shapes, actor_stream_spec = _split_obs_streams(obs_shapes, obs_proc_actor)
        critic_shared_shapes, critic_shared_spec, critic_stream_shapes, critic_stream_spec = _split_obs_streams(obs_shapes, obs_proc_critic)

        self.actor_stream_modules = nn.ModuleDict()
        self.critic_stream_modules = nn.ModuleDict()

        def _make_backbone(input_dim):
            if input_dim <= 0:
                return None
            return StyleTransformer(
                input_dim,
                num_parts=style_num_parts,
                part_dim=style_part_dim,
                num_enc_layers=style_num_enc_layers,
                num_dec_layers=style_num_dec_layers,
                num_heads=style_num_heads,
            )

        def _build_stream_module(prefix, shared_shapes, shared_spec, stream_shapes, stream_spec, hidden_dim):
            modules = nn.ModuleDict()
            stream_dims = {}

            if len(shared_spec) > 0:
                input_net = ForwardProcDict(shared_shapes, shared_spec, add_outputs=False, embed_dim=256 if lstm_dim == 0 else lstm_dim, first_hidden_dim=hidden_dim)
                modules[f"{prefix}shared_input_net"] = input_net
                modules[f"{prefix}shared_backbone"] = _make_backbone(input_net.output_shape + lstm_dim)
                if len(input_net.extra_proj_heads) > 0:
                    modules[f"{prefix}shared_extra_adapter"] = nn.Linear(input_net.first_hidden_dim, input_net.output_shape + lstm_dim)
                stream_dims["shared"] = input_net.output_shape + lstm_dim

            for stream_name in ("content_", "style_"):
                if len(stream_spec[stream_name]) == 0:
                    continue
                input_net = ForwardProcDict(stream_shapes[stream_name], stream_spec[stream_name], add_outputs=False, embed_dim=256 if lstm_dim == 0 else lstm_dim, first_hidden_dim=hidden_dim)
                modules[f"{prefix}{stream_name}input_net"] = input_net
                modules[f"{prefix}{stream_name}backbone"] = _make_backbone(input_net.output_shape + lstm_dim)
                if len(input_net.extra_proj_heads) > 0:
                    modules[f"{prefix}{stream_name}extra_adapter"] = nn.Linear(input_net.first_hidden_dim, input_net.output_shape + lstm_dim)
                stream_dims[stream_name] = input_net.output_shape + lstm_dim
            return modules, stream_dims

        self.actor_stream_modules, self.actor_stream_dims = _build_stream_module("actor_", actor_shared_shapes, actor_shared_spec, actor_stream_shapes, actor_stream_spec, actor_hidden_dims[0])
        self.critic_stream_modules, self.critic_stream_dims = _build_stream_module("critic_", critic_shared_shapes, critic_shared_spec, critic_stream_shapes, critic_stream_spec, critic_hidden_dims[0])

        actor_style_dim = self.actor_stream_dims.get("style_")
        critic_style_dim = self.critic_stream_dims.get("style_")
        self.actor_stream_fallback = nn.Parameter(torch.zeros(actor_style_dim)) if actor_style_dim is not None else None
        self.critic_stream_fallback = nn.Parameter(torch.zeros(critic_style_dim)) if critic_style_dim is not None else None

        # Keep the MLP input width fixed to the full stream layout. When style
        # is disabled we will feed a zero vector placeholder so the policy head
        # always sees the same concatenated size.
        mlp_input_dim_a = sum(self.actor_stream_dims.values())
        mlp_input_dim_c = sum(self.critic_stream_dims.values())

        def _make_activation():
            return get_activation(activation_name)

        def _build_mlp(input_dim, hidden_dims, output_dim):
            layers = []
            last_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(last_dim, hidden_dim))
                if layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(_make_activation())
                last_dim = hidden_dim
            layers.append(nn.Linear(last_dim, output_dim))
            return SequentialWithExtraProj(*layers)

        # Policy heads: features are produced by stream-specific StyleBackbones,
        # then decoded by configurable MLP actor/critic heads.
        self.num_actions = num_actions
        self.actor = _build_mlp(mlp_input_dim_a, actor_hidden_dims, num_actions)
        self.critic = _build_mlp(mlp_input_dim_c, critic_hidden_dims, 1)

        self._apply_stage_freeze()

        print(f"Actor network:\nInput/backbone streams: {self.actor_stream_modules}\nMLP head: {self.actor}")
        print(f"Critic network:\nInput/backbone streams: {self.critic_stream_modules}\nMLP head: {self.critic}")

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        self._logits_debug_reported = False
        self._clip_path_mapping_cache = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    def _is_generation_param_name(self, name: str) -> bool:
        style_stream_prefixes = (
            "actor_stream_modules.actor_style_",
            "critic_stream_modules.critic_style_",
        )
        generation_modules = (
            ".modulator.",
        )
        return name.startswith(style_stream_prefixes) or any(module_name in name for module_name in generation_modules)

    def _apply_stage_freeze(self):
        freeze_generation = self.stage == 1 or self._freeze_style_branch_override
        for name, param in self.named_parameters():
            if freeze_generation and self._is_generation_param_name(name):
                param.requires_grad = False
            else:
                param.requires_grad = True

    def set_stage(self, stage: int):
        self.stage = int(stage)
        self.freeze_style_branch = self._freeze_style_branch_override or self.stage == 1
        self.use_style_stream = self.stage >= 2 and not self.freeze_style_branch
        self._apply_stage_freeze()

    def get_style_lr_scale(self, current_learning_iteration: int) -> float:
        if self.freeze_style_branch or not self.use_style_stream:
            return 0.0
        if current_learning_iteration < self.style_lr_warmup_steps:
            return self.style_lr_scale
        return 1.0

    def get_decoder_gate_values(self) -> Dict[str, Dict[str, float]]:
        gate_values = {}
        with torch.no_grad():
            for module_name, module in self.named_modules():
                if not isinstance(module, StyleTransformer):
                    continue
                raw_gate = module.content_decoder_gate.detach()
                gate_values[module_name] = {
                    "raw": float(raw_gate.item()),
                    "sigmoid": float(torch.sigmoid(raw_gate).item()),
                    "requires_grad": bool(module.content_decoder_gate.requires_grad),
                }
        return gate_values

    def get_optimizer_param_groups(self, base_lr: float):
        generation_params = []
        base_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if self._is_generation_param_name(name):
                generation_params.append(param)
            else:
                base_params.append(param)

        param_groups = []
        if len(base_params) > 0:
            param_groups.append({"params": base_params, "lr": base_lr, "name": "base"})
        if len(generation_params) > 0:
            param_groups.append({"params": generation_params, "lr": base_lr * self.style_lr_scale if self.stage >= 2 else 0.0, "name": "style"})
        return param_groups
    
    def re_init_std(self, init_noise_std=1.0):
        self.std.data[:] = init_noise_std 

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _get_first_nonfinite_env_id(self, tensor):
        if tensor is None or torch.isfinite(tensor).all():
            return None
        if tensor.ndim == 0:
            return 0
        if tensor.ndim == 1:
            bad = (~torch.isfinite(tensor)).nonzero(as_tuple=False).flatten()
        else:
            bad_mask = ~torch.isfinite(tensor).reshape(tensor.shape[0], -1).all(dim=1)
            bad = bad_mask.nonzero(as_tuple=False).flatten()
        if len(bad) == 0:
            return None
        return int(bad[0].item())

    def _load_clip_path_mapping(self):
        if self._clip_path_mapping_cache is not None:
            return self._clip_path_mapping_cache

        self._clip_path_mapping_cache = []

        try:
            from legged_gym.tensor_utils import replay_data as replay_data_module
            runtime_mapping = getattr(replay_data_module, 'LATEST_CLIP_PATHS', None)
            if isinstance(runtime_mapping, list) and len(runtime_mapping) > 0:
                self._clip_path_mapping_cache = [str(x) for x in runtime_mapping]
                return self._clip_path_mapping_cache
        except Exception:
            pass

        raw = os.environ.get("VIDEOMIMIC_CLIP_PATHS", "").strip()
        if not raw:
            return self._clip_path_mapping_cache

        try:
            if raw.startswith("["):
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    self._clip_path_mapping_cache = [str(x) for x in parsed]
                    return self._clip_path_mapping_cache

            if os.path.isfile(raw):
                with open(raw, "r") as f:
                    content = f.read().strip()

                if not content:
                    return self._clip_path_mapping_cache

                if content.startswith("["):
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        self._clip_path_mapping_cache = [str(x) for x in parsed]
                        return self._clip_path_mapping_cache

                self._clip_path_mapping_cache = [line.strip() for line in content.splitlines() if line.strip()]
        except Exception as e:
            print(f"Failed to parse VIDEOMIMIC_CLIP_PATHS: {e}")

        return self._clip_path_mapping_cache

    def _try_print_source_h5(self, observations, fallback_tensor=None):
        env_id = None
        if isinstance(observations, dict):
            for value in observations.values():
                env_id = self._get_first_nonfinite_env_id(value)
                if env_id is not None:
                    break
        if env_id is None:
            env_id = self._get_first_nonfinite_env_id(fallback_tensor)

        if env_id is None:
            print("source_h5: unknown (no non-finite env_id could be inferred)")
            return

        clip_idx = None
        if isinstance(observations, dict):
            for key in ["clip_index", "episode_indices", "episode_index"]:
                if key in observations:
                    value = observations[key]
                    if value.ndim == 0:
                        clip_idx = int(value.item())
                    elif value.ndim == 1 and env_id < value.shape[0]:
                        clip_idx = int(value[env_id].item())
                    elif value.ndim >= 2 and env_id < value.shape[0]:
                        clip_idx = int(value[env_id, 0].item())
                    break

        print(f"non_finite_env_id: {env_id}")
        if clip_idx is None:
            print("clip_index: unavailable in observations (add clip_index/episode_indices to obs_dict for exact h5 mapping)")
            return

        print(f"clip_index: {clip_idx}")
        mapping = self._load_clip_path_mapping()
        if 0 <= clip_idx < len(mapping):
            print(f"source_h5: {mapping[clip_idx]}")
        else:
            print("source_h5: unresolved (runtime mapping unavailable, fallback is VIDEOMIMIC_CLIP_PATHS)")

    def _encode_stream(self, stream_modules, module_prefix, stream_name, observations, apply_backbone: bool = True, style_input: Optional[torch.Tensor] = None):
        if stream_name == "":
            input_net_key = f"{module_prefix}shared_input_net"
            backbone_key = f"{module_prefix}shared_backbone"
            adapter_key = f"{module_prefix}shared_extra_adapter"
        else:
            input_net_key = f"{module_prefix}{stream_name}input_net"
            backbone_key = f"{module_prefix}{stream_name}backbone"
            adapter_key = f"{module_prefix}{stream_name}extra_adapter"

        if input_net_key not in stream_modules or backbone_key not in stream_modules:
            return None, None

        input_net = stream_modules[input_net_key]
        backbone = stream_modules[backbone_key]
        extra_adapter = stream_modules[adapter_key] if adapter_key in stream_modules else None

        stream_obs = {k: v for k, v in observations.items() if k.startswith(stream_name) or (stream_name == "" and not (k.startswith("content_") or k.startswith("style_")))}
        if len(stream_obs) == 0:
            return None, None

        processed_obs, extra_proj_outputs = input_net(stream_obs)
        if extra_proj_outputs is not None and len(extra_proj_outputs) > 0 and extra_adapter is not None:
            sum_extra = None
            for e in extra_proj_outputs:
                sum_extra = e if sum_extra is None else (sum_extra + e)
            processed_obs = processed_obs + extra_adapter(sum_extra)
        if apply_backbone and backbone is not None:
            if self.stage == 1:
                processed_obs = backbone(processed_obs, content_only=True)
            else:
                backbone_style = style_input if style_input is not None else processed_obs
                processed_obs = backbone(processed_obs, sty=backbone_style, content_only=False)
        return processed_obs, extra_proj_outputs

    def _encode_dual_stream(self, stream_modules, module_prefix, content_obs, style_obs):
        content_feat, content_extra = self._encode_stream(stream_modules, module_prefix, "content_", content_obs, apply_backbone=False)
        style_feat, style_extra = self._encode_stream(stream_modules, module_prefix, "style_", style_obs, apply_backbone=False)

        if content_feat is None and style_feat is None:
            return None, None

        if content_feat is None:
            content_feat = style_feat
        if style_feat is None:
            style_feat = content_feat

        backbone_key = f"{module_prefix}content_backbone"
        if backbone_key not in stream_modules:
            backbone_key = f"{module_prefix}style_backbone"
        backbone = stream_modules[backbone_key] if backbone_key in stream_modules else None

        if backbone is not None and not self.stage == 1:
            fused = backbone(content_feat, sty=style_feat, content_only=False)
        else:
            fused = content_feat

        extra_proj_outputs = []
        if content_extra is not None:
            extra_proj_outputs.extend(content_extra)
        if style_extra is not None:
            extra_proj_outputs.extend(style_extra)

        return fused, extra_proj_outputs

    def _make_zero_stream_feature(self, reference_tensor: torch.Tensor, feature_dim: int, fallback_parameter: Optional[torch.Tensor] = None):
        if fallback_parameter is None:
            return reference_tensor.new_zeros(reference_tensor.shape[0], feature_dim)
        return fallback_parameter.unsqueeze(0).expand(reference_tensor.shape[0], -1)

    def _fuse_actor_features(self, observations):
        features = []
        extra_proj_outputs = []

        reference_tensor = next(iter(observations.values()))

        shared_feat, shared_extra = self._encode_stream(self.actor_stream_modules, "actor_", "", observations)
        if shared_feat is not None:
            features.append(shared_feat)
        if shared_extra is not None:
            extra_proj_outputs.extend(shared_extra)

        content_obs = {k: v for k, v in observations.items() if k.startswith("content_")}
        style_obs = {k: v for k, v in observations.items() if k.startswith("style_")}
        if self.use_style_stream and len(content_obs) > 0 and len(style_obs) > 0:
            dual_feat, dual_extra = self._encode_dual_stream(self.actor_stream_modules, "actor_", content_obs, style_obs)
            if dual_feat is not None:
                features.append(dual_feat)
            if dual_extra is not None:
                extra_proj_outputs.extend(dual_extra)
        else:
            content_feat, content_extra = self._encode_stream(self.actor_stream_modules, "actor_", "content_", observations)
            if content_feat is not None:
                features.append(content_feat)
            if content_extra is not None:
                extra_proj_outputs.extend(content_extra)

            style_feat, style_extra = self._encode_stream(self.actor_stream_modules, "actor_", "style_", observations)
            if self.use_style_stream:
                if style_feat is not None:
                    features.append(style_feat)
                if style_extra is not None:
                    extra_proj_outputs.extend(style_extra)
            else:
                style_dim = self.actor_stream_dims.get("style_")
                if style_dim is not None:
                    features.append(self._make_zero_stream_feature(reference_tensor, style_dim, self.actor_stream_fallback))

        if len(features) == 0:
            return None, None
        fused = torch.cat(features, dim=-1)
        return fused, extra_proj_outputs

    def _fuse_critic_features(self, observations):
        features = []
        extra_proj_outputs = []

        reference_tensor = next(iter(observations.values()))

        shared_feat, shared_extra = self._encode_stream(self.critic_stream_modules, "critic_", "", observations)
        if shared_feat is not None:
            features.append(shared_feat)
        if shared_extra is not None:
            extra_proj_outputs.extend(shared_extra)

        content_obs = {k: v for k, v in observations.items() if k.startswith("content_")}
        style_obs = {k: v for k, v in observations.items() if k.startswith("style_")}
        if self.use_style_stream and len(content_obs) > 0 and len(style_obs) > 0:
            dual_feat, dual_extra = self._encode_dual_stream(self.critic_stream_modules, "critic_", content_obs, style_obs)
            if dual_feat is not None:
                features.append(dual_feat)
            if dual_extra is not None:
                extra_proj_outputs.extend(dual_extra)
        else:
            content_feat, content_extra = self._encode_stream(self.critic_stream_modules, "critic_", "content_", observations)
            if content_feat is not None:
                features.append(content_feat)
            if content_extra is not None:
                extra_proj_outputs.extend(content_extra)

            style_feat, style_extra = self._encode_stream(self.critic_stream_modules, "critic_", "style_", observations)
            if self.use_style_stream:
                if style_feat is not None:
                    features.append(style_feat)
                if style_extra is not None:
                    extra_proj_outputs.extend(style_extra)
            else:
                style_dim = self.critic_stream_dims.get("style_")
                if style_dim is not None:
                    features.append(self._make_zero_stream_feature(reference_tensor, style_dim, self.critic_stream_fallback))

        if len(features) == 0:
            return None, None
        fused = torch.cat(features, dim=-1)
        return fused, extra_proj_outputs
    
    def update_distribution(self, observations, call_input_net=True):
        extra_proj_outputs = None
        if call_input_net:
            obs_after_proc, extra_proj_outputs = self._fuse_actor_features(observations)
        else:
            obs_after_proc = observations
        logits = self.actor(obs_after_proc)

        if not self._logits_debug_reported:
            def _tensor_is_finite(tensor):
                return tensor is not None and torch.isfinite(tensor).all()

            logits_finite = _tensor_is_finite(logits)
            obs_after_proc_finite = _tensor_is_finite(obs_after_proc)
            extra_proj_finite = True
            if extra_proj_outputs is not None:
                extra_proj_finite = all(_tensor_is_finite(extra_proj_output) for extra_proj_output in extra_proj_outputs)

            if not logits_finite or not obs_after_proc_finite or not extra_proj_finite:
                print("\n===== First non-finite detected in update_distribution =====")
                print(f"obs_after_proc finite: {obs_after_proc_finite}")
                print(f"extra_proj_outputs finite: {extra_proj_finite}")
                print(f"logits finite: {logits_finite}")

                if not obs_after_proc_finite:
                    print(f"obs_after_proc nan: {torch.isnan(obs_after_proc).any()}")
                    print(f"obs_after_proc inf: {torch.isinf(obs_after_proc).any()}")

                if extra_proj_outputs is not None:
                    for idx, extra_proj_output in enumerate(extra_proj_outputs):
                        print(f"extra_proj_outputs[{idx}] nan: {torch.isnan(extra_proj_output).any()}")
                        print(f"extra_proj_outputs[{idx}] inf: {torch.isinf(extra_proj_output).any()}")
                        print(f"extra_proj_outputs[{idx}] key: idx_{idx}")

                print(f"actor_std finite: {torch.isfinite(self.std).all()}")
                if not torch.isfinite(self.std).all():
                    print(f"actor_std nan: {torch.isnan(self.std).any()}")
                    print(f"actor_std inf: {torch.isinf(self.std).any()}")

                self._try_print_source_h5(observations, fallback_tensor=logits)

                self._logits_debug_reported = True

        try:
            std = self.std.expand_as(logits)
            self.distribution = Normal(logits, std)
        except ValueError as e:
            print(f"Error updating distribution: {e}")
            print(f"Logits: {logits}")
            print(f"Std: {std}")

            for k in observations:
                obs_val = observations[k]
                if obs_val.is_floating_point() or obs_val.is_complex():
                    print(f'{k} nan: {torch.isnan(obs_val).any()}')
                    print(f'{k} inf: {torch.isinf(obs_val).any()}')
                else:
                    print(f'{k} nan: False (non-floating dtype={obs_val.dtype})')
                    print(f'{k} inf: False (non-floating dtype={obs_val.dtype})')

            # check if any nan or inf in logits or std
            print(f"Logits nan: {torch.isnan(logits).any()}")
            print(f"Logits inf: {torch.isinf(logits).any()}")
            print(f"Std nan: {torch.isnan(std).any()}")
            print(f"Std inf: {torch.isinf(std).any()}")
            self._try_print_source_h5(observations, fallback_tensor=logits)
            raise e

    def act(self, observations, call_input_net=True, **kwargs):
        self.update_distribution(observations, call_input_net=call_input_net)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, call_input_net=True, monitor_activations=False):
        # Register activation hooks if monitoring is enabled
        if monitor_activations:
            self.register_activation_hooks()
            
        extra_proj_outputs = None
        if call_input_net:
            obs_after_proc, extra_proj_outputs = self._fuse_actor_features(observations)
        else:
            obs_after_proc = observations
        logits = self.actor(obs_after_proc)
        
        # Print activation statistics if monitoring is enabled
        if monitor_activations:
            self.print_activation_stats()
            self.remove_activation_hooks()
            
        return logits

    def evaluate(self, critic_observations, call_input_net=True, monitor_activations=False, **kwargs):
        # Register activation hooks if monitoring is enabled
        if monitor_activations:
            self.register_activation_hooks()
            
        extra_proj_outputs = None
        if call_input_net:
            crit_after_proc, extra_proj_outputs = self._fuse_critic_features(critic_observations)
        else:
            crit_after_proc = critic_observations
        value = self.critic(crit_after_proc)
        
        # Print activation statistics if monitoring is enabled
        if monitor_activations:
            self.print_activation_stats()
            self.remove_activation_hooks()
            
        # if self.normalise_value:
        #     self.value_normalisation.inverse(value)
        return value

    # def learn_value(self, values):
    #     if self.normalise_value:
    #         self.value_normalisation.update(values.view(-1, 1))

    def register_activation_hooks(self):
        """Register forward hooks on each layer to track activations."""
        # Clear any existing hooks
        self.remove_activation_hooks()
        self.activation_values = {}
        
        # Hook function to save activations
        def hook_fn(name):
            def hook(module, input, output):
                self.activation_values[name] = output.detach()
            return hook
        
        # Register hooks for actor layers
        actor_modules = [self.actor] if isinstance(self.actor, nn.Linear) else [m for m in self.actor if isinstance(m, nn.Linear)]
        for i, module in enumerate(actor_modules):
            hook = module.register_forward_hook(hook_fn(f"actor_layer_{i}"))
            self.activation_hooks.append(hook)
            
        # Register hooks for critic layers
        critic_modules = [self.critic] if isinstance(self.critic, nn.Linear) else [m for m in self.critic if isinstance(m, nn.Linear)]
        for i, module in enumerate(critic_modules):
            hook = module.register_forward_hook(hook_fn(f"critic_layer_{i}"))
            self.activation_hooks.append(hook)
    
    def remove_activation_hooks(self):
        """Remove all registered forward hooks."""
        for hook in self.activation_hooks:
            hook.remove()
        self.activation_hooks = []
    
    def print_activation_stats(self):
        """Print statistics of activations in each layer."""
        if not self.activation_values:
            print("No activation values recorded. Call register_activation_hooks() before forward pass.")
            return
            
        print("\n===== Activation Statistics =====")
        for name, activation in sorted(self.activation_values.items()):
            act_mean = activation.mean().item()
            act_min = activation.min().item()
            act_max = activation.max().item()
            act_std = activation.std().item()
            print(f"{name:20s}: mean={act_mean:.6f}, min={act_min:.6f}, max={act_max:.6f}, std={act_std:.6f}")
        print("=================================\n")

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
