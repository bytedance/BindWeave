# Copyright (c) 2025, Bytedance Ltd. and/or its affiliates  
# Copyright (c) 2024, Huawei Technologies Co., Ltd        
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved. 
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
from contextlib import nullcontext
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch_npu
from einops import rearrange
from megatron.core import mpu, tensor_parallel
from megatron.legacy.model.enums import AttnType
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args
from mindspeed.core.context_parallel.unaligned_cp.mapping import (
    all_to_all,
    gather_forward_split_backward,
    split_forward_gather_backward,
)
from mindspeed.core.parallel_state import get_context_parallel_group_for_hybrid_ulysses
from mindspeed.core.context_parallel.ulysses_context_parallel import UlyssesContextAttention

from mindspeed_mm.models.common import MultiModalModule
from mindspeed_mm.models.common.attention import FlashAttention, ParallelAttention
from mindspeed_mm.models.common.embeddings import TextProjection
from mindspeed_mm.models.common.normalize import normalize, FP32LayerNorm


class WanDiT(MultiModalModule):

    def __init__(
        self,
        model_type: str = "t2v",
        patch_size: Tuple[int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 16,
        hidden_size: int = 2048,
        ffn_dim: int = 8192,
        freq_dim: int = 256,
        text_dim: int = 4096,
        img_dim: int = 1280,
        out_dim: int = 16,
        num_heads: int = 16,
        num_layers: int = 32,
        qk_norm: bool = True,
        qk_norm_type: str = "rmsnorm",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        max_seq_len: int = 1024,
        fa_layout: str = "bnsd",
        clip_token_len: int = 257,
        pre_process: bool = True,
        post_process: bool = True,
        global_layer_idx: Optional[Tuple] = None,
        attention_async_offload: bool = False,
        fp32_calculate: bool = False,
        **kwargs,
    ):
        super().__init__(config=None)

        if model_type not in ["t2v", "i2v", "flf2v"]:
            raise ValueError("Please only select among 't2v', 'i2v' and 'flf2v' tasks")

        if not ((hidden_size % num_heads) == 0 and (hidden_size // num_heads) % 2 == 0):
            raise ValueError(
                "The dimension must be divisible by num_heads, and result of 'dim // num_heads' must be even"
            )

        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.hidden_size = hidden_size
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.img_dim = img_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.qk_norm = qk_norm
        self.qk_norm_type = qk_norm_type
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.max_seq_len = max_seq_len
        self.fa_layout = fa_layout
        self.clip_token_len = clip_token_len
        self.pre_process = pre_process
        self.post_process = post_process
        self.global_layer_idx = global_layer_idx if global_layer_idx is not None else tuple(range(num_layers))
        self.head_dim = hidden_size // num_heads
        args = get_args()
        config = core_transformer_config_from_args(args)
        self.recompute_granularity = args.recompute_granularity
        self.distribute_saved_activations = args.distribute_saved_activations
        self.recompute_method = args.recompute_method
        self.recompute_layers = (
            args.recompute_num_layers
            if args.recompute_num_layers is not None
            else num_layers
        )

        self.recompute_skip_core_attention = args.recompute_skip_core_attention
        self.recompute_num_layers_skip_core_attention = args.recompute_num_layers_skip_core_attention
        self.attention_async_offload = attention_async_offload
        self.fp32_calculate = fp32_calculate
        
        self.h2d_stream = torch_npu.npu.Stream() if attention_async_offload else None
        self.d2h_stream = torch_npu.npu.Stream() if attention_async_offload else None

        if self.recompute_granularity == "selective":
            raise ValueError(
                "recompute_granularity does not support selective mode in wanVideo"
            )
        if self.distribute_saved_activations:
            raise NotImplementedError(
                "distribute_save_activation is currently not supported"
            )

        self.enable_tensor_parallel = mpu.get_tensor_model_parallel_world_size() > 1
        self.sequence_parallel = args.sequence_parallel and self.enable_tensor_parallel

        # context parallel setting
        self.context_parallel_algo = (
            args.context_parallel_algo
            if mpu.get_context_parallel_world_size() > 1
            else None
        )
        if (
            self.context_parallel_algo is not None
            and self.context_parallel_algo
            not in ["ulysses_cp_algo", "hybrid_cp_algo", "megatron_cp_algo"]
        ):
            raise NotImplementedError(
                f"Context_parallel_algo {self.context_parallel_algo} is not implemented"
            )

        # rope
        self.rope = RoPE3DWan(head_dim=self.head_dim, max_seq_len=self.max_seq_len)
        # Add a projector for the text features
        self.text_projector = TextProjection(3584, self.text_dim, partial(nn.GELU, approximate="tanh"))
        if self.pre_process:
            # embeddings
            self.patch_embedding = nn.Conv3d(
                self.in_dim,
                self.hidden_size,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
            self.text_embedding = TextProjection(
                self.text_dim, self.hidden_size, partial(nn.GELU, approximate="tanh")
            )
            self.time_embedding = nn.Sequential(
                nn.Linear(self.freq_dim, self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
            if self.fp32_calculate:
                self.time_embedding = self.time_embedding.to(torch.float32)

            # time emb projection
            self.time_projection = nn.Sequential(
                nn.SiLU(), nn.Linear(self.hidden_size, self.hidden_size * 6)
            )

        # attention blocks
        self.blocks = nn.ModuleList(
            [
                WanDiTBlock(
                    model_type,
                    self.hidden_size,
                    self.ffn_dim,
                    self.num_heads,
                    self.qk_norm,
                    self.qk_norm_type,
                    self.cross_attn_norm,
                    self.eps,
                    rope=self.rope,
                    fa_layout=self.fa_layout,
                    clip_token_len=clip_token_len,
                    attention_async_offload=self.attention_async_offload,
                    h2d_stream=self.h2d_stream,
                    d2h_stream=self.d2h_stream,
                    layer_idx=index,
                    num_layers=self.num_layers,
                    fp32_calculate=self.fp32_calculate,
                )
                for index in range(self.num_layers)
            ]
        )

        if self.post_process:
            # head
            self.head = Head(self.hidden_size, self.out_dim, self.patch_size, self.eps)

        if self.pre_process and model_type in ["i2v", "flf2v"]:
            self.img_emb = MLPProj(self.img_dim, self.hidden_size, model_type == 'flf2v', clip_token_len, self.fp32_calculate)

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the module (assuming that all the module parameters have the same dtype)."""
        params = tuple(self.parameters())
        if len(params) > 0:
            return params[0].dtype
        else:
            buffers = tuple(self.buffers())
            return buffers[0].dtype

    @property
    def device(self) -> torch.device:
        """The device of the module (assuming that all the module parameters are in the same device)."""
        params = tuple(self.parameters())
        if len(params) > 0:
            return params[0].device
        else:
            buffers = tuple(self.buffers())
            return buffers[0].device

    def sinusoidal_embedding_1d(self, dim, position, theta=10000):
        sinusoid = torch.outer(
            position.type(torch.float64),
            torch.pow(
                theta,
                -torch.arange(
                    dim // 2, dtype=torch.float64, device=position.device
                ).div(dim // 2),
            ),
        )
        embs = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
        return embs.to(position.dtype)

    def _checkpointed_forward(self, blocks, x, *args):
        "Forward method with activation checkpointing."
        num_layers = len(blocks)
        recompute_layers = self.recompute_layers
        recompute_num_layers_skip_core_attention = (
            self.recompute_num_layers_skip_core_attention
            if self.recompute_skip_core_attention
            else 0
        )

        def custom(start, end):
            def custom_forward(*args):
                for index in range(start, end):
                    layer = blocks[index]
                    x_ = layer(*args)
                return x_

            return custom_forward

        if self.recompute_method == "uniform":
            # Uniformly divide the total number of Transformer layers and
            # checkpoint the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            _layer_num = 0
            while _layer_num < num_layers:
                x = tensor_parallel.checkpoint(
                    custom(_layer_num, _layer_num + recompute_layers),
                    self.distribute_saved_activations,
                    x,
                    *args,
                )
                _layer_num += recompute_layers

        elif self.recompute_method == "block":
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            for _layer_num in range(num_layers):
                if _layer_num < recompute_layers:
                    x = tensor_parallel.checkpoint(
                        custom(_layer_num, _layer_num + 1),
                        self.distribute_saved_activations,
                        x,
                        *args,
                    )
                elif _layer_num < recompute_layers + recompute_num_layers_skip_core_attention:
                    block = blocks[_layer_num]
                    x = block(x, *args, recompute_skip_core_attention=True)
                else:
                    block = blocks[_layer_num]
                    x = block(x, *args)
        else:
            raise ValueError(
                f"Invalid activation recompute method {self.recompute_method}."
            )

        return x

    def patchify(self, embs: torch.Tensor):
        # get f, h, w from b c f h w
        grid_sizes = embs.shape[2:]

        # b c f h w  -> b (f h w) c
        patch_out = rearrange(embs, "b c f h w -> b (f h w) c").contiguous()

        return patch_out, grid_sizes

    def unpatchify(self, embs, frames, height, width):
        # b (f h w) (x y z c) -> b c (f x) (h y) (w z)
        patch_out = rearrange(
            embs,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=frames,
            h=height,
            w=width,
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )
        return patch_out

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        prompt: torch.Tensor,
        hidden_states: torch.Tensor,
        prompt_mask: torch.Tensor = None,
        hidden_states_mask: torch.Tensor = None,
        i2v_clip_feature: torch.Tensor = None,
        i2v_vae_feature: torch.Tensor = None,
        **kwargs,
    ):
        if self.pre_process:
            timestep = timestep.to(x[0].device)
            # time embeddings
            times = self.time_embedding(
                self.sinusoidal_embedding_1d(self.freq_dim, timestep)
            )
            time_emb = self.time_projection(times).unflatten(1, (6, self.hidden_size))

            # prompt embeddings
            bs = prompt.size(0)
            prompt = prompt.view(bs, -1, prompt.size(-1)) # (b, seq_len, dim) [1 512 4096]
            hidden_states = hidden_states.view(bs, -1, hidden_states.size(-1)) # (b, seq_len, dim) [1 512 4096]
            # Apply the projector to align dimensions
            if hidden_states.shape[-1] != self.text_dim:
                hidden_states = self.text_projector(hidden_states)
            prompt = torch.cat([hidden_states, prompt], dim=1) # (b, seq_len1+seq_len2, dim)
            # prompt_mask [1 1 512]
            if prompt_mask is not None:
                prompt_mask = torch.cat([hidden_states_mask, prompt_mask], dim=2)
                seq_lens = prompt_mask.view(bs, -1).sum(dim=-1) 
                for i, seq_len in enumerate(seq_lens):
                    prompt[i, seq_len:] = 0
            prompt_emb = self.text_embedding(prompt)
            # cat i2v & flf2v
            if self.model_type in ["i2v", "flf2v"]:
                i2v_clip_feature = i2v_clip_feature.to(x)
                i2v_vae_feature = i2v_vae_feature.to(x)
                x = torch.cat([x, i2v_vae_feature], dim=1)  # (b, c[x+y], f, h, w)
                clip_embedding = self.img_emb(i2v_clip_feature.float() if self.fp32_calculate else i2v_clip_feature.to(time_emb.dtype))
                prompt_emb = torch.cat([clip_embedding, prompt_emb], dim=1)

            # patch embedding
            patch_emb = self.patch_embedding(x.to(time_emb.dtype))

            embs, grid_sizes = self.patchify(patch_emb)

            # rotary positional embeddings
            batch_size, frames, height, width = (
                embs.shape[0],
                grid_sizes[0],
                grid_sizes[1],
                grid_sizes[2],
            )
        else:
            batch_size, _, frames, height, width = kwargs["ori_shape"]
            height, width = height // self.patch_size[1], width // self.patch_size[2]
            prompt_emb = kwargs['prompt_emb']
            time_emb = kwargs['time_emb']
            times = kwargs['times']
            embs = x

        rotary_pos_emb = self.rope(batch_size, frames, height, width)

        # RNG context
        if self.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # cp split
        if self.context_parallel_algo is not None:
            if self.pre_process:
                embs = split_forward_gather_backward(
                    embs, mpu.get_context_parallel_group(), dim=1, grad_scale="down"
                )  # b s h
            rotary_pos_emb = split_forward_gather_backward(
                rotary_pos_emb,
                mpu.get_context_parallel_group(),
                dim=0,
                grad_scale="down",
            )

        with rng_context:
            if self.recompute_granularity == "full":
                embs = self._checkpointed_forward(
                    self.blocks,
                    embs,
                    prompt_emb,
                    time_emb,
                    rotary_pos_emb,
                )
            else:
                for block in self.blocks:
                    embs = block(embs, prompt_emb, time_emb, rotary_pos_emb)

        out = embs
        if self.post_process:
            if self.context_parallel_algo is not None:
                embs = gather_forward_split_backward(
                    embs, mpu.get_context_parallel_group(), dim=1, grad_scale="up"
                )
            embs_out = self.head(embs, times)
            out = self.unpatchify(embs_out, frames, height, width)

        rtn = (out, prompt, prompt_emb, time_emb, times, prompt_mask)

        return rtn

    def pipeline_set_prev_stage_tensor(self, input_tensor_list, extra_kwargs):
        """
        Implemented for pipeline parallelism. The input tensor is got from last PP stage.
        Args:
            input_tensor_list: same as the return value of pipeline_set_next_stage_tensor
            extra_kwargs: kwargs for forward func.

        Returns:
            predictor_input_list: values for predictor forward.
            training_loss_input_list: values to calculate loss.
        """
        (prev_output, prompt, prompt_emb, time_emb, times, prompt_mask,
         latents, timesteps, noise) = input_tensor_list
        predictor_input_list = [prev_output, timesteps, prompt, None, prompt_mask]
        training_loss_input_list = [latents, None, timesteps, noise, None]
        extra_kwargs['prompt_emb'] = prompt_emb
        extra_kwargs['time_emb'] = time_emb
        extra_kwargs['times'] = times
        extra_kwargs["ori_shape"] = latents.shape
        return predictor_input_list, training_loss_input_list

    def pipeline_set_next_stage_tensor(self, input_list, output_list, extra_kwargs=None):
        """
        input_list: [latents, noised_latents, timesteps, noise, video_mask]
        output_list (predict_output):[out, prompt, prompt_emb, time_emb, times, prompt_mask]

        return as
        prev_output, prompt, prompt_emb, prompt_emb, time_emb, times, prompt_mask,
        latents, timesteps, noise

        which should be corresponded with initialize_pipeline_tensor_shapes
        """
        latents, _, timesteps, noise, _ = input_list
        if timesteps.dtype != torch.float32:
            timesteps = timesteps.to(torch.float32)

        return list(output_list) + [latents, timesteps, noise]

    @staticmethod
    def initialize_pipeline_tensor_shapes():
        args = get_args()
        micro_batch_size = args.micro_batch_size
        dtype = args.params_dtype

        model_cfg = args.mm.model
        data_cfg = args.mm.data.dataset_param.preprocess_parameters
        hidden_size = model_cfg.predictor.hidden_size
        height = getattr(data_cfg, "max_height", 480)
        width = getattr(data_cfg, "max_width", 832)
        vae_scale_factor = getattr(model_cfg.predictor, "vae_scale_factor", [4, 8, 8])
        latent_size = ((data_cfg.num_frames + 3) // vae_scale_factor[0], height // vae_scale_factor[1], width // vae_scale_factor[2])
        divisor = model_cfg.predictor.patch_size[0] * model_cfg.predictor.patch_size[1] * \
                  model_cfg.predictor.patch_size[2]
        seq_len = latent_size[0] * latent_size[1] * latent_size[2] // divisor // mpu.get_context_parallel_world_size()
        channels = model_cfg.predictor.out_dim
        text_dim = model_cfg.predictor.text_dim
        text_len = model_cfg.predictor.text_len
        img_token_len = model_cfg.predictor.clip_token_len if model_cfg.predictor.model_type == 'i2v' else 0
        pipeline_tensor_shapes = [
            {'shape': (micro_batch_size, seq_len, hidden_size), 'dtype': dtype},  # prev_output
            {'shape': (micro_batch_size, text_len, text_dim), 'dtype': dtype},  # prompt
            {'shape': (micro_batch_size, text_len + img_token_len, hidden_size), 'dtype': dtype},  # prompt_emb
            {'shape': (micro_batch_size, 6, hidden_size), 'dtype': dtype},  # time_emb
            {'shape': (micro_batch_size, hidden_size), 'dtype': dtype},  # times
            {'shape': (micro_batch_size, 1, text_len), 'dtype': dtype},  # origin_prompt_mask
            {'shape': (micro_batch_size, channels, *latent_size), 'dtype': dtype},  # latents(x0)
            {'shape': (micro_batch_size,), 'dtype': torch.float32},  # timesteps
            {'shape': (micro_batch_size, channels, *latent_size), 'dtype': dtype},  # noise
        ]
        return pipeline_tensor_shapes


class WanDiTBlock(nn.Module):

    def __init__(
        self,
        model_type: "t2v",
        hidden_size: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: bool = True,
        qk_norm_type: str = "rmsnorm",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        dropout: float = 0.0,
        rope=None,
        fa_layout=None,
        clip_token_len: int = 257,
        attention_async_offload: bool = False,
        layer_idx: int = 0,
        num_layers: int = 40,
        fp32_calculate: bool = False,
        h2d_stream: Optional[torch_npu.npu.Stream] = None,
        d2h_stream: Optional[torch_npu.npu.Stream] = None,
        **kwargs
    ):
        super().__init__()

        self.model_type = model_type
        self.rope = rope
        self.hidden_size = hidden_size
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.clip_token_len = clip_token_len
        self.fp32_calculate = fp32_calculate

        args = get_args()
        self.distribute_saved_activations = args.distribute_saved_activations

        self.attention_async_offload_param = {
            "async_offload": attention_async_offload,
            "block_idx": layer_idx,
            "depth": num_layers,
            "h2d_stream": h2d_stream,
            "d2h_stream": d2h_stream,
        }

        self.self_attn = WanVideoParallelAttention(
            query_dim=hidden_size,
            key_dim=None,
            num_attention_heads=num_heads,
            hidden_size=hidden_size,
            proj_q_bias=attention_bias,
            proj_k_bias=attention_bias,
            proj_v_bias=attention_bias,
            proj_out_bias=attention_out_bias,
            dropout=dropout,
            use_qk_norm=qk_norm,
            norm_type=qk_norm_type,
            norm_eps=eps,
            rope=rope,
            attention_type=AttnType.self_attn,
            has_img_input=False,
            fa_layout=fa_layout,
        )

        self.cross_attn = WanVideoParallelAttention(
            query_dim=hidden_size,
            key_dim=None,
            num_attention_heads=num_heads,
            hidden_size=hidden_size,
            proj_q_bias=attention_bias,
            proj_k_bias=attention_bias,
            proj_v_bias=attention_bias,
            proj_out_bias=attention_out_bias,
            dropout=dropout,
            use_qk_norm=qk_norm,
            norm_type=qk_norm_type,
            norm_eps=eps,
            attention_type=AttnType.cross_attn,
            has_img_input=model_type in ["i2v", "flf2v"],
            fa_layout=fa_layout,
        )

        self.norm1 = nn.LayerNorm(self.hidden_size, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(self.hidden_size, eps=eps, elementwise_affine=False)
        self.norm3 = FP32LayerNorm(self.hidden_size, eps=eps) if fp32_calculate else nn.LayerNorm(self.hidden_size, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, self.ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.ffn_dim, self.hidden_size),
        )
        # modulation
        self.modulation = nn.Parameter(
            torch.randn(1, 6, self.hidden_size) / self.hidden_size**0.5
        )

    def modulate(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
        return x * (1 + scale) + shift

    def forward(
        self,
        latents,
        prompt,
        time_emb,
        rotary_pos_emb,
        recompute_skip_core_attention=False
    ):
        # before self attention process
        if recompute_skip_core_attention:
            query, key, value, gate_msa, shift_mlp, scale_mlp, gate_mlp = tensor_parallel.checkpoint(
                self._before_self_attention,
                self.distribute_saved_activations,
                time_emb, 
                latents,
                rotary_pos_emb,
            )
        else:
            query, key, value, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._before_self_attention(
                time_emb, 
                latents,
                rotary_pos_emb
            )

        # self attention
        self_attn_out = self.self_attn.core_attention_flash(
            query=query,
            key=key, 
            value=value, 
            **self.attention_async_offload_param
        )

        # after self attention
        if recompute_skip_core_attention:
            latents = tensor_parallel.checkpoint(
                self._after_self_attention,
                self.distribute_saved_activations,
                self_attn_out,
                latents,
                prompt,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp
            )
        else:
            latents = self._after_self_attention(
                self_attn_out, 
                latents, 
                prompt, 
                gate_msa, 
                shift_mlp, 
                scale_mlp, 
                gate_mlp
            )

        return latents

    def _before_self_attention(
        self,
        time_emb,
        latents,
        rotary_pos_emb
    ):
        dtype = time_emb.dtype
        modu_dtype = torch.float32 if self.fp32_calculate else dtype
        device = time_emb.device
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=modu_dtype, device=device) + time_emb.to(modu_dtype)
        ).chunk(6, dim=1)

        self_attn_input = self.modulate(
            self.norm1(latents.to(torch.float32)), shift_msa, scale_msa
        ).to(dtype)

        # before self attention
        query, key, value = self.self_attn.function_before_core_attention(
            query=self_attn_input,
            input_layout="bsh",
            rotary_pos_emb=rotary_pos_emb.to(time_emb.device)
        )

        return (
            query, key, value,
            gate_msa, shift_mlp, scale_mlp, gate_mlp
        )
    
    def _after_self_attention(
        self,
        self_attn_out,
        latents,
        prompt,
        gate_msa,
        shift_mlp,
        scale_mlp,
        gate_mlp
    ):
        dtype = torch.float32 if self.fp32_calculate else latents.dtype
        self_attn_out = self.self_attn.function_after_core_attention(self_attn_out, output_layout="bsh")
        
        latents = (latents + gate_msa * self_attn_out).to(latents.dtype)

        # cross attention
        crs_attn_input = self.norm3(latents.to(dtype)).to(latents.dtype)

        # i2v & flf2v
        if self.model_type in ["i2v", "flf2v"]:
            img_clip_token_len = 2 * self.clip_token_len if self.model_type == "flf2v" else self.clip_token_len
            img = prompt[:, :img_clip_token_len]
            txt = prompt[:, img_clip_token_len:]
            crs_attn_out = self.cross_attn(
                query=crs_attn_input,
                key=(img, txt),
                input_layout="bsh",
            )
        # t2v
        else:
            txt = prompt
            crs_attn_out = self.cross_attn(
                query=crs_attn_input,
                key=txt,
                input_layout="bsh",
            )

        latents = latents + crs_attn_out
        modu_out = self.modulate(self.norm2(latents.to(dtype)), shift_mlp, scale_mlp).to(latents.dtype)

        # ffn
        latents = ((latents.to(dtype)) + gate_mlp * self.ffn(modu_out).to(dtype)).to(latents.dtype)

        return latents


class RoPE3DWan(nn.Module):
    def __init__(self, head_dim, max_seq_len):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        self.freqs = self.get_freq(head_dim)

    def get_freq(self, head_dim):
        if head_dim <= 0:
            raise ValueError("head dimension must be greater than 0")

        dim1 = head_dim - 2 * (head_dim // 3)
        dim2 = head_dim // 3

        # generate frequency matrices
        freqs1 = self.rope_params(self.max_seq_len, dim1)
        freqs2 = self.rope_params(self.max_seq_len, dim2)
        freqs3 = self.rope_params(self.max_seq_len, dim2)
        return freqs1, freqs2, freqs3

    def rope_params(self, max_seq_len, dim, theta=10000):
        if dim % 2 != 0:
            raise ValueError("Dimension must be even")

        # compute frequencies
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
        freqs = torch.outer(torch.arange(max_seq_len, device=freqs.device), freqs)

        # convert to complex numbers
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def apply_rotary_pos_emb(self, tokens, freqs):
        dtype = tokens.dtype
        cos, sin = torch.chunk(torch.view_as_real(freqs.to(torch.complex64)), 2, dim=-1)

        B, S, N, D = tokens.shape

        def rotate_half(x):
            half_1, half_2 = torch.chunk(x.reshape((B, S, N, D // 2, 2)), 2, dim=-1)
            return torch.cat((-half_2, half_1), dim=-1).reshape((B, S, N, D))

        cos = cos.expand(-1, -1, -1, -1, 2).flatten(-2)
        sin = sin.expand(-1, -1, -1, -1, 2).flatten(-2)
        res = tokens * cos + rotate_half(tokens) * sin

        return res.to(dtype)

    def forward(self, b, f, h, w):
        seq_len = f * h * w

        # get freqs
        freqs = (
            torch.cat(
                [
                    self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(seq_len, 1, 1, -1)
            .expand(seq_len, b, 1, -1)
        )
        return freqs


class WanVideoParallelAttention(ParallelAttention):

    def __init__(
        self,
        query_dim: int,
        key_dim: Optional[int],
        num_attention_heads: int,
        hidden_size: int,
        proj_q_bias: bool = False,
        proj_k_bias: bool = False,
        proj_v_bias: bool = False,
        proj_out_bias: bool = False,
        dropout: float = 0.0,
        use_qk_norm: bool = False,
        norm_type: str = None,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: int = AttnType.self_attn,
        has_img_input: bool = False,
        fa_layout: str = "bnsd",
        rope=None,
        **kwargs,
    ):
        super().__init__(
            query_dim=query_dim,
            key_dim=key_dim,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            proj_q_bias=proj_q_bias,
            proj_k_bias=proj_k_bias,
            proj_v_bias=proj_v_bias,
            proj_out_bias=proj_out_bias,
            dropout=dropout,
            use_qk_norm=use_qk_norm,
            norm_type=norm_type,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            is_qkv_concat=False,
            attention_type=attention_type,
            is_kv_concat=False,
            fa_layout=fa_layout,
            rope=rope,
            **kwargs,
        )
        args = get_args()

        if self.cp_size > 1 and attention_type == AttnType.self_attn \
            and args.context_parallel_algo in ["megatron_cp_algo", "hybrid_cp_algo"]:
            # The input layout of `ringattn_context_parallel` must be 'sbh'
            fa_layout = "sbh"

        self.core_attention_flash = FlashAttention(
            attention_dropout=dropout,
            fa_layout=fa_layout,
            softmax_scale=1 / math.sqrt(self.head_dim),
        )

        if self.cp_size > 1 and attention_type == AttnType.self_attn \
            and args.context_parallel_algo in ["ulysses_cp_algo", "hybrid_cp_algo"]:

            if args.context_parallel_algo == "hybrid_cp_algo":
                ulysses_group = get_context_parallel_group_for_hybrid_ulysses()
            else:
                ulysses_group = mpu.get_context_parallel_group()            
            
            self.core_attention_flash = UlyssesContextAttention(self.core_attention_flash, ulysses_group)
        
        if self.cp_size > 1 and attention_type == AttnType.cross_attn:
            #In the case of cross attention, it is equivalent to performing the raw npu_fusion_attention for the slicing q
            self.core_attention_flash.context_parallel_algo = "ulysses_cp_algo"

        # Normalize
        if self.use_qk_norm:
            self.q_norm = normalize(
                norm_type=norm_type,
                in_channels=hidden_size,
                eps=norm_eps,
                affine=norm_elementwise_affine,
                **kwargs,
            )
            self.k_norm = normalize(
                norm_type=norm_type,
                in_channels=hidden_size,
                eps=norm_eps,
                affine=norm_elementwise_affine,
                **kwargs,
            )
            if isinstance(self.q_norm, nn.LayerNorm):
                for param in self.q_norm.parameters():
                    setattr(param, "sequence_parallel", self.sequence_parallel)
            if isinstance(self.k_norm, nn.LayerNorm):
                for param in self.k_norm.parameters():
                    setattr(param, "sequence_parallel", self.sequence_parallel)

        self.has_img_input = has_img_input
        if self.has_img_input:
            args = get_args()
            config = core_transformer_config_from_args(args)

            self.k_img = tensor_parallel.ColumnParallelLinear(
                query_dim,
                hidden_size,
                config=config,
                init_method=config.init_method,
                bias=proj_q_bias,
                gather_output=False,
            )
            self.v_img = tensor_parallel.ColumnParallelLinear(
                query_dim,
                hidden_size,
                config=config,
                init_method=config.init_method,
                bias=proj_q_bias,
                gather_output=False,
            )
            self.k_norm_img = normalize(
                norm_type=norm_type,
                in_channels=hidden_size,
                eps=norm_eps,
                affine=norm_elementwise_affine,
                **kwargs,
            )

    def get_query_key_value_tensors(self, hidden_states, key_value_states):
        """
        Derives `query` tensor from `hidden_states`, and `key`/`value` tensor
        from `hidden_states` or `key_value_states`.
        """
        if self.has_img_input:
            img_key_value_states, context_key_value_states = key_value_states
            # Attention heads [s, b, h] --> [s, b, h]
            query = self.proj_q(hidden_states)[0]
            img_key = self.k_img(img_key_value_states)[0]
            img_value = self.v_img(img_key_value_states)[0]
            key = self.proj_k(context_key_value_states)[0]
            value = self.proj_v(context_key_value_states)[0]
        else:
            # Attention heads [s, b, h] --> [s, b, h]
            query = self.proj_q(hidden_states)[0]
            key = self.proj_k(key_value_states)[0]
            value = self.proj_v(key_value_states)[0]

        if self.use_qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)
            if self.has_img_input:
                img_key = self.k_norm_img(img_key)

        # [s, b, h] --> [s, b, n, d]
        batch_size = query.shape[1]
        query = query.view(
            -1, batch_size, self.num_attention_heads_per_partition, self.head_dim
        )
        key = key.view(
            -1, batch_size, self.num_attention_heads_per_partition, self.head_dim
        )
        value = value.view(
            -1, batch_size, self.num_attention_heads_per_partition, self.head_dim
        )

        if self.has_img_input:
            img_key = img_key.view(
                -1, batch_size, self.num_attention_heads_per_partition, self.head_dim
            )
            img_value = img_value.view(
                -1, batch_size, self.num_attention_heads_per_partition, self.head_dim
            )
            key = [img_key, key]
            value = [img_value, value]
        return query, key, value

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[Union[torch.Tensor, Tuple[torch.Tensor]]] = None,
        mask: Optional[torch.Tensor] = None,
        input_layout: str = "sbh",
        rotary_pos_emb: Optional[torch.Tensor] = None,
    ):
        if self.has_img_input:
            query, key, value = self.function_before_core_attention(
                query, key, input_layout, rotary_pos_emb
            )
            img_core_attn_out = self.core_attention_flash(query, key[0], value[0], mask)
            core_attn_out = self.core_attention_flash(query, key[1], value[1], mask)
            core_attn_out = img_core_attn_out + core_attn_out
            out = self.function_after_core_attention(core_attn_out, input_layout)
            return out
        else:
            return super().forward(query, key, mask, input_layout, rotary_pos_emb)


class Head(nn.Module):

    def __init__(
        self, dim: int, out_dim: int, patch_size: List[int], eps: float = 1e-6, fp32_calculate: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.fp32_calculate = fp32_calculate

        self.norm = FP32LayerNorm(dim, eps=eps, elementwise_affine=False) if fp32_calculate else nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, latents, times):
        shift, scale = (
            self.modulation.to(dtype=torch.float32 if self.fp32_calculate else times.dtype, device=times.device) + times
        ).chunk(2, dim=1)
        out = self.head((self.norm(latents.float() if self.fp32_calculate else latents) * (1 + scale) + shift).to(latents.dtype))
        return out


class MLPProj(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, flf_pos_emb=False, clip_token_len=257, fp32_calculate=False):
        super().__init__()

        self.proj = nn.Sequential(
            FP32LayerNorm(in_dim) if fp32_calculate else nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            FP32LayerNorm(out_dim) if fp32_calculate else nn.LayerNorm(out_dim),
        )
        if flf_pos_emb:   # NOTE: only used in "flf2v"
            self.emb_pos = nn.Parameter(torch.zeros(1, clip_token_len * 2, in_dim))

    def forward(self, image_emb):
        if hasattr(self, 'emb_pos'):
            bs, n, d = image_emb.shape
            image_emb = image_emb.view(-1, 2 * n, d)
            image_emb = image_emb + self.emb_pos
        return self.proj(image_emb)
