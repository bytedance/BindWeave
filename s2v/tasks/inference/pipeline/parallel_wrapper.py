# coding=utf-8
# Copyright (c) 2025, Bytedance Ltd. and/or its affiliates. All rights reserved.  
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reversed.
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from copy import deepcopy

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron.core import mpu, InferenceParams
from megatron.core.distributed import DistributedDataParallel as LocalDDP
from megatron.inference.text_generation.communication import recv_from_prev_pipeline_rank_, send_to_next_pipeline_rank
from megatron.legacy.model import Float16Module as MegatronFloat16Module
from megatron.training import get_args
from megatron.training.checkpointing import load_checkpoint
from megatron.training.training import get_model
from megatron.training.utils import unwrap_model
from mindspeed_mm.utils.transformer_model_config import get_model_config


class ParallelWrapper:

    def __init__(self, model):

        model = get_model(model, wrap_with_ddp=False)
        load_checkpoint(model, None, None, 'load')
        self.model = unwrap_model(model, (torchDDP, LocalDDP, MegatronFloat16Module))[0].eval()
        # Pipelining arguments.
        args = get_args()
        self.vlm_config = deepcopy(args.mm.model.infer_config)
        self.text_decoder_config = get_model_config(self.vlm_config.text_decoder)
        self.pipeline_size_larger_than_one = args.pipeline_model_parallel_size > 1
        # Threshold of pipelining.
        self.pipelining_batch_x_seqlen = getattr(self.text_decoder_config, 'inference_batch_times_seqlen_threshold', 512)
        if self.vlm_config.generation_config.max_length and self.vlm_config.generation_config.max_new_tokens:
            print(
                "warning: Both `max_new_tokens` (= %s) and `max_length` (= %s) seem to have been set. `max_new_tokens` will take precedence. "
                % (self.vlm_config.generation_config.max_new_tokens, self.vlm_config.generation_config.max_length))
        self.inference_params = None
        # Some models input parameters do not support the splitting into micro batch sizes.
        self.split_batch = getattr(self.vlm_config.generation_config, "split_batch", True)

    def __call__(self, **kwargs):
        """Invocation of the forward methods. """
        input_ids = kwargs.get("input_ids", None)
        batch_size = input_ids.size(0)
        sequence_length = input_ids.size(1)

        if self.inference_params is None and hasattr(self.vlm_config.generation_config,
                                                     "kv_cache") and self.vlm_config.generation_config.kv_cache:
            max_length = _get_max_length(self.vlm_config.generation_config, sequence_length)
            self.inference_params = InferenceParams(batch_size, max_length)

        model_forward_kwargs = kwargs

        # Pipelining case.
        if self.pipeline_size_larger_than_one:
            if not self.split_batch:
                return self._with_pipelining_forward_step_without_split_batch(model_forward_kwargs)
            else:
                current_batch_x_seqlen = input_ids.size(0) * input_ids.size(1)
                micro_batch_size = 1
                if current_batch_x_seqlen >= self.pipelining_batch_x_seqlen:
                    micro_batch_size = max(1, self.pipelining_batch_x_seqlen // input_ids.size(1))
                return self._with_pipelining_forward_step(model_forward_kwargs, micro_batch_size)

        return self._no_pipelining_forward_step(model_forward_kwargs)

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        return getattr(self.model, item)

    def _forward(self, **kwargs):
        if self.inference_params is not None:
            kwargs["inference_params"] = self.inference_params
        return self.model(**kwargs)

    def _forward_step_helper(self, model_forward_kwargs, recv_buffer=None):
        """Single forward step. Update the allocate memory flag so
        only the first time the memory is allocated."""
        tokens = model_forward_kwargs.get("input_ids", None)
        batch_size = tokens.size(0)
        sequence_length = tokens.size(1)
        if recv_buffer is None:
            recv_buffer = _allocate_recv_buffer(batch_size, sequence_length)

        # Receive from previous stage.
        recv_from_prev_pipeline_rank_(recv_buffer)

        # Forward pass through the model.
        self.model.set_input_tensor(recv_buffer)
        output_tensor = self._forward(**model_forward_kwargs)

        # Send output to the next stage.
        send_to_next_pipeline_rank(output_tensor)

        return output_tensor

    def _no_pipelining_forward_step(self, model_forward_kwargs, recv_buffer=None):
        """If recv_buffer is none, we will allocate one on the fly."""
        # Run a simple forward pass.
        output_tensor = self._forward_step_helper(model_forward_kwargs, recv_buffer=recv_buffer) # hidden states
        tokens = model_forward_kwargs.get("input_ids", None)
        # Update the sequence length offset.
        if self.inference_params is not None:
            self.inference_params.sequence_len_offset += tokens.size(1)

        logits = None
        if mpu.is_pipeline_last_stage():
            logits = output_tensor

        return logits

    def _with_pipelining_forward_step_without_split_batch(self, model_forward_kwargs):
        # Some models input parameters do not support the splitting into micro batch sizes.

        input_ids = model_forward_kwargs.get("input_ids", None)
        batch_size = input_ids.size(0)
        sequence_length = input_ids.size(1)

        # Preallocate memory for output logits.
        logits = None
        if mpu.is_pipeline_last_stage():
            args = get_args()
            logits = torch.empty(
                (batch_size, sequence_length, args.padded_vocab_size),
                dtype=torch.float32, device=torch.cuda.current_device())

        # Preallocate recv buffer.
        recv_buffer = _allocate_recv_buffer(batch_size, sequence_length)

        output = self._forward_step_helper(model_forward_kwargs, recv_buffer=recv_buffer)
        # Adjust the batch size offset to account for the micro-batch.
        if self.inference_params is not None:
            self.inference_params.batch_size_offset += batch_size

        if mpu.is_pipeline_last_stage():
            if isinstance(output, dict):
                logits = output["logits"]
        if self.inference_params is not None:
            # Once we are done with all the micro-batches, we can
            # adjust the sequence length offset.
            self.inference_params.sequence_len_offset += sequence_length
            # and reset the batch size offset
            self.inference_params.batch_size_offset = 0

        return logits



    def _with_pipelining_forward_step(self, model_forward_kwargs, micro_batch_size):
        """No interleaving is supported.
           Ensure the first dimension of `model_forward_kwargs` is the batch size.
        """

        first_dims = [v.shape[0] for k, v in model_forward_kwargs.items() if (k != "position_ids" and k != "cache_position" and v is not None)]
        if "position_ids" in model_forward_kwargs.keys():
            first_dims.append(model_forward_kwargs["position_ids"].shape[1])
        if not len(set(first_dims)) == 1:
            raise Exception(
                "All values in model_forward_kwargs must have the same first dimension, which represents the batch size.")

        tokens = model_forward_kwargs.get("input_ids")

        sequence_length = tokens.size(1)
        batch_size = tokens.size(0)

        # Divide the batch dimension into micro batches.
        num_micro_batches, last_chunk = divmod(batch_size,
                                               micro_batch_size)

        if last_chunk > 0:
            num_micro_batches += 1

        # Preallocate memory for output logits.
        logits = None
        if mpu.is_pipeline_last_stage():
            args = get_args()
            logits = torch.empty(
                (batch_size, sequence_length, args.padded_vocab_size),
                dtype=torch.float32, device=torch.cuda.current_device())

        # Preallocate recv buffer.
        recv_buffer = _allocate_recv_buffer(micro_batch_size, sequence_length)

        for micro_batch_index in range(num_micro_batches):
            # Slice among the batch dimenion.
            start = micro_batch_index * micro_batch_size
            end = min(start + micro_batch_size, batch_size)
            this_micro_batch_size = end - start

            model_forward_kwargs = {
                key: (value[start:end, ...] if value is not None else None)
                for key, value in model_forward_kwargs.items()
            }

            # Run a simple forward pass.
            if this_micro_batch_size != micro_batch_size:
                recv_buffer = None
            output = self._forward_step_helper(model_forward_kwargs, recv_buffer=recv_buffer)

            # Adjust the batch size offset to account for the micro-batch.
            if self.inference_params is not None:
                self.inference_params.batch_size_offset += this_micro_batch_size

            # Copy logits.
            if mpu.is_pipeline_last_stage():
                if isinstance(output, dict):
                    logits = output["logits"][start:end, ...]
        if self.inference_params is not None:
            # Once we are done with all the micro-batches, we can
            # adjust the sequence length offset.
            self.inference_params.sequence_len_offset += sequence_length
            # and reset the batch size offset
            self.inference_params.batch_size_offset = 0
        return logits


def _get_recv_buffer_dtype(args):
    """Receive happens between the layers."""
    if args.fp32_residual_connection:
        return torch.float
    return args.params_dtype


def _allocate_recv_buffer(batch_size, sequence_length):
    """Receive happens between the layers with size [s, b, h]."""
    if mpu.is_pipeline_first_stage():
        return None
    args = get_args()
    vlm_config = deepcopy(args.mm.model)
    text_decoder_config = get_model_config(vlm_config.text_decoder)
    recv_size = (sequence_length, batch_size, text_decoder_config.hidden_size)
    return torch.empty(recv_size,
                       dtype=_get_recv_buffer_dtype(args),
                       device=torch.cuda.current_device())


def _get_max_length(config, inputs_seq_length):
    if config.max_new_tokens and not config.max_length:
        max_length = inputs_seq_length + config.max_new_tokens
    elif config.max_length and not config.max_new_tokens:
        if config.max_length < inputs_seq_length:
            raise ValueError(
                f"generation config max length:{config.max_length} must larger than inputs seq length:{inputs_seq_length}")
        max_length = config.max_length
    elif config.max_length and config.max_new_tokens:
        max_length = inputs_seq_length + config.max_new_tokens
    else:
        raise ValueError("You must set either `max_new_tokens` or `max_length`.")
    return max_length
