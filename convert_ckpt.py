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


import argparse
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any, Dict, List

import mindspeed.megatron_adaptor
import torch
from safetensors.torch import load_file as safe_load
from checkpoint.vlm_model.mm_to_hf import save_by_index_json, copy_files_except_suffix
from checkpoint.vlm_model.hf_to_mm import load_from_hf

DIT_CONVERSION_MAPPING = {
    "condition_embedder.text_embedder.linear_1.bias": "text_embedding.linear_1.bias",
    "condition_embedder.text_embedder.linear_1.weight": "text_embedding.linear_1.weight",
    "condition_embedder.text_embedder.linear_2.bias": "text_embedding.linear_2.bias",
    "condition_embedder.text_embedder.linear_2.weight": "text_embedding.linear_2.weight",
    "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
    "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
    "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
    "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
    "condition_embedder.time_proj.bias": "time_projection.1.bias",
    "condition_embedder.time_proj.weight": "time_projection.1.weight",
    "condition_embedder.image_embedder.ff.net.0.proj.weight": "img_emb.proj.1.weight",
    "condition_embedder.image_embedder.ff.net.0.proj.bias": "img_emb.proj.1.bias",
    "condition_embedder.image_embedder.ff.net.2.weight": "img_emb.proj.3.weight",
    "condition_embedder.image_embedder.ff.net.2.bias": "img_emb.proj.3.bias",
    "condition_embedder.image_embedder.norm1.weight": "img_emb.proj.0.weight",
    "condition_embedder.image_embedder.norm1.bias": "img_emb.proj.0.bias",
    "condition_embedder.image_embedder.norm2.weight": "img_emb.proj.4.weight",
    "condition_embedder.image_embedder.norm2.bias": "img_emb.proj.4.bias",
    "condition_embedder.image_embedder.pos_embed": "img_emb.emb_pos",
    "scale_shift_table": "head.modulation",
    "proj_out.bias": "head.head.bias",
    "proj_out.weight": "head.head.weight",
    "condition_embedder.text_projector.linear_1.weight": "text_projector.linear_1.weight",
    "condition_embedder.text_projector.linear_1.bias":   "text_projector.linear_1.bias",
    "condition_embedder.text_projector.linear_2.weight": "text_projector.linear_2.weight",
    "condition_embedder.text_projector.linear_2.bias":   "text_projector.linear_2.bias",
}


class CheckpointConverter:
    def __init__(self, source_path: str, ckpt_path: str, target_path: str, mode: str, pp_vpp_layers: list):
        self.source_path = source_path
        self.ckpt_path = ckpt_path
        self.target_path = target_path
        self.mode = mode
        self.state_dict = None

        self.pp_vpp_layers = pp_vpp_layers
        if pp_vpp_layers and isinstance(pp_vpp_layers[0], list):
            self.vpp_size = len(pp_vpp_layers)
            self.pp_size = len(pp_vpp_layers[0])
        else:
            self.vpp_size = 1
            self.pp_size = len(pp_vpp_layers) if pp_vpp_layers is not None else 1

    def load_weight(self, _weight_path):
        if _weight_path.endswith(".safetensors"):
            return safe_load(_weight_path)
        else:
            return torch.load(_weight_path, map_location="cpu")

    def load_from_mm(self, _load_dir: str) -> list[dict]:
        LATEST_TXT = "latest_checkpointed_iteration.txt"
        mm_save_dir = Path(_load_dir)
        save_iteration = mm_save_dir.joinpath(LATEST_TXT).read_text()
        save_dir = mm_save_dir.joinpath(
            f"iter_{int(save_iteration):07}"
            if save_iteration != "release"
            else save_iteration
        )

        current_path = save_dir.joinpath("mp_rank_00")
        pt_path = current_path.joinpath("model_optim_rng.pt")

        state_dicts = torch.load(pt_path, map_location="cpu")["model"]

        return state_dicts

    def replace_state_dict(
            self,
            state_dict: Dict[str, Any],
            conversion_mapping: Dict,
    ) -> Dict[str, Any]:
        for ori_key, mm_key in conversion_mapping.items():
            if self.mode == "convert_to_hf" and mm_key in state_dict.keys():
                state_dict[ori_key] = state_dict.pop(mm_key)
            elif self.mode == "convert_to_mm" and ori_key in state_dict.keys():
                state_dict[mm_key] = state_dict.pop(ori_key)
        return state_dict

    def convert_attn_to_mm(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        new_checkpoint = {}
        state_dict = state_dict.get("blocks", state_dict)

        for key, value in state_dict.items():
            if "norm_added_q" in key:  # keys to ignore
                continue
            new_key = key.replace("attn1.norm_q", "self_attn.q_norm")
            new_key = new_key.replace("attn1.norm_k", "self_attn.k_norm")
            new_key = new_key.replace("attn2.norm_q", "cross_attn.q_norm")
            new_key = new_key.replace("attn2.norm_k", "cross_attn.k_norm")
            new_key = new_key.replace("attn1.to_q.", "self_attn.proj_q.")
            new_key = new_key.replace("attn1.to_k.", "self_attn.proj_k.")
            new_key = new_key.replace("attn1.to_v.", "self_attn.proj_v.")
            new_key = new_key.replace("attn1.to_out.0.", "self_attn.proj_out.")
            new_key = new_key.replace("attn2.to_q.", "cross_attn.proj_q.")
            new_key = new_key.replace("attn2.to_k.", "cross_attn.proj_k.")
            new_key = new_key.replace("attn2.to_v.", "cross_attn.proj_v.")
            new_key = new_key.replace("attn2.add_k_proj", "cross_attn.k_img")
            new_key = new_key.replace("attn2.add_v_proj", "cross_attn.v_img")
            new_key = new_key.replace("attn2.norm_added_k", "cross_attn.k_norm_img")
            new_key = new_key.replace("attn2.to_out.0.", "cross_attn.proj_out.")
            new_key = new_key.replace(".ffn.net.0.proj.", ".ffn.0.")
            new_key = new_key.replace(".ffn.net.2.", ".ffn.2.")
            new_key = new_key.replace("scale_shift_table", "modulation")
            new_key = new_key.replace(".norm2.", ".norm3.")

            new_checkpoint[new_key] = value

        return new_checkpoint

    def convert_attn_to_hf(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        new_checkpoint = {}
        state_dict = state_dict.get("blocks", state_dict)

        for key, value in state_dict.items():
            new_key = key.replace("self_attn.q_norm.", "attn1.norm_q.")
            new_key = new_key.replace("self_attn.k_norm.", "attn1.norm_k.")
            new_key = new_key.replace("cross_attn.q_norm.", "attn2.norm_q.")
            new_key = new_key.replace("cross_attn.k_norm.", "attn2.norm_k.")
            new_key = new_key.replace("self_attn.proj_q.", "attn1.to_q.")
            new_key = new_key.replace("self_attn.proj_k.", "attn1.to_k.")
            new_key = new_key.replace("self_attn.proj_v.", "attn1.to_v.")
            new_key = new_key.replace("self_attn.proj_out.", "attn1.to_out.0.")
            new_key = new_key.replace("cross_attn.proj_q.", "attn2.to_q.")
            new_key = new_key.replace("cross_attn.proj_k.", "attn2.to_k.")
            new_key = new_key.replace("cross_attn.proj_v.", "attn2.to_v.")
            new_key = new_key.replace("cross_attn.k_img.", "attn2.add_k_proj.")
            new_key = new_key.replace("cross_attn.v_img.", "attn2.add_v_proj.")
            new_key = new_key.replace("cross_attn.k_norm_img.", "attn2.norm_added_k.")
            new_key = new_key.replace("cross_attn.proj_out.", "attn2.to_out.0.")
            new_key = new_key.replace(".ffn.0.", ".ffn.net.0.proj.")
            new_key = new_key.replace(".ffn.2.", ".ffn.net.2.")
            new_key = new_key.replace("modulation", "scale_shift_table")
            new_key = new_key.replace(".norm3.", ".norm2.")

            new_checkpoint[new_key] = value

        return new_checkpoint

    def split_by_index_json(
            self, state_dict: Dict[str, Any], _model_path: str
    ) -> list[dict]:
        index_json_path = os.path.join(
            _model_path, "diffusion_pytorch_model.safetensors.index.json"
        )
        return_dicts = []
        with open(index_json_path, "r", encoding="utf-8") as file:
            weight_map = json.load(file)["weight_map"]
        for key, value in weight_map.items():
            if "norm_added_q" in key:  # keys to ignore
                continue
            index = int(value.split("-")[1])
            while index > len(return_dicts):
                return_dicts.append({})
            if key not in state_dict:
                raise KeyError(f"Key '{key}' not found in converted state_dict. "
                               f"Please ensure it is present in DIT_CONVERSION_MAPPING or generated upstream.")
            return_dicts[index - 1][key] = state_dict[key]
        return return_dicts

    def split_by_pp_vpp(self, state_dicts):
        state_dicts = [state_dicts, ]
        return_dict = []
        if self.pp_size <= 1:
            for tp_rank, state_dict in enumerate(state_dicts):
                return_dict.append((tp_rank, state_dict))
            return return_dict

        if self.vpp_size > 1:
            pp_sizes_flat = [
                layers
                for vpp_layer in self.pp_vpp_layers
                for layers in vpp_layer
            ]
        else:
            pp_sizes_flat = self.pp_vpp_layers

        print(f"pp_sizes_flat: {pp_sizes_flat}")
        postprocess_weight_names = ['head.head.weight', 'head.head.bias', 'head.modulation']
        for pp_rank, _ in enumerate(pp_sizes_flat):
            is_first = pp_rank == 0
            is_last = pp_rank == len(pp_sizes_flat) - 1
            start_layer, end_layer = sum(pp_sizes_flat[:pp_rank]), sum(pp_sizes_flat[:pp_rank + 1])
            for tp_rank, state_dict in enumerate(state_dicts):
                pp_tp_param = dict()
                for k in list(state_dict.keys()):
                    if k.startswith("blocks"):
                        idx = int(k.split('.')[1])
                        if start_layer <= idx < end_layer:
                            cur_idx, tmps = str(idx - start_layer), k.split('.')
                            new_k = '.'.join(tmps[:1] + [cur_idx] + tmps[2:])
                            pp_tp_param[new_k] = state_dict[k]
                    elif k in postprocess_weight_names:
                        # for pp rank -1
                        if is_last:
                            pp_tp_param[k] = state_dict[k]
                    else:
                        # for pp rank 0
                        if is_first:
                            pp_tp_param[k] = state_dict[k]
                return_dict.append((tp_rank, pp_tp_param))

        return return_dict

    def save_to_mm(self, state_dicts, save_dir, latest_checkpointed_iteration="release"):
        if os.path.exists(save_dir):
            print(f"save dir: {save_dir} exists, please check.")
            return
        else:
            os.makedirs(save_dir)

        flags = os.O_WRONLY | os.O_CREAT
        stat_mode = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(os.path.join(save_dir, "latest_checkpointed_iteration.txt"), flags, stat_mode),
                       "w") as fout:
            fout.write(latest_checkpointed_iteration)
        if latest_checkpointed_iteration == "release":
            directory = "release"
        else:
            directory = "iter_{:07d}".format(latest_checkpointed_iteration)

        if self.pp_size <= 1:
            for tp_rank, state_dict in state_dicts:
                save_dict = {}
                filename = f"mp_rank_{tp_rank:02d}"
                os.makedirs(os.path.join(save_dir, directory, filename))
                save_path = os.path.join(save_dir, directory, filename, "model_optim_rng.pt")
                save_dict["model"] = state_dict
                torch.save(save_dict, save_path)
            return

        for pp_rank in range(self.pp_size):
            tp_rank, state_dict = state_dicts[pp_rank]
            os.makedirs(os.path.join(save_dir, directory, f"mp_rank_{tp_rank:02d}_{pp_rank:03d}"))
            save_path = os.path.join(save_dir, directory, f"mp_rank_{tp_rank:02d}_{pp_rank:03d}", "model_optim_rng.pt")

            save_dict = {}
            if self.vpp_size > 1:
                save_dict = {
                    f"model{vpp_rank}": state_dicts[vpp_rank * self.pp_size + pp_rank][1]
                    for vpp_rank in range(self.vpp_size)
                }
                save_dict['checkpoint_version'] = 3.0
            else:
                save_dict["model"] = state_dict
            torch.save(save_dict, save_path)

    def forward(self):
        if self.mode == "convert_to_hf":
            self.state_dict = self.load_from_mm(self.source_path)
            self.state_dict = self.replace_state_dict(
                self.state_dict, conversion_mapping=DIT_CONVERSION_MAPPING
            )
            self.state_dict = self.convert_attn_to_hf(self.state_dict)
            self.state_dict = self.split_by_index_json(self.state_dict, self.ckpt_path)
            copy_files_except_suffix(Path(self.ckpt_path), Path(self.target_path))
            save_by_index_json(self.state_dict, self.target_path)
            print("Checkpoint successfully converted from MM to Hugging Face format.")
        elif self.mode == "convert_to_mm":
            self.state_dict = load_from_hf(Path(self.source_path))
            self.state_dict = self.replace_state_dict(
                self.state_dict, conversion_mapping=DIT_CONVERSION_MAPPING
            )
            self.state_dict = self.convert_attn_to_mm(self.state_dict)
            self.state_dict = self.split_by_pp_vpp(self.state_dict)
            self.save_to_mm(self.state_dict, self.target_path)
            print("Checkpoint successfully converted from Hugging Face to MM format.")

        else:
            raise ValueError(
                "please select the mode only from convert_to_hf or convert_to_mm"
            )


def parse_nested_list(value):
    try:
        pp_vpp_layers = json.loads(value)
        if not isinstance(pp_vpp_layers, list):
            raise argparse.ArgumentTypeError(f"Invalid input list format:{value}")
        return pp_vpp_layers
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid input list format:{value}, {e}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_path",
        type=str,
        default="./ckpt/wan2.1/iter_####",
        help="Source path of trained model for mm to hf, or path of weights for transformer",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./Wan2.1-T2V-14B-Diffusers/transformer",
        help="Checkpoint path of original model, only used for mm to hf mode",
    )
    parser.add_argument(
        "--target_path",
        type=str,
        default="./save_ckpt/wan2.1/",
        help="Save path of MM checkpoint for mm to hf mode, or output path for tranformed weight from hf to mm",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["convert_to_hf", "convert_to_mm"],
        required=True,
        help="Selection of conversion mode: convert_to_hf or convert_to_mm",
    )
    parser.add_argument(
        "--pp_vpp_layers",
        type=parse_nested_list,
        required=False,
        default=[],
        help="The pp and vpp layers per stage, only used for convert to mm : default means disable, \
             '[7, 8, 8, 7]' meas pp_size=4, '[[7, 8], [7,8]]' means pp_size=2 and vpp_size=2"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    converter = CheckpointConverter(
        source_path=args.source_path,
        ckpt_path=args.ckpt_path,
        target_path=args.target_path,
        mode=args.mode,
        pp_vpp_layers=args.pp_vpp_layers
    )
    converter.forward()
