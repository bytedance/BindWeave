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

import os
import argparse
import json
import imageio
import requests
from PIL import Image
import torch
import io
import mindspeed.megatron_adaptor
import torch.distributed as dist
from megatron.core import mpu
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training import get_args

from mindspeed_mm.configs.config import merge_mm_args, mm_extra_args_provider
from mindspeed_mm.arguments import extra_args_provider_decorator
from mindspeed_mm.tasks.inference.pipeline import sora_pipeline_dict
from mindspeed_mm.tasks.inference.pipeline.utils.sora_utils import (
    save_videos,
    save_video_grid,
    load_prompts,
    load_images,
    load_conditional_pixel_values,
    load_videos,
)
from mindspeed_mm.models.predictor import PredictModel
from mindspeed_mm.models.diffusion import DiffusionModel
from mindspeed_mm.models.ae import AEModel
from mindspeed_mm.models.text_encoder import TextEncoder
from mindspeed_mm import Tokenizer
from mindspeed_mm.utils.utils import get_dtype, get_device, is_npu_available
from s2v.tools.inference.vlm_extend import ArkClient
if is_npu_available():
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.npu.config.allow_internal_format = False


def prepare_pipeline(args, device):
    ori_args = get_args()
    args.pipeline_config.seed = ori_args.seed
    vae = AEModel(args.ae).get_model().to(device, args.ae.dtype).eval()
    text_encoder = TextEncoder(args.text_encoder).get_model().to(device).eval()
    predict_model = PredictModel(args.predictor).get_model()
    if ori_args.load is not None:
        load_checkpoint([predict_model], None, None, strict=True)
    predict_model = predict_model.to(device, args.predictor.dtype).eval()
    scheduler = DiffusionModel(args.diffusion).get_model()
    tokenizer = Tokenizer(args.tokenizer).get_tokenizer()
    if not hasattr(vae, 'dtype'):
        vae.dtype = args.ae.dtype
    sora_pipeline_class = sora_pipeline_dict[args.pipeline_class]
    sora_pipeline = sora_pipeline_class(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler,
                                        predict_model=predict_model, config=args.pipeline_config)
    return sora_pipeline



def download(value, retry=3, timeout=20):
    err = None
    for _ in range(retry+1):
        try:
            rsp = requests.get(value, timeout=timeout)
            assert rsp.status_code == 200, 'download {} error'.format(
                value)
            return rsp.content
        except Exception as e:
            err = e
    return err


def main():
    initialize_megatron(extra_args_provider=extra_args_provider_decorator(mm_extra_args_provider), args_defaults={})
    args = get_args()
    merge_mm_args(args)
    if not hasattr(args, "dist_train"):
        args.dist_train = False
    assert args.micro_batch_size == 1
    args.mm.model.micro_batch_size = args.micro_batch_size
    args = args.mm.model
    if not hasattr(args, "dual_image"):
        args.dual_image = False
    vlm_client = ArkClient(
            model_name="",
            api_key="",
            base_url=""
        )
    # prepare arguments
    torch.set_grad_enabled(False)
    dtype = get_dtype(args.dtype)
    device = get_device(args.device)
    strength = args.strength if hasattr(args, "strength") else None
    mask_type = args.mask_type if hasattr(args, "mask_type") else None
    crop_for_hw = args.crop_for_hw if hasattr(args, "crop_for_hw") else None
    max_hxw = args.max_hxw if hasattr(args, "max_hxw") else None
    strength = args.strength if hasattr(args, "strength") else None
    save_fps = args.fps // args.frame_interval

    # prepare pipeline
    from s2v.monkey_patch import patch_wan_s2v_dit, patch_wan_s2v_pipeline
    patch_wan_s2v_dit()
    patch_wan_s2v_pipeline()
    sora_pipeline = prepare_pipeline(args, device)

    # == Iter over all samples ==
    video_grids = []
    start_idx = 0
    rank = mpu.get_data_parallel_rank()
    world_size = mpu.get_data_parallel_world_size()


    meta_path = args.META_PATH
    base_img_dir = args.BASE_IMG_DIR  
    output_dir = args.OUT_DIR 
    os.makedirs(output_dir, exist_ok=True) 
    sample_meta_list = []

   
    with open(meta_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  
    sample_keys = sorted(data.keys())  # 如['singleobj_1', 'singleobj_2', ...]
    sample_meta_list = [data[key] for key in sample_keys]  

    batch_size = 1
    for i in range(rank * batch_size, len(sample_meta_list), batch_size * world_size):
        meta = sample_meta_list[i]
        # get hidden states
        hidden_states_path = meta['hidden_states_path']
        hidden_states_path_negative = meta['hidden_states_path_negative']
        hidden_states = torch.load(hidden_states_path) if hidden_states_path else None
        hidden_states_negative = torch.load(hidden_states_path_negative) if hidden_states_path_negative else None
        current_key = sample_keys[i]  

        img_rel_path = meta['img_paths'][0]
        img_abs_path = os.path.join(base_img_dir, img_rel_path)
        try:
            main_image = Image.open(img_abs_path).convert('RGB')
        except Exception as e:
            print(f"can not load main image {img_abs_path}: {e}")
            continue  
        
        ref_image_list = []
        for rel_path in meta['img_paths']:
            ref_img_abs_path = os.path.join(base_img_dir, rel_path)
            try:
                ref_image = Image.open(ref_img_abs_path).convert('RGB')
                ref_image_list.append(ref_image)
            except Exception as e:
                print(f"can not load reference images {ref_img_abs_path}: {e}")
        sample = {
            'prompt': meta['prompt'],
            'image': main_image,       
            'ref_image_list': ref_image_list,  
            'hidden_states': hidden_states, 
            'hidden_states_negative': hidden_states_negative
        }
        # image transform process; get vae latents; get clip latents; get text
        videos = sora_pipeline.predict_raw_samples(sample_list=[sample])

        video_grids.append(videos)
        if mpu.get_context_parallel_rank() == 0 and mpu.get_tensor_model_parallel_rank() == 0:

            save_path = os.path.join(output_dir, f"{current_key}.mp4")
            imageio.mimwrite(save_path, videos[0], fps=save_fps, quality=6)

    if len(video_grids) > 0:
        video_grids = torch.cat(video_grids, dim=0).to(device)

    if len(sample_meta_list) < args.micro_batch_size * world_size:
        active_ranks = range(len(sample_meta_list) // args.micro_batch_size)
    else:
        active_ranks = range(world_size)
    active_ranks = [x * mpu.get_tensor_model_parallel_world_size() * mpu.get_context_parallel_world_size() for x in active_ranks]

    dist.barrier()
    gathered_videos = []
    rank = dist.get_rank()
    if rank == 0:
        for r in active_ranks:
            if r != 0:  # main process does not need to receive from itself
                # receive tensor shape
                shape_tensor = torch.empty(5, dtype=torch.int, device=device)
                dist.recv(shape_tensor, src=r)
                shape_videos = shape_tensor.tolist()

                # create receiving buffer based on received shape
                received_videos = torch.empty(shape_videos, dtype=video_grids.dtype, device=device)
                dist.recv(received_videos, src=r)
                gathered_videos.append(received_videos.cpu())
            else:
                gathered_videos.append(video_grids.cpu())
    elif rank in active_ranks:
        # send tensor shape first
        shape_tensor = torch.tensor(video_grids.shape, dtype=torch.int, device=device)
        dist.send(shape_tensor, dst=0)

        # send the tensor
        dist.send(video_grids, dst=0)
    dist.barrier()
    if rank == 0:
        video_grids = torch.cat(gathered_videos, dim=0)
        save_video_grid(video_grids, args.save_path, save_fps)
        print("Inference finished.")
        print("Saved %s samples to %s" % (video_grids.shape[0], args.save_path))


if __name__ == "__main__":
    main()