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
    
    full_args = get_args()
    merge_mm_args(full_args)

    args = full_args.mm.model
    
    assert full_args.micro_batch_size == 1
    args.micro_batch_size = full_args.micro_batch_size

    if not hasattr(args, "dual_image"):
        args.dual_image = False

    vlm_client = ArkClient(
        model_name="",
        api_key="",
        base_url=""
    )


    torch.set_grad_enabled(False)
    rank = dist.get_rank()  
    world_size = dist.get_world_size() 
    
    torch.cuda.set_device(full_args.local_rank)

    from s2v.monkey_patch import patch_wan_r2v_dit, patch_wan_r2v_pipeline
    patch_wan_r2v_dit()
    patch_wan_r2v_pipeline()


    meta_path = './Open-Domain_Eval.json'
    base_img_dir = './OpenS2V-Eval'
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)

    with open(meta_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sample_keys = sorted(data.keys())
    sample_meta_list = [data[key] for key in sample_keys]
    

    local_results = []
    batch_size = 1
    for i in range(rank * batch_size, len(sample_meta_list), batch_size * world_size):
        meta = sample_meta_list[i].copy()
        current_key = sample_keys[i]

        ref_image_list = []
        for rel_path in meta['img_paths']:
            ref_img_abs_path = os.path.join(base_img_dir, rel_path)
            try:
                ref_image = Image.open(ref_img_abs_path).convert('RGB')
                ref_image_list.append(ref_image)
            except Exception as e:
                print(f"Rank {rank}: cannot load reference image {ref_img_abs_path}: {e}")
        
        image_bytes_list = []
        for image in ref_image_list:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG', quality=90)
            image_bytes_list.append(img_byte_arr.getvalue())

        if not image_bytes_list:
            refined_prompt = meta['prompt']
        else:
            image_placeholders = ' '.join(['<image>'] * len(image_bytes_list))
            vlm_prompt = f"""
You are an expert in composing prompts for video generation. Your task is to rewrite a user's prompt to create a single, unified, and coherent scene, based on reference images.

**Critical Goal:** The final video must NOT have any "split-screen", "twinning", or "double-subject" artifacts. The output must be a single, continuous scene.

**How to achieve this:**
1.  **Describe a Unified Scene:** Combine the user's idea and the visual details from the images into a description of ONE single, coherent scene. Do not describe multiple separate scenes.
2.  **Focus on Composition:** Describe the subject(s) and their relationship to the environment. Use compositional language (e.g., "a character in the center of the frame," "a wide shot of a person walking through a forest").
3.  **Maintain Subject Consistency:** Use the key visual features from the reference images (clothing, appearance, etc.) to describe the subject(s) accurately.
4.  **Natural Integration:** The description should make the subject feel naturally part of the environment, with realistic lighting and three-dimensionality, not like a "copy-pasted" sticker.
5.  **Describe Clear and Significant Motion:** Clearly describe the main action or movement in the scene. Use descriptive verbs to convey the motion (e.g., "walking briskly," "gently swaying," "gliding across"). You can also suggest camera movements like "a slow pan" or "a gentle zoom" to enhance the sense of movement. The motion should be an integral part of the scene.
**Input:**
- **User's Idea:** "{{original_prompt}}"
- **Reference Images:** {image_placeholders}

**Your Task:**
- Rewrite the user's idea into a single, descriptive paragraph that clearly describes the scene and its motion.
- The rewritten prompt must be in English.
- Your output must ONLY be the rewritten prompt.
"""
            final_vlm_prompt = vlm_prompt.format(original_prompt=meta['prompt'])
            response = vlm_client.predict(prompt=final_vlm_prompt, image_list=image_bytes_list)
            refined_prompt = response.get('content', meta['prompt']).strip()

        print("-" * 50)
        print(f"Rank {rank}, Sample {i+1} | Images used: {len(image_bytes_list)}")
        print(f"Original Prompt: {meta['prompt']}")
        print(f"Refined Prompt: {refined_prompt}")
        print("-" * 50)
        
        meta['prompt_old'] = meta['prompt']
        meta['prompt'] = refined_prompt
        
        local_results.append((current_key, meta))


    dist.barrier()

    all_results_list = [None] * world_size  
    dist.all_gather_object(all_results_list, local_results) 


    if rank == 0:
        print("Consolidating results and writing to file...")
        for rank_results in all_results_list:
            if rank_results:
                for key, updated_meta in rank_results:
                    data[key] = updated_meta
        
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Successfully updated {meta_path}")


if __name__ == "__main__":
    main()