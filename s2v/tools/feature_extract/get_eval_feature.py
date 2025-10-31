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

import copy
import json
import os
import random
import time
from typing import List, Optional, Union
import numpy as np
import mindspeed.megatron_adaptor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed
from megatron.core import mpu
from megatron.training import get_args, print_rank_0
from megatron.training.initialize import initialize_megatron, set_jit_fusion_options
from numpy import save
from typing import Dict, Any, List, Tuple
from pycocotools import mask as mask_util
from PIL import Image
import cv2
from tqdm import tqdm
from mindspeed_mm.configs.config import merge_mm_args, mm_extra_args_provider
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset 
from mindspeed_mm.data.data_utils.constants import (
    FILE_INFO,
    PROMPT_IDS,
    PROMPT_MASK,
    VIDEO,
    VIDEO_MASK,
)
from mindspeed_mm.data.data_utils.transform_pipeline import get_transforms
from s2v.data.datasets.t2v_dataset import T2VDataset
from mindspeed_mm.models.ae import AEModel
from mindspeed_mm.models.text_encoder import TextEncoder
from mindspeed_mm.tools.profiler import Profiler
from mindspeed_mm.utils.utils import get_device, get_dtype, is_npu_available
import gc
from mindspeed_mm.utils.extra_processor.i2v_processors import I2V_PROCESSOR_MAPPINGS
# from mindspeed_mm.tasks.inference.pipeline.qwen2vl_pipeline import Qwen2VlPipeline
from s2v.tasks.inference.pipeline.qwen2vl_pipeline import Qwen2VlPipeline
from filelock import FileLock  

from s2v.utils.extra_processor.wan_i2v_processor import WanVideoI2VProcessor
I2V_PROCESSOR_MAPPINGS["wan_i2v_processor"] = WanVideoI2VProcessor

if is_npu_available():
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.npu.config.allow_internal_format = False

NEGATIVE_PROMOPT = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards, split screen, divided frame, frame split, multiple parts, separated,motionless, frozen, still image, no movement, stationary, unmoving"

def prepare_model(args):

    qwen_pipeline = Qwen2VlPipeline(args.mm.model.infer_config)
    return qwen_pipeline


def process_opens2v_eval(qwen_pipeline, json_path, output_dir):
    """
    Processes the OpenS2V-Eval JSON file to extract features and save them.

    Args:
        qwen_pipeline: The Qwen2VlPipeline for feature extraction.
        json_path (str): The path to the input JSON file.
        output_dir (str): The directory to save the feature files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)
    base_img_dir = 'OpenS2V-Eval'
    if torch.distributed.get_rank() == 0:
        items = tqdm(data.items(), desc='Processing eval cases')
    else:
        items = data.items()
    
    for key, value in items:
        prompt = value['prompt']
        img_paths = value['img_paths']
        ref_pil_images = []
        for img_path in img_paths:
            full_img_path = os.path.join(base_img_dir, img_path)
            if os.path.exists(full_img_path):
                image = Image.open(full_img_path).convert('RGB')
                width, height = image.size
                if width*height > 720*1280:
                    new_size = (int(width * 0.3), int(height * 0.3))
                else:
                    new_size = (int(width * 0.75), int(height * 0.75))
                resized_image = image.resize(new_size)
                ref_pil_images.append(resized_image)
            else:
                print(f"Warning: Image not found at {full_img_path}")

        if not ref_pil_images:
            print(f"Warning: No images found for case {key}, skipping.")
            continue

        # Extract hidden states
        with torch.no_grad():
            hidden_states = qwen_pipeline(prompt, ref_pil_images[:4])
            hidden_states_negative = qwen_pipeline(NEGATIVE_PROMOPT, ref_pil_images[:4])
            hidden_states = hidden_states.permute(1, 0, 2)
            hidden_states_negative = hidden_states_negative.permute(1, 0, 2)
        # Only save files from rank 0
        if torch.distributed.get_rank() == 0:
            # Save hidden states
            feature_filename = f"{key}_hidden_states.pt"
            feature_path = os.path.join(output_dir, feature_filename)
            torch.save(hidden_states.cpu(), feature_path)
            feature_path_negative = os.path.join(output_dir, f"{key}_hidden_states_negative.pt")
            torch.save(hidden_states_negative.cpu(), feature_path_negative)

            # Update JSON data
            data[key]['hidden_states_path'] = feature_path
            data[key]['hidden_states_path_negative'] = feature_path_negative

    # Save updated JSON only from rank 0
    if torch.distributed.get_rank() == 0:
        output_json_path = os.path.join(os.path.dirname(json_path), "Open-Domain_Eval_with_features.json")
        with open(output_json_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Processing complete. Updated JSON saved to {output_json_path}")


def extract_feature():
    initialize_megatron(extra_args_provider=mm_extra_args_provider, args_defaults={})
    args = get_args()
    merge_mm_args(args)
    

    qwen_pipeline = prepare_model(args)
    process_opens2v_eval(qwen_pipeline, './s2v/OpenS2V-Eval/Open-Domain_Eval.json', './s2v/OpenS2V-Eval/features')

    
    


if __name__ == "__main__":
    extract_feature()