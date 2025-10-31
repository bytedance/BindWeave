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
import io
import torch.nn as nn
import torch.nn.functional as F
from filelock import FileLock
import json, os
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
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset # 对应常规的Dataloader和Dataset类
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
from s2v.tasks.inference.pipeline.qwen2vl_pipeline import Qwen2VlPipeline
from filelock import FileLock  
from s2v.tools.feature_extract.vlm_extend import ArkClient
from s2v.utils.extra_processor.wan_i2v_processor import WanVideoI2VProcessor
I2V_PROCESSOR_MAPPINGS["wan_i2v_processor"] = WanVideoI2VProcessor

if is_npu_available():
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.npu.config.allow_internal_format = False




def rle_to_mask(rle, img_width, img_height):
    rle_obj = {"counts": rle["counts"].encode("utf-8"), "size": [img_height, img_width]}
    return mask_util.decode(rle_obj)

def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    largest_contour = max(contours, key=cv2.contourArea)
    polygon = largest_contour.reshape(-1, 2).tolist()
    return polygon

def polygon_to_mask(
    polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]
) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], color=(1,))
    return mask

def refine_masks(
    masks: torch.BoolTensor, polygon_refinement: bool = False
) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)
    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask
    return masks


def pad_to_size(img_np, target_height, target_width, pad_value=255):
    """
    Pads a NumPy image array (HWC or HW) to the target size.
    - img_np: (H, W, C) or (H, W)
    - target_height: desired height
    - target_width: desired width
    - pad_value: pixel value to pad with (0=black, 255=white, etc.)
    """
    h, w = img_np.shape[:2]

    # Compute how much to pad
    pad_top = (target_height - h) // 2
    pad_bottom = target_height - h - pad_top
    pad_left = (target_width - w) // 2
    pad_right = target_width - w - pad_left

    # Create pad widths
    if img_np.ndim == 3: # h,w,c
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    else:  # grayscale or mask
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))

    # Apply padding
    padded_img = np.pad(img_np, pad_width, mode='constant', constant_values=pad_value)
    return padded_img


def convert_to_size(img_np, target_height, target_width, pad_value=0):
    h = img_np.shape[0]
    w = img_np.shape[1]

    if h > target_height:
        top_crop = (h-target_height) // 2
        bottom_crop = h - target_height - top_crop
        img_np = img_np[top_crop:h-bottom_crop]
    if w > target_width:
        left_crop = (w-target_width) // 2
        right_crop = w - target_width - left_crop
        img_np = img_np[:,left_crop:w-right_crop]
    
    h = img_np.shape[0]
    w = img_np.shape[1]
    if (h < target_height) or (w < target_width):
        img_np = pad_to_size(img_np=img_np, target_height=target_height, target_width=target_width, pad_value=pad_value)
    return img_np

class WanTextVideoDataset(T2VDataset):
    def __init__(
        self,
        task,
        save_vlm_cache,
        basic_param,
        vid_img_process: dict,
        use_text_processer: bool = False,
        enable_text_preprocessing: bool = True,
        text_preprocess_methods: Optional[Union[dict, List[dict]]] = None,
        tokenizer_config: Optional[Union[dict, List[dict]]] = None,
        **kwargs,
    ):
        video_only_transforms = vid_img_process.get("train_pipeline", {}).get("video_only", None)
        if video_only_transforms is None:
            raise ValueError('"video_only" key not found in vid_img_process["train_pipeline"]')

        video_and_first_frame_transforms = vid_img_process.get("train_pipeline", {}).get("video_and_first_frame", None)
        if video_and_first_frame_transforms is None:
            raise ValueError('"video_and_first_frame" key not found in vid_img_process["train_pipeline"]')

        video_only_preprocess = {"video": video_only_transforms}
        vid_img_process["train_pipeline"] = {"video": video_and_first_frame_transforms}
        
        video_only_preprocess = {"video": video_only_transforms}
        vid_img_process["train_pipeline"] = {"video": video_and_first_frame_transforms}
        super().__init__(
            basic_param=basic_param,
            vid_img_process=vid_img_process,
            use_text_processer=use_text_processer,
            enable_text_preprocessing=enable_text_preprocessing,
            text_preprocess_methods=text_preprocess_methods,
            tokenizer_config=tokenizer_config
        )

        self.TARGET_H, self.TARGET_W = vid_img_process['max_height'], vid_img_process['max_width']
        self.video_only_preprocess = get_transforms(
            is_video=True, 
            train_pipeline=video_only_preprocess,
            transform_size={"max_height": vid_img_process['max_height'], "max_width": vid_img_process['max_width']}
        )
        self.task = task # t2v or i2v


    def append_cache_line_threadsafe(self, outer_key, new_decisions):
        if not self.vlm_cache_path or not new_decisions:
            return
        os.makedirs(os.path.dirname(self.vlm_cache_path), exist_ok=True)
        lock = FileLock(self.vlm_cache_path + ".lock")
        with lock:  
            with open(self.vlm_cache_path, 'a') as f:
                json.dump({outer_key: new_decisions}, f)
                f.write('\n')


    def process_background_image(self, background_base_path, main_part, cross_part, outer_key):

        try:
            background_image = Image.open(
                os.path.join(
                    background_base_path.replace(main_part, cross_part),
                    f"{outer_key.split('_step6')[0] +'_step7'}.png",
                )
            ).convert('RGB')  
        except Exception as e:
            print(f"Error loading background image: {e}")
            background_image = None  
            return None, None

        original_width, original_height = background_image.size
        target_width, target_height = 1280, 720
        

        original_ratio = original_width / original_height
        target_ratio = target_width / target_height
        

        if original_ratio > target_ratio:

            scale = target_width / original_width
        else:

            scale = target_height / original_height
        
 
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        

        resized_image = background_image.resize((new_width, new_height), Image.LANCZOS)
        

        pad_width = (target_width - new_width) // 2
        pad_height = (target_height - new_height) // 2
        
  
        pad_width_remainder = (target_width - new_width) % 2
        pad_height_remainder = (target_height - new_height) % 2
        

        resized_np = np.array(resized_image)
        

        pad_top = pad_height
        pad_bottom = pad_height + pad_height_remainder
        pad_left = pad_width
        pad_right = pad_width + pad_width_remainder
        

        padded_np = np.pad(
            resized_np,
            pad_width=((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='constant',
            constant_values=0
        )
        

        mask = np.zeros((target_height, target_width), dtype=np.uint8)

        mask[
            pad_top : pad_top + new_height,
            pad_left : pad_left + new_width
        ] = 1
        
        return padded_np, mask




    def get_cropped_subiect_image(
        self,
        input_image,
        annotation_idx,
        class_names,
        mask_data,
        image_width,
        image_height,
        bbox_data,
        use_bbox=False,
    ):
        # class_name = class_names[f"{annotation_idx}"]["class_name"]
        class_name = class_names[f"{annotation_idx}"].get("class_name", None)
        # gme_score = bbox_data[int(annotation_idx) - 1]["gme_score"]
        gme_score = bbox_data[int(annotation_idx) - 1].get("gme_score", 0)
        aes_score = bbox_data[int(annotation_idx) - 1].get("aes_score", 0)
        # aes_score = bbox_data[int(annotation_idx) - 1]["aes_score"]

        if use_bbox:
            bbox = bbox_data[int(annotation_idx) - 1]["bbox"]
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = bbox[2]
            y_max = bbox[3]

            x_min = int(max(x_min, 0))
            y_min = int(max(y_min, 0))
            x_max = int(min(x_max, image_width - 1))
            y_max = int(min(y_max, image_height - 1))

            resized_image = input_image[y_min : y_max + 1, x_min : x_max + 1]
        else:
            mask_rle = mask_data[annotation_idx]
            mask = rle_to_mask(mask_rle, image_width, image_height) # [720 1280]
            

            # Find the bounding box of the mask
            rows, cols = np.where(mask == 1)
            if len(rows) == 0 or len(cols) == 0:
                return None, None, 0, 0, 0

            y_min, y_max = np.min(rows), np.max(rows)
            x_min, x_max = np.min(cols), np.max(cols)

            # Adjust if the region goes out of bounds
            x_min = int(max(x_min, 0))
            y_min = int(max(y_min, 0))
            x_max = int(min(x_max, image_width - 1))
            y_max = int(min(y_max, image_height - 1))

            # Crop the region from the original image and mask
            cropped_image = input_image[y_min : y_max + 1, x_min : x_max + 1]
            cropped_mask = mask[y_min : y_max + 1, x_min : x_max + 1] # [h w]

            # Create a white background of the same size as the crop
            white_background = np.ones_like(cropped_image) * 255

            # Apply the mask to the cropped image
            white_background[cropped_mask == 1] = cropped_image[cropped_mask == 1]
            resized_image = white_background # [h w 3]

        pil_image = Image.fromarray(
            cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB).astype(np.uint8)
        ).convert("RGB")

        
        # Create RGBA array
        cropped_image = np.array(pil_image)
        alpha = (cropped_mask * 255).astype(np.uint8)
        rgba = np.dstack([cropped_image, alpha])
        rgba = Image.fromarray(rgba)
        max_rotate=45
        max_scale=1.5

        degree = np.random.randint(0,2*max_rotate) - max_rotate
        rotated = rgba.rotate(degree, expand=True, resample=Image.BICUBIC)

        W,H = rotated.size
        scale = np.exp((np.random.random() * 2 - 1) * max_scale)
        scale = min(scale, self.TARGET_H / H)
        scale = min(scale, self.TARGET_W / W)
        new_h, new_w = int(H * scale), int(W * scale)
        # Ensure dimensions are at least 1
        new_h = max(1, new_h)
        new_w = max(1, new_w)
        resized = rotated.resize((new_w, new_h), resample=Image.LANCZOS)
        # padding or crop
        image = np.array(resized)
        image_rgba = convert_to_size(img_np=image, target_height=self.TARGET_H, target_width=self.TARGET_W, pad_value=0) #rgbd

        crop_ratio = (pil_image.size[0] * pil_image.size[1]) / (
            image_height * image_width
        )

        return image_rgba, class_name, gme_score, aes_score, crop_ratio

    def __getitem__(self, index):
        example = {}
        sample = self.dataset_prog.cap_list[index]

        outer_key = list(sample.keys())[0]
        file_path = sample[outer_key]["metadata"]["path"]

        if not os.path.exists(file_path):
            raise AssertionError(f"file {file_path} do not exist!")
        frame_indice = sample[outer_key]["metadata"]["sample_frame_index"]
        vframes, info, is_decord_read = self.video_reader(file_path)

        # For input subject image
        # crop & cut
        s_x, e_x, s_y, e_y = sample[outer_key]["metadata"]["crop"] # [0 1280 0 720]
        frame_idx = sample[outer_key]["annotation"]["ann_frame_data"]["ann_frame_idx"]
        input_image = (
            vframes.get_batch([int(frame_idx)]).asnumpy()[0][..., ::-1].astype(np.uint8)
        )
        input_image = input_image[s_y:e_y, s_x:e_x]
        image_width = input_image.shape[1]
        image_height = input_image.shape[0]

        class_names = sample[outer_key]["annotation"]["mask_map"]
        bbox_data = sample[outer_key]["annotation"]["ann_frame_data"]["annotations"]
        mask_data = sample[outer_key]["annotation"]["mask_annotation"][str(frame_idx)]

        background_threshold = 10.0
        background_image = None
        background_ase_score = sample[outer_key]["annotation"]["ann_frame_data"]["background_ase_score"]
        main_part = sample[outer_key]["metadata"]["main_part"]        
        background_base_path = "s2v/OpenS2V-5M/Background"

        background_base_path = os.path.join(background_base_path, main_part)
        cross_part = main_part
        if background_ase_score >= background_threshold:
            background_np, backfround_mask = self.process_background_image(
                background_base_path=background_base_path,
                main_part=main_part,
                cross_part=cross_part,
                outer_key=outer_key
            )
        else:

            background_np = None  


        background_rgba = None
        if background_np is not None and backfround_mask is not None:
            alpha_channel_mask = (backfround_mask * 255).astype(np.uint8)
            background_rgba = np.dstack([background_np, alpha_channel_mask])

        batch_info = {"subjects": []}

        aes_threshold = 4.5 
        gme_threshold = 0.6 
        count_sub = 0
        ref_rgba_np_list = []
        crop_ratio_tmp = 0
        annotation_indices = list(mask_data.keys())
        for i, annotation_idx in enumerate(mask_data):
            subject_image_rgba, class_name, gme_score, aes_score, crop_ratio = (
                self.get_cropped_subiect_image(
                    input_image,
                    annotation_idx,
                    class_names,
                    mask_data,
                    image_width,
                    image_height,
                    bbox_data,
                    use_bbox=False,
                )
            )

            if subject_image_rgba is None:
                continue
            if crop_ratio > crop_ratio_tmp:
                last_valid_subject = subject_image_rgba  # 更新最后一个有效图像
            crop_ratio_tmp = crop_ratio

            if crop_ratio < 0.1:
                continue
            if aes_score < aes_threshold and gme_score < gme_threshold:
                continue
            ref_rgba_np_list.append(subject_image_rgba)


        if not ref_rgba_np_list and last_valid_subject is not None:
            ref_rgba_np_list.append(last_valid_subject)
            count_sub = 1  
        ref_rgba_np = np.array(ref_rgba_np_list) # THWC(rgba), uint8

        del ref_rgba_np_list
        ref_rgb_tensor = torch.tensor(ref_rgba_np[:,:,:,:3])
        ref_rgb_tensor = torch.permute(ref_rgb_tensor, (0, 3, 1, 2)) # TCHW, uint8
        ref_rgb_tensor = ref_rgb_tensor.float() / 255.0
        
        # process ref masks
        ref_mask_np = (ref_rgba_np[:,:,:,3] > 127).astype('int') # n h w
        del ref_rgba_np
        ref_mask_tensor = torch.tensor(ref_mask_np).float()
        del ref_mask_np
        example['ref_rgb_tensor'] = ref_rgb_tensor
        example['ref_rgb_tensor_mllm'] = ref_rgb_tensor
        example['ref_mask_tensor'] = ref_mask_tensor
        example["subjects"] = batch_info["subjects"]
        example["count_sub"] = count_sub
        start_frame_idx = sample[outer_key]["metadata"].get("start_frame_idx", 0)

        try:
            video = self.video_processer(
                vframes,
                info,
                is_decord_read=is_decord_read,
                predefine_num_frames=len(frame_indice),
                start_frame_idx=start_frame_idx,
                resolution_crop=sample[outer_key]["metadata"]["crop"]
            )

            if self.task == "i2v":
                if is_decord_read:
                    first_frame = video[:, 0, :, :] # c t h w 
                    example["first_frame"] = first_frame               
                else:
                    raise NotImplementedError(f"Only support video_reader_type: decoder.")
            
            video = self.video_only_preprocess(video)
            example[VIDEO] = video
        except Exception as e:   
            print("read error", file_path, e)
            example[VIDEO] = torch.zeros(3, 81, 720, 1280)

        text = sample[outer_key]["metadata"]["face_cap_qwen"] 
        if not isinstance(text, list):
            text = [text]
        text = [random.choice(text)]

        if self.use_text_processer:
            prompt_ids, prompt_mask = self.get_text_processer(text)
            example[PROMPT_IDS], example[PROMPT_MASK] = prompt_ids, prompt_mask
        else:
            example["text"] = text
        example["raw_text"] = text
        # for feature extract, trace source file name
        example[FILE_INFO] = file_path
        return example
    

def prepare_model(args, extract_video_feature, extract_text_feature, device):
    if extract_video_feature:
        vae = AEModel(args.mm.model.ae).to(device, args.mm.model.ae.dtype).eval()
    else:
        vae = None
    if extract_text_feature:
        text_encoder = TextEncoder(args.mm.model.text_encoder).to(device).eval()
    else:
        text_encoder = None
    return vae, text_encoder


def get_pt_name(file_name):
    pt_name = os.path.basename(file_name).replace(".", "_") + ".pt"
    return pt_name


def extract_feature():
    
    initialize_megatron(extra_args_provider=mm_extra_args_provider, args_defaults={})
    args = get_args()
    merge_mm_args(args)

    extract_video_feature = args.mm.tool.sorafeature.extract_video_feature
    extract_text_feature = args.mm.tool.sorafeature.extract_text_feature
    data_storage_mode = args.mm.tool.sorafeature.data_storage_mode
    
    save_path = args.mm.tool.sorafeature.save_path
    if torch.distributed.get_rank() == 0:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if data_storage_mode == 'standard':
            if not os.path.exists(os.path.join(save_path, 'videos')):
                os.makedirs(os.path.join(save_path, 'videos'))
            if not os.path.exists(os.path.join(save_path, 'labels')):
                os.makedirs(os.path.join(save_path, 'labels'))
            if not os.path.exists(os.path.join(save_path, 'images')):
                os.makedirs(os.path.join(save_path, 'images'))
        elif data_storage_mode == "sorafeatured":
            if not os.path.exists(os.path.join(save_path, 'features')):
                os.makedirs(os.path.join(save_path, 'features'))
        else:
            raise NotImplementedError(f"Data storage mode {data_storage_mode} is not implemented! ")

    
    set_jit_fusion_options()
    torch.set_grad_enabled(False)
    dtype = get_dtype(args.mm.model.ae.dtype)
    device = get_device("npu")

    dataset_param = args.mm.data.dataset_param.to_dict()
    task = args.mm.tool.task if hasattr(args.mm.tool, "task") else "t2v"
    save_vlm_cache = args.mm.tool.save_vlm_cache
    if task == "t2v":
        delattr(args.mm.model.ae, "i2v_processor")
    train_dataset = WanTextVideoDataset(
        task, 
        save_vlm_cache,
        dataset_param["basic_parameters"],
        dataset_param["preprocess_parameters"],
        **dataset_param
    )
    train_dataloader = build_mm_dataloader(
        train_dataset,
        args.mm.data.dataloader_param,
        process_group=mpu.get_data_parallel_group(),
        dataset_param=args.mm.data.dataset_param,
    )

    # master rank, write data info jsonl
    if torch.distributed.get_rank() == 0:
        jsonl_path = os.path.join(save_path, 'data_append.jsonl')
        lock_path = f"{jsonl_path}.lock"
        lock = FileLock(lock_path)

        with lock:
            with open(jsonl_path, 'a', encoding="utf-8") as json_file:
                for data_sample in train_dataset.data_samples:
                    source_data_storage_mode = args.mm.data.dataset_param.basic_parameters.data_storage_mode
                    if source_data_storage_mode == "combine":
                        source_file_key = "path"
                    elif source_data_storage_mode == "standard":
                        source_file_key = FILE_INFO
                    else: 
                        raise NotImplementedError(f"Extract features from data storage mode {source_data_storage_mode} is not implemented")
                    outer_key = list(data_sample.keys())[0]

                    file_name = data_sample[outer_key]["metadata"][source_file_key]
                    pt_name = get_pt_name(file_name)
                    required_fields = {
                        "path": data_sample[outer_key]["metadata"]["path"],
                        "cap": data_sample[outer_key]["metadata"]["face_cap_qwen"],
                        "num_frames": data_sample[outer_key]["metadata"]["num_frames"],
                        "fps": data_sample[outer_key]["metadata"]['fps'], 
                        "resolution": data_sample[outer_key]["metadata"]["resolution"],
                        "start_frame_idx": data_sample[outer_key]["metadata"]["start_frame_idx"],
                        "sample_frame_index": data_sample[outer_key]["metadata"]["sample_frame_index"],
                        "sample_num_frames": data_sample[outer_key]["metadata"]["sample_num_frames"]
                    }
                    data_info = copy.deepcopy(required_fields)
                    if data_storage_mode == "standard":
                        data_info.update({
                            "file": os.path.join('videos', pt_name),
                            "captions": os.path.join('labels', pt_name),
                            "image_latent": os.path.join('images', pt_name)
                        })
                    elif data_storage_mode == "sorafeatured":
                        data_info.update({
                            'file': f"features/{pt_name}"
                        })
                    json_file.write(json.dumps(data_info) + '\n')
        
    vae, text_encoder = prepare_model(args, extract_video_feature, extract_text_feature, device)
    
    qwen_pipeline = Qwen2VlPipeline(args.mm.model.infer_config)

    start_time = time.time()
    print_rank_0(f"Features extraction begins. {len(train_dataloader)} data in total.")
    counter = 0

    if hasattr(args.mm.tool, "profile"):
        prof = Profiler(args.mm.tool.profile)
        prof.start()

    for batch in train_dataloader:
        if batch:      
            video = batch.pop(VIDEO).to(device, dtype)
            prompt_ids = batch.pop(PROMPT_IDS)
            prompt_mask = batch.pop(PROMPT_MASK)
            file_names = batch.pop(FILE_INFO)
            ref_rgb_tensor_mllm = batch.pop("ref_rgb_tensor_mllm")
            text = batch.pop("raw_text")
        else:
            raise ValueError("Batch is None!")
        
        if torch.sum(video) == 0:
            continue
        
        bs = video.shape[0]
        counter += bs     
        
        if extract_video_feature:
            ref_pil_images = []
            ref_rgb_tensor_mllm = ref_rgb_tensor_mllm.squeeze(0) 
            ref_rgb_tensor_mllm = F.interpolate(ref_rgb_tensor_mllm, scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True)
            for t in range(ref_rgb_tensor_mllm.shape[0]):
                frame_tensor = ref_rgb_tensor_mllm[t]
                frame_np = frame_tensor.permute(1, 2, 0).cpu().numpy()  
                frame_np = (frame_np * 255).astype(np.uint8)  
                ref_pil_images.append(Image.fromarray(frame_np))
            hidden_states = qwen_pipeline(text,ref_pil_images[:4]) 
            latents, latents_dict = vae.encode(video, **batch)
            
        else:
            latents = video
        
        if extract_text_feature:
            prompt, prompt_mask = text_encoder.encode(prompt_ids, prompt_mask)
            hidden_states = hidden_states.permute(1, 0, 2)
            hidden_states = hidden_states.unsqueeze(1) # [1 1 1471 3584]
            hidden_states_mask = torch.ones(hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[2], device=hidden_states.device, dtype=torch.int64)
        else:
            prompt = prompt_ids
        
        if data_storage_mode == 'standard':
            for i in range(bs):
                pt_name = get_pt_name(file_names[i])
                latent = latents[i].cpu()
                torch.save(latent, os.path.join(save_path, 'videos', pt_name))
                if isinstance(prompt, list) or isinstance(prompt, tuple):
                    prompt = [_prompt[i].cpu() for _prompt in prompt]
                else:
                    prompt = prompt[i].cpu()
                torch.save(prompt, os.path.join(save_path, "labels", pt_name))
                data_to_save = {
                    "file": os.path.join('videos', pt_name),
                    "captions": os.path.join('labels', pt_name)
                }
                if latents_dict is not None:
                    for k in latents_dict:
                        latents_dict[k] = latents_dict[k][i]
                    torch.save(latents_dict, os.path.join(save_path, 'images', pt_name))
            print_rank_0(f"consumed sample {counter} | elapsed time {(time.time() - start_time):.2f} | file {pt_name}")
        
        elif data_storage_mode == 'sorafeatured':
            for i in range(bs):
                latent_i = latents[i].cpu()
                if isinstance(prompt_ids, list) or isinstance(prompt_ids, tuple):
                    prompts_i = [_prompt[i].cpu() for _prompt in prompt]
                    hidden_states_i = [_hidden_states[i].cpu() for _hidden_states in hidden_states]
                    prompt_masks_i = [_prompt_mask[i].cpu() for _prompt_mask in prompt_mask]
                    hidden_states_masks_i = [_hidden_states_mask[i].cpu() for _hidden_states_mask in hidden_states_mask]
                else:
                    prompts_i = prompt[i].cpu()
                    hidden_states_i = hidden_states[i].cpu()
                    prompt_masks_i = prompt_mask[i]
                    hidden_states_masks_i = hidden_states_mask[i]
                
                data_to_save = {
                    "video": latent_i,
                    "prompt_ids": prompts_i,
                    "prompt_mask": prompt_masks_i,
                    "hidden_states": hidden_states_i,
                    "hidden_states_mask": hidden_states_masks_i
                }

                if latents_dict:
                    for key in latents_dict.keys():
                        data_to_save[key] = latents_dict[key][i].cpu()

                pt_name = get_pt_name(file_names[i])
                torch.save(data_to_save, os.path.join(save_path, "features", pt_name))
            print_rank_0(f"consumed sample {counter} | elapsed time {(time.time() - start_time):.2f} | file {pt_name}")
        
        del video, latents, prompt_ids, prompt_mask, file_names
        if 'latents_dict' in locals():
            del latents_dict
        if 'prompt' in locals():
            del prompt
        if is_npu_available():
            torch.npu.empty_cache()
        if 'data_to_save' in locals():
            del data_to_save
        
        if hasattr(args.mm.tool, "profile"):
            prof.step()
    
    if hasattr(args.mm.tool, "profile"):
        prof.stop()

    duration = time.time() - start_time
    print_rank_0(f"{counter} feature vectors extracted in {duration:.2f} seconds.")


if __name__ == "__main__":
    extract_feature()
