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

import torch
from transformers import CLIPVisionModel
from megatron.training import get_args
from mindspeed_mm.data.data_utils.transform_pipeline import get_transforms
import random

class WanVideoI2VProcessor(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_encoder = CLIPVisionModel.from_pretrained(config["image_encoder"]).eval()

        args = get_args()
        first_frame_clip_preprocess = {
            "video": args.mm.data.dataset_param.preprocess_parameters.train_pipeline.first_frame_clip
        }

        first_frame_vae_preprocess = {
            "video": args.mm.data.dataset_param.preprocess_parameters.train_pipeline.first_frame_vae
        }

        global_shape_info = {
            "max_height": args.mm.data.dataset_param.preprocess_parameters.max_height,
            "max_width": args.mm.data.dataset_param.preprocess_parameters.max_width,
            "max_hxw": args.mm.data.dataset_param.preprocess_parameters.max_hxw,
        }

        self.first_frame_clip_transform = get_transforms(
            is_video=True, train_pipeline=first_frame_clip_preprocess, transform_size=global_shape_info
        )

        self.first_frame_vae_transform = get_transforms(
            is_video=True, train_pipeline=first_frame_vae_preprocess, transform_size=global_shape_info
        )


        self.enable_i2v_vae_encode_tiling = config.get("i2v_vae_encode_tiling", "auto")

    def get_image_clip_feature(self, image_tensor):
        # image_tensor: tensor, dtype float, shape (3,H,W), value: pixel / 255.0
        image_encoder_input = self.first_frame_clip_transform(image_tensor).to(
            dtype=self.image_encoder.dtype, device=self.image_encoder.device
        )
        clip_features = self.image_encoder(image_encoder_input, output_hidden_states=True).hidden_states[-2]
        return clip_features

    def get_image_vae_feature(self, videos, image_tensor, vae_model):
        # image_tensor: tensor, dtype float, shape (3,H,W), value: pixel / 255.0
        vae_input = self.first_frame_vae_transform(image_tensor).unsqueeze(2) # (1, 3, 1, h, w), (b, c, t, h, w)
        vae_input = vae_input.to(dtype=videos.dtype, device=videos.device)
        vae_features = vae_model.encode(vae_input) # (1, 16, 1, h, w)
        return vae_features

    def get_ref_features(self, videos, ref_rgb_tensor, ref_mask_tensor, vae_model):
        bs, T, C, H, W = ref_rgb_tensor.shape # bs = 1, C = 3, T = n, H = 1280, W = 720

        ref_clip_feature_list = []
        ref_vae_feature_list = []
        for i in range(T):
            ref_clip_feature = self.get_image_clip_feature(ref_rgb_tensor[:,i]) # (1, 257, 1280)
            ref_vae_feature = self.get_image_vae_feature(videos, ref_rgb_tensor[:,i], vae_model) # (1, 16, 1, h, w)
            mask = ref_mask_tensor[:,i:i+1] # (1, 1, H, W)

            h,w = ref_vae_feature.shape[-2:]

            mask = torch.nn.functional.interpolate(mask, (h, w), mode='nearest') # (1, 1, h ,w)

            mask = mask.view(bs, 1, 1, h, w) # (1, 1, 1, h ,w)
            temp = torch.zeros(bs, 3, 1, h, w, device=mask.device, dtype=mask.dtype)
            mask = torch.concat([mask, temp], dim=1) # (1, 4, 1, h, w)
            ref_vae_feature_with_mask = torch.concat([mask.to(dtype=ref_vae_feature.dtype, device=ref_vae_feature.device), ref_vae_feature], dim=1) # (1, 4+16, 1, h, w)

            ref_clip_feature_list.append(ref_clip_feature)
            ref_vae_feature_list.append(ref_vae_feature_with_mask)
        
        return {
            'ref_clip_feature_list': ref_clip_feature_list,
            'ref_vae_feature_list': ref_vae_feature_list
        }


    def merge_ref_vae_feature(self, i2v_vae_feature, ref_vae_feature_list, max_ref=4):
        bs, c, t, h, w = i2v_vae_feature.shape # b, 20, t, h, w
        max_ref = 4  

        if len(ref_vae_feature_list) > max_ref:
            ref_vae_feature_list = random.sample(ref_vae_feature_list, max_ref)
        else:
            pass  
        n = len(ref_vae_feature_list)
        pad = torch.zeros(bs, c, max_ref-n, h, w, device=i2v_vae_feature.device, dtype=i2v_vae_feature.dtype) # (1, 4+16, t+max_ref, h, w)
        ref = torch.concat(ref_vae_feature_list, dim=2) # (1, 4+16, n, h, w)
        ref = torch.concat([ref, pad], dim=2) # (1, 4+16, max_ref, h, w)
        feature = torch.concat([ref, i2v_vae_feature], dim=2) # [b 4+16 4 h w] + [b, 20, t, h, w]-> [b 20 4+t h w]
        return feature
    
    def merge_ref_clip_feature(self, i2v_clip_feature, ref_clip_feature_list, max_ref=4):
        bs, t, c = i2v_clip_feature.shape # 1, 257, 1280
        if len(ref_clip_feature_list) > max_ref:
            ref_clip_feature_list = random.sample(ref_clip_feature_list, max_ref)
        else:
            pass  
        n = len(ref_clip_feature_list)
        pad = torch.zeros(bs, 0, c, device=i2v_clip_feature.device, dtype=i2v_clip_feature.dtype)
        
        if n < max_ref:
            pad = torch.zeros(bs,(max_ref-n)*t,c, device=i2v_clip_feature.device, dtype=i2v_clip_feature.dtype)
        if n > 0:
            ref = torch.concat(ref_clip_feature_list, dim=1)
            pad = torch.concat([ref, pad], dim=1)
        feature = pad
        
        return feature

    def __call__(self, vae_model, videos, first_frame, ref_rgb_tensor=None, ref_mask_tensor=None, **kwargs):
        image_encoder_input = self.first_frame_clip_transform(first_frame).to(
            dtype=self.image_encoder.dtype, device=self.image_encoder.device
        )
        clip_features = self.image_encoder(image_encoder_input, output_hidden_states=True).hidden_states[-2]

        bs, _, t, h, w = videos.shape
        mask = torch.zeros(bs, t, h // 8, w // 8, device=videos.device)
        # mask[:, 1:] = 0
        mask = torch.concat([torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1), mask[:, 1:]], dim=1)
        mask = mask.view(bs, mask.shape[1] // 4, 4, h // 8, w // 8).transpose(1, 2) # (b, 4, t, h, w)

        vae_input = torch.zeros(bs, 3, t, h, w).to(dtype=videos.dtype, device=videos.device)
        vae_model_tiling_state = vae_model.get_tiling_state()
        if self.enable_i2v_vae_encode_tiling != "auto" and self.enable_i2v_vae_encode_tiling != vae_model_tiling_state:
            self.set_vae_tiling_state(vae_model, self.enable_i2v_vae_encode_tiling)

        vae_features = vae_model.encode(vae_input) # [b 16 t h w]
        # import pdb; pdb.set_trace()
        vae_features = torch.concat([mask.to(vae_features.dtype), vae_features], dim=1) # (b, 4+16, t, h, w)

        if ref_rgb_tensor is not None:
            ref_features = self.get_ref_features(videos=videos, ref_rgb_tensor=ref_rgb_tensor, ref_mask_tensor=ref_mask_tensor, vae_model=vae_model)
        else:
            ref_features = {'ref_clip_feature_list': [], 'ref_vae_feature_list': []}

        clip_features = self.merge_ref_clip_feature(i2v_clip_feature=clip_features, ref_clip_feature_list=ref_features['ref_clip_feature_list'])
        vae_features = self.merge_ref_vae_feature(i2v_vae_feature=vae_features, ref_vae_feature_list=ref_features['ref_vae_feature_list']) # [b 20 4 h w]

        # back vae tiling mode for video encode
        if vae_model.get_tiling_state() != vae_model_tiling_state:
            self.set_vae_tiling_state(vae_model, vae_model_tiling_state)
        
        return {
            "i2v_clip_feature": clip_features,
            "i2v_vae_feature": vae_features
        }
    
    def set_vae_tiling_state(self, vae_model, use_tiling):
        if use_tiling:
            vae_model.enable_tiling()
        else:
            vae_model.disable_tiling()
