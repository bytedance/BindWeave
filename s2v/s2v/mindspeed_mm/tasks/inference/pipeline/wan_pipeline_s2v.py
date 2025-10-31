# Copyright (c) 2025, Bytedance Ltd. and/or its affiliates. 
# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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

from typing import List, Optional, Union
import html
import math
import os
import io

import numpy as np
import PIL.Image
import ftfy
import regex as re
import torch
from torchvision.transforms import v2
from torchvision.transforms.functional import center_crop
from transformers import CLIPVisionModel
from megatron.training import get_args
from megatron.core import mpu
from mindspeed_mm.utils.utils import get_device

from mindspeed_mm.tasks.inference.pipeline.wan_pipeline import WanPipeline
from mindspeed_mm.data.data_utils.data_transform import shortsideresize, resize


NEGATIVE_PROMOPT = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"


class ConditionImageTransform:
    def __init__(self, num_frames=89):
        self._vae_transform = v2.Compose(
            [
                v2.Resize(size=[1280, 720]),
                v2.Normalize(mean=[0.5], std=[0.5])
            ]
        )
        self._clip_transform = v2.Compose(
            [
                v2.Resize(size=[224, 224]),
                v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ]
        )
        self._num_frames = num_frames

    def get_image_clip_feature(self, image_tensor, image_encoder):
        image_encoder_input = self._clip_transform(image_tensor).to(
            dtype=image_encoder.dtype, device=image_encoder.device
        )
        clip_features = image_encoder(image_encoder_input, output_hidden_states=True).hidden_states[-2]
        return clip_features

    def get_i2v_vae_feature(self, image_tensor, vae_model, dtype, device):
        # image vae
        vae_input = self._vae_transform(image_tensor) # (1, 3, 1280, 720)
        _, _, H, W = vae_input.shape
        vae_input = torch.concat([vae_input.unsqueeze(2), torch.zeros(1, 3, self._num_frames - 1, H, W)], dim=2).to(device=device, dtype=dtype)
        vae_feature = vae_model.encode(vae_input) # (1, 16, t, 160, 90)

        # mask
        b, c, t, h, w = vae_feature.shape
        msk = torch.ones(b, 4, t, h, w).to(dtype=dtype, device=device)
        msk[:,:,1:] = 0
        vae_feature = torch.concat([msk, vae_feature], dim=1)

        return vae_feature
    
    def __call__(self, image, image_encoder, vae_model, dtype, device):
        rgb_np = np.array(image) # HWC, uint8
        rgb_tensor = torch.tensor(rgb_np[None,...]) # NHWC
        rgb_tensor = torch.permute(rgb_tensor, (0, 3, 1, 2)) # NCHW, uint8
        rgb_tensor = rgb_tensor.float() / 255.0 # (1, 3, H, W)

        # clip feature
        clip_feature = self.get_image_clip_feature(rgb_tensor, image_encoder)

        # vae feature
        vae_feature = self.get_i2v_vae_feature(rgb_tensor, vae_model=vae_model, dtype=dtype, device=device)

        return {
            'i2v_clip_feature': clip_feature,
            'i2v_vae_feature': vae_feature,
        }


class RefTransform:
    def __init__(self):
        self._vae_transform = v2.Normalize(mean=[0.5], std=[0.5])
        self._clip_transform = v2.Compose(
            [
                v2.Resize(size=[224, 224]),
                v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ]
        )
    
    def get_image_clip_feature(self, image_tensor, image_encoder):
        # image_tensor: tensor, dtype float, shape (1,3,H,W), value: pixel / 255.0
        image_encoder_input = self._clip_transform(image_tensor).to(
            dtype=image_encoder.dtype, device=image_encoder.device
        )
        clip_features = image_encoder(image_encoder_input, output_hidden_states=True).hidden_states[-2]
        return clip_features

    def get_image_vae_feature(self, image_tensor, vae_model, dtype, device):
        # image_tensor: tensor, dtype float, shape (1,3,H,W), value: pixel / 255.0
        vae_input = self._vae_transform(image_tensor).unsqueeze(2) # (1, 3, 1, h, w), (b, c, t, h, w)
        vae_input = vae_input.to(dtype=dtype, device=device)
        vae_features = vae_model.encode(vae_input) # (1, 16, 1, h, w)
        return vae_features

    def get_ref_features(self, ref_rgb_tensor, ref_mask_tensor, image_encoder, vae_model, dtype, device):
        bs, T, C, H, W = ref_rgb_tensor.shape # bs = 1, T = n, C = 3, H = 1280, W = 720

        ref_clip_feature_list = []
        ref_vae_feature_list = []
        for i in range(T):
            ref_clip_feature = self.get_image_clip_feature(ref_rgb_tensor[:,i], image_encoder=image_encoder) # (1, 257, 1280)
            ref_vae_feature = self.get_image_vae_feature(ref_rgb_tensor[:,i], vae_model=vae_model, dtype=dtype, device=device) # (1, 16, 1, h, w), (b, c, t, h, w)
            mask = ref_mask_tensor[:,i:i+1] # (1, 1, H, W)
            h,w = ref_vae_feature.shape[-2:]
            # transforms.Resize(size=(h,w), interpolation=Image.NEAREST)
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

    def __call__(self, image_list, image_encoder, vae_model, dtype, device):
        ref_rgba_np_list = [np.array(image) for image in image_list]
        ref_rgba_np = np.array(ref_rgba_np_list) # THWC(rgba), uint8
        ref_rgb_tensor = torch.tensor(ref_rgba_np[:,:,:,:3])
        ref_rgb_tensor = torch.permute(ref_rgb_tensor, (0, 3, 1, 2)) # TCHW, uint8
        ref_rgb_tensor = ref_rgb_tensor.float() / 255.0

        ref_mask_np = (ref_rgba_np[:,:,:,3] > 127).astype('int') # n h w
        ref_mask_tensor = torch.tensor(ref_mask_np).float()

        ref_rgb_tensor = ref_rgb_tensor.unsqueeze(0) # (1, T, C, H, W)
        ref_mask_tensor = ref_mask_tensor.unsqueeze(0) # (1, T, H, W)

        ref_features = self.get_ref_features(ref_rgb_tensor=ref_rgb_tensor, 
                                             ref_mask_tensor=ref_mask_tensor, 
                                             image_encoder=image_encoder, 
                                             vae_model=vae_model, 
                                             dtype=dtype, 
                                             device=device)
        return ref_features




class WanS2VPipeline(WanPipeline):
    def __init__(self, vae, tokenizer, text_encoder, scheduler, predict_model, image_encoder=None, config=None):
        super().__init__(vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=scheduler, predict_model=predict_model, image_encoder=image_encoder, config=config)
        self._condition_transform = ConditionImageTransform(num_frames=self.num_frames)
        self._ref_transform = RefTransform()

    @torch.no_grad
    def predict_extracted_samples(self, sample_list):
        model_kwargs = self.process_extracted_features(sample_list)
        videos = self.model_predict(model_kwargs)
        return videos
    
    @torch.no_grad
    def process_extracted_features(self, sample_list, 
                max_sequence_length=512,
                negative_prompt=NEGATIVE_PROMOPT,
                device=get_device("npu"), 
                dtype=torch.bfloat16):
        batch_size = len(sample_list)
        assert batch_size == 1

        # text
        prompt_embeds = torch.concat([v['prompt_ids'] for v in sample_list], dim=0)
        prompt_mask = torch.concat([v['prompt_mask'] for v in sample_list], dim=0)

        seq_lens = prompt_mask.view(batch_size, -1).sum(dim=-1)
        seq_lens = seq_lens.to(torch.int64)
        for i, seq_len in enumerate(seq_lens):
            prompt_embeds[i, seq_len:] = 0

        # negative prompt embeddings
        negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
        negative_prompt_embeds = self._get_prompt_embeds(
            prompt=negative_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

        # visual conditions
        i2v_clip_feature = torch.stack([v['i2v_clip_feature'] for v in sample_list]).to(device=device, dtype=dtype) # [b, 517,1280]
        i2v_vae_feature = torch.stack([v['i2v_vae_feature'] for v in sample_list]).to(device=device, dtype=dtype) # [b, 20,27,160,90]

        # noise
        latent_t = sample_list[0]['video'].shape[1] # [c,t,h,w]
        latent_h = i2v_vae_feature.shape[3]
        latent_w = i2v_vae_feature.shape[4]
        
        shape = (
            batch_size,
            self.vae.model.config.z_dim,
            latent_t,
            latent_h,
            latent_w,
        )
        latents = self.prepare_latents(shape, generator=self.generator, device=device, dtype=dtype) # noise

        # make request
        features = {
            "x": latents,
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "i2v_clip_feature": i2v_clip_feature,
            "i2v_vae_feature": i2v_vae_feature,
        }

        return features
    
    def merge_ref_vae_feature(self, i2v_vae_feature, ref_vae_feature_list, max_ref=4):
        bs, c, t, h, w = i2v_vae_feature.shape # 1, 20, t, h, w

        ref_vae_feature_list = ref_vae_feature_list[:max_ref]
        
        n = len(ref_vae_feature_list)
        if n < max_ref:
            pad = torch.zeros(bs,c,max_ref-n,h,w, device=i2v_vae_feature.device, dtype=i2v_vae_feature.dtype)
        if n > 0:
            ref = torch.concat(ref_vae_feature_list, dim=2)
            pad = torch.concat([ref, pad], dim=2)
        
        feature = torch.concat([pad, i2v_vae_feature], dim=2)
        return feature
    
    def merge_ref_clip_feature(self, i2v_clip_feature, ref_clip_feature_list, max_ref=4):
        # usually mini_batch_size of a data_parallel_rank is 1, no need to align sequence length
        # bs, t, c = i2v_clip_feature.shape # 1, 257, 1280
        ref_clip_feature_list = ref_clip_feature_list[:max_ref]

        n = len(ref_clip_feature_list)

        if n > 0:
            pad = torch.concat(ref_clip_feature_list, dim=1)
            feature = torch.concat([pad, i2v_clip_feature], dim=1)
        else:
            feature = i2v_clip_feature

        return feature

    @torch.no_grad()
    def predict_raw_samples(self, sample_list, 
                max_sequence_length=512, 
                device=get_device("npu"),
                **kwargs):
        batch_size = len(sample_list)
        
        assert batch_size == 1

        negative_prompt = NEGATIVE_PROMOPT
        prompt = [v['prompt'] for v in sample_list]

        # Encode input prompt
        do_classifier_free_guidance = self.scheduler.do_classifier_free_guidance
        prompt_embeds, negative_prompt_embeds = self.encode_texts(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # Prepare latents and model_kwargs
        image = PIL.Image.open(io.BytesIO(sample_list[0]['image']))
        conditions = self._condition_transform(image, image_encoder=self.image_encoder, vae_model=self.vae, dtype=prompt_embeds.dtype, device=prompt_embeds.device)
        clip_features = conditions['i2v_clip_feature']
        vae_features = conditions['i2v_vae_feature']
        b,c,t,h,w = vae_features.shape
        latents = self.prepare_latents((b,c-4,t,h,w), generator=self.generator, device=vae_features.device, dtype=vae_features.dtype) # noise

        ref_image_list = [PIL.Image.open(io.BytesIO(ref_image)) for ref_image in sample_list[0]['ref_image_list']]
        ref_features = self._ref_transform(ref_image_list, image_encoder=self.image_encoder, vae_model=self.vae, dtype=vae_features.dtype, device=vae_features.device)

        clip_features = self.merge_ref_clip_feature(i2v_clip_feature=clip_features, ref_clip_feature_list=ref_features['ref_clip_feature_list'])
        vae_features = self.merge_ref_vae_feature(i2v_vae_feature=vae_features, ref_vae_feature_list=ref_features['ref_vae_feature_list'])

        model_kwargs = {
            "x": latents,
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "i2v_clip_feature": clip_features,
            "i2v_vae_feature": vae_features,
        }


        videos = self.model_predict(model_kwargs)
        return videos

    @torch.no_grad
    def model_predict(self, model_kwargs):
        do_classifier_free_guidance = self.scheduler.do_classifier_free_guidance
        num_inference_steps = self.scheduler.num_inference_steps
        timesteps = self.scheduler.timesteps

        latents = model_kwargs.pop('x')

        # 5. Denoising to get clean latents
        num_warmup_steps = self.scheduler.num_warmup_steps
        guidance_scale = self.scheduler.guidance_scale
        self.scheduler.diffusion.set_timesteps(num_inference_steps)  # reset timesteps
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = latents.to(self.predict_model.dtype)
                timestep = t.expand(latents.shape[0]).to(device=latents.device).float()

                noise_pred = self.predict_model(
                    latent_model_input, timestep, model_kwargs.get("prompt_embeds"), **model_kwargs
                )[0]

                if do_classifier_free_guidance:
                    noise_uncond = self.predict_model(
                        latent_model_input, timestep, model_kwargs.get("negative_prompt_embeds"), **model_kwargs
                    )[0]
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps):
                    progress_bar.update()

        # 6. Post process latents to get video
        latents = latents.to(self.vae.model.dtype)
        latents_mean = (
            torch.tensor(self.vae.model.config.latents_mean)
            .view(1, self.vae.model.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.model.config.latents_std).view(
            1, self.vae.model.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        video = self.decode_latents(latents)
        return video
