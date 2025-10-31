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
from mindspeed_mm.models.predictor.dits.wan_dit import WanDiT


class WanS2VDiT(WanDiT):
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        prompt: torch.Tensor,
        prompt_mask: torch.Tensor = None,
        i2v_clip_feature: torch.Tensor = None,
        i2v_vae_feature: torch.Tensor = None,
        **kwargs,
    ):
        pad_num = 0
        if i2v_vae_feature is not None:
            b,c,t1,h,w = x.shape
            _,_,t2,_,_ = i2v_vae_feature.shape
            pad_num = t2 - t1
        
        if pad_num > 0:
            pad = torch.zeros((b,c,pad_num,h,w), dtype=x.dtype, device=x.device)
            x = torch.concat([pad, x], dim=2)

        out, prompt, prompt_emb, time_emb, times, prompt_mask = super().forward(
                                            x=x,
                                            timestep=timestep,
                                            prompt=prompt,
                                            prompt_mask=prompt_mask,
                                            i2v_clip_feature=i2v_clip_feature,
                                            i2v_vae_feature=i2v_vae_feature)

        if pad_num > 0:
            out = out[:, :, pad_num:]

        rtn = (out, prompt, prompt_emb, time_emb, times, prompt_mask)
        return rtn