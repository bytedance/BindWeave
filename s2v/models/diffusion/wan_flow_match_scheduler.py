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

from typing import Optional

import torch
from tqdm.auto import tqdm



class WanFlowMatchScheduler():

    def __init__(
        self,
        num_inference_timesteps=None,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        guidance_scale=0.0,
        **kwargs
        ):
        self.num_inference_timesteps = num_inference_timesteps
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.guidance_scale = guidance_scale
        self.do_classifier_free_guidance = guidance_scale > 1.0
        if self.num_inference_timesteps is not None:
            self._set_timesteps(self.num_inference_timesteps, training=False)
        else:
            self._set_timesteps(self.num_train_timesteps, training=True)

    def training_losses(
        self,
        model_output: torch.Tensor,
        x_start: Optional[torch.Tensor] = None,
        x_t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        **kwargs
    ):
        reduction = kwargs.get("reduction", "mean")
        target = noise - x_start

        output = model_output[:,:,4:].float() # b,c,t,h,w
        # print('output', output.shape)

        # loss = torch.nn.functional.mse_loss(model_output.float(), target.float(), reduction=reduction)
        loss = torch.nn.functional.mse_loss(output, target.float(), reduction=reduction)
        loss *= self._training_weight(t)
        return loss

    def sample(
        self,
        model,
        latents,
        model_kwargs
    ):
        num_inference_steps = self.num_inference_timesteps or self.num_train_timesteps
        num_warmup_steps = len(self.timesteps) - num_inference_steps

        # for loop denoising to get clean latents
        with tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(self.timesteps):
                latent_model_input = latents
                timestep = t.expand(latent_model_input.shape[0]).to(latents.device)

                noise_pred = model(
                    latent_model_input,
                    timestep,
                    model_kwargs.get('prompt_embeds'),
                    **model_kwargs
                )

                if self.do_classifier_free_guidance:
                    noise_uncond = model(
                        latent_model_input,
                        timestep,
                        model_kwargs.get('negative_prompt_embeds'),
                        **model_kwargs
                    )
                    noise_pred = noise_uncond + self.guidance_scale * (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self._step(noise_pred, t, latents)
                
                if i == len(self.timesteps) - 1 or ((i + 1) > num_warmup_steps):
                    progress_bar.update()
        
        return latents
    
    def q_sample(self, latents, noise=None, t=None, **kwargs):
        if noise is None:
            noise = torch.randn_like(latents)
        
        if t is None:
            timestep_idx = torch.randint(0, self.num_train_timesteps, (1,))
            timestep = self.timesteps[timestep_idx].to(latents.device)
        else:
            timestep = t
            timestep_idx = (t.to("cpu") == self.timesteps).nonzero()[0]
        
        sigma = self.sigmas[timestep_idx].to(latents.device)
        noised_latents = (1 - sigma) * latents + sigma * noise
        return noised_latents, noise, timestep

    def _set_timesteps(self, num_steps=100, training=False):
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min)
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            y = torch.exp(-2 * ((self.timesteps - num_steps / 2) / num_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing


    def _step(self, model_output, timestep, sample):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_idx = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_idx]
        if timestep_idx + 1 >= len(self.timesteps):
            sigma_next = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_next = self.sigmas[timestep_idx + 1]
        prev_sample = sample + model_output * (sigma_next - sigma)
        return prev_sample

    def _training_weight(self, timestep):
        if len(timestep) > 1:
            timestep = timestep[0]
        timestep_idx = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
        weights = self.linear_timesteps_weights[timestep_idx]
        return weights