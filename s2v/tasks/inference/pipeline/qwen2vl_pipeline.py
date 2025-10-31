# coding=utf-8
# Copyright (c) 2025, Bytedance Ltd. and/or its affiliates. All rights reserved.  
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.

from typing import Any, List, Dict, Optional, Tuple

import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers.generation.streamers import TextStreamer

# from mindspeed_mm.models.vlm_model import model_provider
from s2v.models.vlm_model import model_provider
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.generation_mixin import GenerationMixin
from mindspeed_mm.models.text_encoder import Tokenizer

from s2v.tasks.inference.pipeline.parallel_wrapper import ParallelWrapper


class Qwen2VlPipeline(GenerationMixin):

    def __init__(self, infer_config):
        self.infer_config = infer_config
        self.tokenizer = Tokenizer(infer_config.tokenizer).get_tokenizer()
        self.model = ParallelWrapper(model_provider)
        self.image_processor = AutoProcessor.from_pretrained(infer_config.tokenizer.from_pretrained,
                                                             local_files_only=True)
        self.generation_config = infer_config.generation_config
        self.model_config = infer_config.text_decoder
        self.main_input_name = 'input_ids'
        self.min_pixels = infer_config.min_pixels if hasattr(infer_config, "min_pixels") else None
        self.max_pixels = infer_config.max_pixels if hasattr(infer_config, "max_pixels") else None

    def __call__(self, prompt=None, images=None, videos=None, return_ids=False):
        if images:
            if isinstance(images, list):
                image = images
            else:
                image = images
        else:
            image = self.infer_config.image_path if hasattr(self.infer_config, "image_path") else None

        if videos:
            if isinstance(videos, list):
                video = videos[0]
            else:
                video = videos
        else:
            video = self.infer_config.video_path if hasattr(self.infer_config, "video_path") else None

        if not prompt:
            prompt = self.infer_config.prompts

        inputs = self.prepare_inputs(prompt=prompt, images=image, videos=video)

        if inputs is None:
            inputs = {'input_ids': self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.infer_config.device)}

        self.model.inference_params = None

        model_inputs = self.prepare_inputs_for_generation(**inputs)

        hidden_states = self.model(**model_inputs)

        return hidden_states


    def prepare_inputs(self, prompt=None, images=None, videos=None, messages=None):
        if not images and not messages and not videos:
            return None

        if not messages:
            content = [{"type": "text", "text": prompt}]
            if images:
                if not isinstance(images, list):
                    images = [images]
                for image in images:
                    content.append({"type": "image", "image": image})
            if videos:
                if not isinstance(videos, list):
                    videos = [videos]
                for video in videos:
                    content.append({"type": "video", "video": video})
    
            messages = [[
                {
                    "role": "user",
                    "content": content,
                }
            ]]

        prompt = self.image_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.image_processor(
            text=prompt,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.infer_config.device)
        if image_inputs:
            inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
        if video_inputs:
            inputs['pixel_values'] = inputs['pixel_values_videos'].unsqueeze(0)
            inputs['image_grid_thw'] = inputs['video_grid_thw']
        return inputs
    
    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embeddin for text part.
            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.infer_config.image_encoder.vision_encoder.spatial_merge_size
        image_token_id = self.generation_config.image_token_id
        video_token_id = self.generation_config.video_token_id
        vision_start_token_id = self.generation_config.vision_start_token_id
        mrope_position_deltas = []
        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def prepare_inputs_for_generation(self, **kwargs):
        input_ids = kwargs.get("input_ids", None)
        # init cache_position
        if "cache_position" not in kwargs:
            if "inputs_embeds" in kwargs:
                cache_position = torch.ones_like(kwargs["inputs_embeds"][0, :, 0], dtype=torch.int64).cumsum(0) - 1
            else:
                cache_position = torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1
            kwargs["cache_position"] = cache_position
        cache_position = kwargs.get("cache_position", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        image_grid_thw = kwargs.get("image_grid_thw", None)
        rope_deltas = kwargs.get("rope_deltas", None)
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        pixel_values = kwargs.get("pixel_values", None)

        model_inputs = {}

        if self.model.inference_params is not None:
            batch_size, seq_length = input_ids.shape
            input_ids = input_ids[:, [-1]]
            pixel_values = None
            attention_mask = torch.ones((batch_size, seq_length)).to("npu")

        if attention_mask is not None and position_ids is None:
            if cache_position is None or (cache_position is not None and cache_position[0] == 0):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
            else:
                batch_size, seq_length = input_ids.shape
                delta = (
                    cache_position[0] + rope_deltas if cache_position is not None and rope_deltas is not None else 0
                )
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        
        model_inputs.update(
            {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "cache_position": cache_position,
                "rope_deltas": rope_deltas
            }
        )

        return model_inputs
    
    def _update_model_kwargs_for_generation(self, model_kwargs:Dict[str, Any], model_inputs:Dict[str, Any]):
        # updata cache_position
        if model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = model_inputs["cache_position"][-1:] + 1
        else:
            past_positions = model_inputs.pop("cache_position")
            new_positions = torch.arange(
                past_positions[-1] + 1, past_positions[-1] + 2, dtype=past_positions.dtype
            ).to(past_positions.device)
            model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
        
        # update rope_deltas
        if "rope_deltas" in model_inputs:
            model_kwargs["rope_deltas"] = model_inputs["rope_deltas"]

        return model_kwargs

    def evaluate(self, message):
        messages = [{'role': 'user', 'content': self._prepare_content(message)}]
        inputs = self.prepare_inputs(messages=[messages])

        generated_ids = self.generate(**inputs,
                                      do_sample=True if self.generation_config.temperature > 0 else False,
                                      temperature=self.generation_config.temperature,
                                      max_new_tokens=self.generation_config.max_new_tokens)
        # clear cache memory
        self.model.inference_params = None
        if generated_ids is not None:
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            out = self.image_processor.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = out[0]
            return response
        else:
            return None

    def _prepare_content(self, inputs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': 'file://' + s['value']}
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content
