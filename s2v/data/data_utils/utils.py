# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Modified from huggingface diffusers repos
# This source code is licensed under the notice found in the root directory of this source tree.
# --------------------------------------------------------
# References:
# TextProcesser: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deepfloyd_if/pipeline_if.py
# DataSet https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/internvl/train/dataset.py

import json
import os
import re
import gc
import html
import math
import copy
import random
import urllib.parse as ul
from fractions import Fraction
from collections import Counter
from logging import getLogger
from typing import Any, Dict, Optional, Tuple, Union, Sequence

try:
    import decord
except ImportError:
    print("Failed to import decord module.")

import av
import ftfy
import torch
import torchvision
import numpy as np
import pandas as pd
import torchvision.transforms as TT
from PIL import Image
from bs4 import BeautifulSoup
from einops import rearrange
import torch.nn.functional as F
from torchvision import get_video_backend
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
from torchvision.io.video import (
    _align_audio_frames,
    _check_av_available,
    _log_api_usage_once,
    _read_from_stream,
    _video_opt,
)
import transformers
from transformers.models.clip.image_processing_clip import CLIPImageProcessor
from transformers.trainer_pt_utils import LabelSmoother
from packaging import version
import tokenizers
from megatron.training import get_args
from megatron.core import mpu

from mindspeed_mm.data.data_utils.data_transform import (
    TemporalRandomCrop, 
    Expand2Square,
    get_params,
    calculate_statistics,
    maxhwresize
)
from mindspeed_mm.data.data_utils.transform_pipeline import get_transforms
from mindspeed_mm.data.data_utils.conversation import get_conv_template
from mindspeed_mm.data.data_utils.constants import MODEL_CONSTANTS


logger = getLogger(__name__)
VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
TENSOR_EXTENSIONS = (".pt", ".pth")
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class DataFileReader:
    """get the data from different types of files such as csv/json/parquat"""

    def __init__(self, data_storage_mode="standard"):
        self.data_storage_mode = data_storage_mode

    def __call__(self, data_path, return_type="list"):
        if self.data_storage_mode == "standard":
            return self.get_datasamples(data_path, return_type=return_type)
        elif self.data_storage_mode == "combine" or self.data_storage_mode == "sorafeatured":
            return self.get_cap_list(data_path)
        else:
            raise NotImplementedError("Not support now.")

    @staticmethod
    def get_datasamples(data_path, return_type="list"):
        if data_path.endswith(".csv"):
            data_out = pd.read_csv(data_path)
            if return_type == "list":
                return data_out.to_dict("records")
            else:
                return data_out

        elif data_path.endswith(".json"):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):

                records = [{key: value} for key, value in data.items()]
            elif isinstance(data, list):
                records = data
            else:
                records = [data]

            return records

        elif data_path.endswith(".jsonl"):
            data_out = pd.read_json(data_path, lines=True)
            return data_out.to_dict("records")
        elif data_path.endswith(".parquat"):
            data_out = pd.read_parquat(data_path)
            return data_out.to_dict("records")
        elif data_path.endswith(".txt"):
            with open(data_path, 'r') as f:
                data_out = f.readlines()
            data_out = [data.strip() for data in data_out]
            return data_out
        else:
            raise NotImplementedError(f"Unsupported file format: {data_path}")

    def get_cap_list(self, data_path):
        cap_lists = []
        with open(data_path, "r") as f:
            folder_anno = [
                i.strip().split(",")
                for i in f.readlines()
                if len(i.strip()) > 0
            ]
        for folder, anno in folder_anno:

            sub_list = self.get_datasamples(anno)
            # print("sub_list",sub_list)
            print(f"Building {anno}...")
            for sub in sub_list:
                outer_key = list(sub.keys())[0]
                original_path = sub[outer_key]["metadata"]["path"]
                main_part = folder.rsplit("/", 1)[1] 
                sub[outer_key]["metadata"]["main_part"] = main_part
                sub[outer_key]["metadata"]["path"] = os.path.join(folder, original_path)
            cap_lists += sub_list
        return cap_lists


class DecordInit:
    """Using Decord (https://github.com/dmlc/decord) to initialize the video_reader."""

    def __init__(self, num_threads=1):
        self.num_threads = num_threads
        self.ctx = decord.cpu(0)

    def __call__(self, filename):
        """Perform the Decord initialization.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        reader = decord.VideoReader(
            filename, ctx=self.ctx, num_threads=self.num_threads
        )
        return reader

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            f"sr={self.sr},"
            f"num_threads={self.num_threads})"
        )
        return repr_str


class VideoReader:
    """support some methods to read video"""

    def __init__(self, video_reader_type=None, num_threads=1):
        self.video_reader_type = video_reader_type
        if self.video_reader_type == "decoder":
            self.v_decoder = DecordInit(num_threads)

    def __call__(self, video_path):
        is_decord_read = False
        info = None

        if self.video_reader_type == "decoder":
            vframes = self.v_decoder(video_path)
            is_decord_read = True
        elif self.video_reader_type == "torchvision":
            vframes, aframes, info = torchvision.io.read_video(
                filename=video_path, pts_unit="sec", output_format="TCHW"
            )  # [T: temporal, C: channel, H: height, W: width]
        elif self.video_reader_type == "av":
            vframes, aframes, info = read_video_av(filename=video_path, pts_unit="sec", output_format="TCHW")
        else:
            raise NotImplementedError(
                f"Unsupported video reader type: {self.video_reader_type}"
            )
        return vframes, info, is_decord_read


def read_video_av(
        filename: str,
        start_pts: Union[float, Fraction] = 0,
        end_pts: Optional[Union[float, Fraction]] = None,
        pts_unit: str = "pts",
        output_format: str = "THWC",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Reads a video from a file, returning both the video frames and the audio frames

    Args:
        filename (str): path to the video file
        start_pts (int if pts_unit = "pts", float / Fraction if pts_unit = "sec", optional):
            The start presentation time of the video
        end_pts (int if pts_unit = "pts", float / Fraction if pts_unit = "sec", optional):
            The end presentation time
        pts_unit (str, optional): unit in which start_pts and end_pts values will be interpreted,
            either "pts" or "sec". Defaults to "pts".
        output_format (str, optional): The format of the output video tensors. Can be either "THWC" (default) or "TCHW".

    Returns:
        vframes (Tensor[T, H, W, C] or Tensor[T, C, H, W]): the `T` video frames
        aframes (Tensor[K, L]): the audio frames, where `K` is the number of channels and `L` is the number of points
        info (Dict): metadata for the video and audio. Can contain the fields video_fps (float) and audio_fps (int)
    """
    output_format = output_format.upper()
    if output_format not in ("THWC", "TCHW"):
        raise ValueError(f"output_format should be either 'THWC' or 'TCHW', got {output_format}.")

    if not os.path.exists(filename):
        raise RuntimeError(f"File not found: {filename}")

    if get_video_backend() != "pyav":
        vframes, aframes, info = _video_opt._read_video(filename, start_pts, end_pts, pts_unit)
    else:
        _check_av_available()

        if end_pts is None:
            end_pts = float("inf")

        if end_pts < start_pts:
            raise ValueError(
                f"end_pts should be larger than start_pts, got start_pts={start_pts} and end_pts={end_pts}"
            )

        info = {}
        video_frames = []
        audio_frames = []
        audio_timebase = _video_opt.default_timebase

        container = av.open(filename, metadata_errors="ignore")
        try:
            if container.streams.audio:
                audio_timebase = container.streams.audio[0].time_base
            if container.streams.video:
                video_frames = _read_from_stream(
                    container,
                    start_pts,
                    end_pts,
                    pts_unit,
                    container.streams.video[0],
                    {"video": 0},
                )
                video_fps = container.streams.video[0].average_rate
                # guard against potentially corrupted files
                if video_fps is not None:
                    info["video_fps"] = float(video_fps)

            if container.streams.audio:
                audio_frames = _read_from_stream(
                    container,
                    start_pts,
                    end_pts,
                    pts_unit,
                    container.streams.audio[0],
                    {"audio": 0},
                )
                info["audio_fps"] = container.streams.audio[0].rate
        except av.AVError as ex:
            raise ex
        finally:
            container.close()
            del container
            # NOTE: manually garbage collect to close pyav threads
            gc.collect()

        vframes_list = [frame.to_rgb().to_ndarray() for frame in video_frames]
        aframes_list = [frame.to_ndarray() for frame in audio_frames]

        if vframes_list:
            vframes = torch.as_tensor(np.stack(vframes_list))
        else:
            vframes = torch.empty((0, 1, 1, 3), dtype=torch.uint8)

        if aframes_list:
            aframes = np.concatenate(aframes_list, 1)
            aframes = torch.as_tensor(aframes)
            if pts_unit == "sec":
                start_pts = int(math.floor(start_pts * (1 / audio_timebase)))
                if end_pts != float("inf"):
                    end_pts = int(math.ceil(end_pts * (1 / audio_timebase)))
            aframes = _align_audio_frames(aframes, audio_frames, start_pts, end_pts)
        else:
            aframes = torch.empty((1, 0), dtype=torch.float32)

    if output_format == "TCHW":
        # [T,H,W,C] --> [T,C,H,W]
        vframes = vframes.permute(0, 3, 1, 2)

    return vframes, aframes, info


class VideoProcesser:
    """Used for video data preprocessing"""

    def __init__(
            self,
            num_frames=16,
            frame_interval=1,
            train_pipeline=None,
            data_storage_mode="standard",
            data_process_type=None,
            skip_frame_num=0,
            fps=None,
            train_fps=24,
            auto_interval=True,
            speed_factor=1.0,
            drop_short_ratio=1.0,
            max_height=480,
            max_width=640,
            max_hxw=None,
            min_hxw=None,
            force_resolution=True,
            seed=42,
            hw_stride=32,
            hw_aspect_thr=1.5,
            ae_stride_t=4,
            sp_size=4,
            train_sp_batch_size=1,
            gradient_accumulation_size=1,
            batch_size=1,
            min_num_frames=29,
            transform_size=None,
            **kwargs,
    ):
        self.num_frames = num_frames
        self.train_pipeline = train_pipeline
        self.video_transforms = None
        self.temporal_sample = TemporalRandomCrop(num_frames * frame_interval)
        self.data_storage_mode = data_storage_mode
        self.data_process_type = data_process_type
        self.skip_frame_num = skip_frame_num
        self.fps = fps
        self.auto_interval = auto_interval
        self.frame_interval = frame_interval        

        self.max_height = max_height
        self.max_width = max_width
        self.transform_size = transform_size
        if self.data_storage_mode == "combine" or self.data_storage_mode == "sorafeatured":
            self.train_fps = train_fps
            self.speed_factor = speed_factor
            self.drop_short_ratio = drop_short_ratio
            self.max_hxw = max_hxw
            self.min_hxw = min_hxw
            self.force_resolution = force_resolution
            self.seed = seed
            self.generator = torch.Generator().manual_seed(self.seed) 
            self.hw_stride = hw_stride
            self.hw_aspect_thr = hw_aspect_thr
            self.ae_stride_t = ae_stride_t
            self.sp_size = sp_size
            self.train_sp_batch_size = train_sp_batch_size
            self.gradient_accumulation_size = gradient_accumulation_size
            self.batch_size = batch_size
            self.min_num_frames = min_num_frames


    def __call__(self, vframes, num_frames=None, frame_interval=None, image_size=None, is_decord_read=False,
                 predefine_num_frames=13, start_frame_idx=0, clip_total_frames=-1, resolution_crop=(None, None, None, None)
    ):
        if image_size:
            self.video_transforms = get_transforms(is_video=True, train_pipeline=self.train_pipeline,
                                                   image_size=image_size)
        else:
            self.video_transforms = get_transforms(is_video=True, train_pipeline=self.train_pipeline,
                                                   transform_size=self.transform_size)
        if self.data_storage_mode == "standard":
            if self.data_process_type == "CogvideoX":
                return self.cog_data_process(vframes)

            total_frames = len(vframes)
            if num_frames:
                self.num_frames = num_frames
                self.temporal_sample = TemporalRandomCrop(num_frames * frame_interval)
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            if end_frame_ind - start_frame_ind < self.num_frames:
                raise AssertionError("the video does not have enough frames.")
            frame_indice = np.linspace(
                start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int
            )
            if is_decord_read:
                video = vframes.get_batch(frame_indice).asnumpy()
                video = torch.from_numpy(video)
                # THWC -> TCHW,  [T: temporal, C: channel, H: height, W: width]
                video = video.permute(0, 3, 1, 2)
            else:
                video = vframes[frame_indice]  # TCHW

            video = self.video_transforms(video)
            # TCHW -> CTHW
            video = video.permute(1, 0, 2, 3)
        else:
            video = self.combine_data_video_process(
                vframes,
                is_decord_read=is_decord_read,
                predefine_num_frames=predefine_num_frames,
                start_frame_idx=start_frame_idx,
                clip_total_frames=clip_total_frames,
                resolution_crop=resolution_crop
            )
        return video


    def cog_data_process(self, video_frames):
        actual_fps = video_frames.get_avg_fps()
        ori_video_len = len(video_frames)

        if ori_video_len / actual_fps * self.fps > self.num_frames:
            num_frames = self.num_frames
            start = int(self.skip_frame_num)
            end = int(start + num_frames / self.fps * actual_fps)
            end_safety = min(int(start + num_frames / self.fps * actual_fps), int(ori_video_len))
            indices = np.arange(start, end, (end - start) // num_frames).astype(int)
            temp_frames = video_frames.get_batch(np.arange(start, end_safety)).asnumpy()
            if temp_frames is None:
                raise ValueError("temp_frames is unexpectedly None")
            tensor_frames = torch.from_numpy(temp_frames)
            tensor_frames = tensor_frames[torch.tensor((indices - start).tolist())]
        else:
            if ori_video_len > self.num_frames:
                num_frames = self.num_frames
                start = int(self.skip_frame_num)
                end = int(ori_video_len - self.skip_frame_num)
                indices = np.arange(start, end, max((end - start) // num_frames, 1)).astype(int)
                temp_frames = video_frames.get_batch(np.arange(start, end)).asnumpy()
                if temp_frames is None:
                    raise ValueError("temp_frames is unexpectedly None")
                tensor_frames = torch.from_numpy(temp_frames)
                tensor_frames = tensor_frames[torch.tensor((indices - start).tolist())]
            else:

                def nearest_smaller_4k_plus_1(n):
                    remainder = n % 4
                    if remainder == 0:
                        return n - 3
                    else:
                        return n - remainder + 1

                start = int(self.skip_frame_num)
                end = int(ori_video_len - self.skip_frame_num)
                # 3D VAE requires the number of frames to be 4k+1
                num_frames = nearest_smaller_4k_plus_1(end - start)
                end = int(start + num_frames)
                temp_frames = video_frames.get_batch(np.arange(start, end)).asnumpy()
                if temp_frames is None:
                    raise ValueError("temp_frames is unexpectedly None")
                tensor_frames = torch.from_numpy(temp_frames)

        # the len of indices may be less than num_frames, due to round error
        tensor_frames = pad_last_frame(
            tensor_frames, self.num_frames
        )
        # [T, H, W, C] -> [T, C, H, W]
        tensor_frames = tensor_frames.permute(0, 3, 1, 2)
        tensor_frames = resize_for_rectangle_crop(tensor_frames, [self.max_height, self.max_width],
                                                  reshape_mode="center")
        tensor_frames = (tensor_frames - 127.5) / 127.5
        return tensor_frames


    def combine_data_video_process(
            self, vframes, is_decord_read=True, predefine_num_frames=13,
            start_frame_idx=0, clip_total_frames=-1, resolution_crop=(None, None, None, None)
    ):
        
        total_frames = len(vframes) if clip_total_frames == -1 else clip_total_frames
        fps = vframes.get_avg_fps() if vframes.get_avg_fps() > 0 else 30.0
        s_x, e_x, s_y, e_y = resolution_crop
        # import pdb; pdb.set_trace()
        if self.auto_interval:
            # resample in case high fps, such as 50/60/90/144 -> train_fps(e.g, 24)
            frame_interval = (
                1.0 if abs(fps - self.train_fps) < 0.1 else fps / self.train_fps
            )
        else:
            frame_interval = self.frame_interval
        # some special video should be set to a different number
        frame_indices = np.arange(start_frame_idx, total_frames, frame_interval).astype(
            int
        )
        frame_indices = frame_indices[frame_indices < total_frames]
        # speed up
        max_speed_factor = len(frame_indices) / self.num_frames
        if self.speed_factor > 1 and max_speed_factor > 1:
            speed_factor = min(self.speed_factor, max_speed_factor)
            target_frame_count = int(len(frame_indices) / speed_factor)
            speed_frame_idx = np.linspace(
                0, len(frame_indices) - 1, target_frame_count, dtype=int
            )
            frame_indices = frame_indices[speed_frame_idx]

        #  too long video will be temporal-crop randomly
        if len(frame_indices) > self.num_frames:
            begin_index, end_index = self.temporal_sample(len(frame_indices))
            frame_indices = frame_indices[begin_index:end_index]

        # to find a suitable end_frame_idx, to ensure we do not need pad video
        end_frame_idx = self.find_closest_y(
            len(frame_indices), vae_stride_t=self.ae_stride_t, model_ds_t=self.sp_size
        )
        if end_frame_idx == -1:  # too short that can not be encoded exactly by videovae
            raise IndexError(
                f"video has {total_frames} frames, but need to sample {len(frame_indices)} frames ({frame_indices})"
            )
        frame_indices = frame_indices[:end_frame_idx]
        if predefine_num_frames != len(frame_indices):
            raise ValueError(
                f"predefine_num_frames ({predefine_num_frames}) is not equal with frame_indices ({len(frame_indices)})"
            )
        if len(frame_indices) < self.num_frames and self.drop_short_ratio >= 1:
            raise IndexError(
                f"video has {total_frames} frames, but need to sample {len(frame_indices)} frames ({frame_indices})"
            )
        video = vframes.get_batch(frame_indices).asnumpy()
        video = torch.from_numpy(video) # [77 720 1280 3]
        # import pdb;pdb.set_trace()
        # (T, H, W, C) -> (T C H W)
        video = video.permute(0, 3, 1, 2) # [77 3 720 1280]
        if s_y is not None:
            video = video[:, :, s_y: e_y, s_x: e_x]

        h, w = video.shape[-2:]
        if self.force_resolution: # False
            if h / w > 17 / 16 or h / w < 8 / 16:
                raise AssertionError(
                    f"Only videos with a ratio (h/w) less than 17/16 and more than 8/16 are supported. But the video found ratio is {round(h / w, 2)} with the shape of {video.shape}"
                )
        # TCHW -> TCHW
        video = self.video_transforms(video) # 
        video = video.permute(1, 0, 2, 3)
        return video

    def define_frame_index(self, cap_list):
        new_cap_list = []
        sample_num_frames = []
        sample_size = []
        aesthetic_score = []
        cnt_too_long = 0
        cnt_too_short = 0
        cnt_no_cap = 0
        cnt_no_resolution = 0
        cnt_no_aesthetic = 0
        cnt_resolution_mismatch = 0
        cnt_aspect_mismatch = 0
        cnt_resolution_too_small = 0
        cnt_movie = 0
        cnt_img = 0


        for i in cap_list:
            outer_key = list(i.keys())[0]
            path = i[outer_key]["metadata"]["path"]
            cap = i[outer_key]["metadata"].get("face_cap_qwen", None)
            # ======no caption=====
            if cap is None:
                cnt_no_cap += 1
                continue
            # ======no aesthetic=====
            if i[outer_key]["metadata"].get("aesthetic", None) is None or i[outer_key]["metadata"].get("aes", None) is None:
                cnt_no_aesthetic += 1
            else:
                aesthetic_score.append(i[outer_key]["metadata"].get("aesthetic", None) or i[outer_key]["metadata"].get("aes", None))

            # ======resolution mismatch=====
            if i[outer_key]["metadata"].get("resolution", None) is None:
                cnt_no_resolution += 1
                continue
            else:
                if i[outer_key]["metadata"]["resolution"].get("height", None) is None or i[outer_key]["metadata"]["resolution"].get("width", None) is None:
                    cnt_no_resolution += 1
                    continue
                else:
                    height, width = i[outer_key]["metadata"]["resolution"]["height"], i[outer_key]["metadata"]["resolution"]["width"]
                    
                    if not self.force_resolution:
                        if height <= 0 or width <= 0:
                            cnt_no_resolution += 1
                            continue
                        
                        tr_h, tr_w = maxhwresize(height, width, self.max_hxw)
                        _, _, sample_h, sample_w = get_params(tr_h, tr_w, self.hw_stride)
                        if sample_h <= 0 or sample_w <= 0:
                            cnt_resolution_mismatch += 1
                            continue
                        if sample_h * sample_w < self.min_hxw:
                            cnt_resolution_too_small += 1
                            continue
                        # filter aspect
                        is_pick = filter_resolution(
                            sample_h, 
                            sample_w, 
                            max_h_div_w_ratio=self.hw_aspect_thr, 
                            min_h_div_w_ratio=1 / self.hw_aspect_thr
                        )
                        if not is_pick:
                            cnt_aspect_mismatch += 1
                            continue
                        i[outer_key]["metadata"]["resolution"].update(dict(sample_height=sample_h, sample_width=sample_w))
                        # import pdb;pdb.set_trace()
                    else:
                        aspect = self.max_height / self.max_width
                        is_pick = filter_resolution(
                            height, 
                            width, 
                            max_h_div_w_ratio=self.hw_aspect_thr * aspect, 
                            min_h_div_w_ratio=1 / self.hw_aspect_thr * aspect
                        )
                        if not is_pick:
                            cnt_aspect_mismatch == 1
                            continue
                        sample_h, sample_w = self.max_height, self.max_width

                        i[outer_key]["metadata"]["resolution"].update(dict(sample_height=sample_h, sample_width=sample_w))

            ext = os.path.splitext(path)[-1].lower()
            if ext.lower() in VID_EXTENSIONS: 
                # ======no fps and duration=====
                duration = i[outer_key]["metadata"].get("duration", None)
                fps = i[outer_key]["metadata"].get("fps", None)
                if fps is None or (duration is None and i[outer_key]["metadata"].get("num_frames", None) is None):
                    continue

                i[outer_key]["metadata"]["num_frames"] = round(fps * duration) if i[outer_key]["metadata"].get("num_frames", None) is None else i[outer_key]["metadata"]["num_frames"]

                if self.auto_interval:
                    # resample in case high fps, such as 50/60/90/144 -> train_fps(e.g, 24)
                    frame_interval = 1.0 if abs(fps - self.train_fps) < 0.1 else fps / self.train_fps
                else:
                    frame_interval = 1.0

                start_frame_idx = i[outer_key]["metadata"].get("face_cut", [0])[0]
                end_frame_idx = i[outer_key]["metadata"].get("face_cut", [0])[1]
                
                i[outer_key]["metadata"]["start_frame_idx"] = start_frame_idx
                frame_indices = np.arange(
                    start_frame_idx, end_frame_idx, frame_interval
                ).astype(int)
                frame_indices = frame_indices[frame_indices <  i[outer_key]["metadata"]["num_frames"]]
                # comment out it to enable dynamic frames training
                if (
                        len(frame_indices) < self.num_frames
                        and torch.rand(1, generator=self.generator).item() < self.drop_short_ratio
                ):
                    cnt_too_short += 1
                    continue

                #  too long video will be temporal-crop randomly
                if len(frame_indices) > self.num_frames:
                    begin_index, end_index = self.temporal_sample(len(frame_indices))
                    frame_indices = frame_indices[begin_index:end_index]
                # to find a suitable end_frame_idx, to ensure we do not need pad video
                end_frame_idx = self.find_closest_y(
                    len(frame_indices), vae_stride_t=self.ae_stride_t, model_ds_t=self.sp_size
                )
                if (
                        end_frame_idx == -1
                ):  # too short that can not be encoded exactly by videovae
                    cnt_too_short += 1
                    continue
                frame_indices = frame_indices[:end_frame_idx]

                if "/storage/dataset/movie" in i[outer_key]["metadata"]["path"]:
                    cnt_movie += 1

                i[outer_key]["metadata"]["sample_frame_index"] = frame_indices.tolist()
                i[outer_key]["metadata"]["sample_num_frames"] = len(i[outer_key]["metadata"]["sample_frame_index"])

                new_cap_list.append(i)
            elif ext.lower() in IMG_EXTENSIONS:  # image
                cnt_img += 1
                i[outer_key]["metadata"]["sample_frame_index"] = [0]
                i[outer_key]["metadata"]["sample_num_frames"] = 1
                new_cap_list.append(i)
            elif ext.lower() in TENSOR_EXTENSIONS: # tensor
                new_cap_list.append(i)
            else:
                raise NameError(
                    f"Unknown file extention {path.split('.')[-1]}"
                )
            
            sample_num_frames.append(i[outer_key]["metadata"]["sample_num_frames"])
            sample_size.append(f"{len(i[outer_key]['metadata']['sample_frame_index'])}x{sample_h}x{sample_w}")

        counter = Counter(sample_size)
        total_batch_size = self.batch_size * torch.distributed.get_world_size() * self.gradient_accumulation_size
        total_batch_size = total_batch_size // self.sp_size * self.train_sp_batch_size
        filter_major_num = 4 * total_batch_size
        len_before_filter_major = len(new_cap_list)
        print("len_before_filter_major",len_before_filter_major)
        new_cap_list, sample_size = zip(*[[i, j] for i, j in zip(new_cap_list, sample_size) if counter[j] >= filter_major_num])
        # import pdb;pdb.set_trace()
        counter = Counter(sample_size)
        cnt_filter_minority = len_before_filter_major - len(new_cap_list)
        print(
            f"no_cap: {cnt_no_cap}, too_long: {cnt_too_long}, too_short: {cnt_too_short},"
            f"cnt_filter_minority:{cnt_filter_minority} \n"
            f"no_resolution: {cnt_no_resolution}, resolution_mismatch: {cnt_resolution_mismatch}, "
            f"cnt_resolution_too_small: {cnt_resolution_too_small}, cnt_aspect_mismatch: {cnt_aspect_mismatch}, "
            f"Counter(sample_num_frames): {Counter(sample_num_frames)}, Counter(sample_size): {counter}, "
            f"cnt_movie: {cnt_movie}, cnt_img: {cnt_img}, "
            f"before filter: {len(cap_list)}, after filter: {len(new_cap_list)}"
        )

        if len(aesthetic_score) > 0:
            stats_aesthetic = calculate_statistics(aesthetic_score)
            print(
                f"aesthetic_score: {len(aesthetic_score)}, cnt_no_aesthetic: {cnt_no_aesthetic}\n"
                f"{len([i for i in aesthetic_score if i[outer_key]['metadata']['aesthetic_score']>=5.75])} > 5.75, 4.5 > {len([i for i in aesthetic_score if i[outer_key]['metadata']['aesthetic_score']<=4.5])}\n"
                f"Mean: {stats_aesthetic.get('mean')}, Var: {stats_aesthetic.get('variance')}, Std: {stats_aesthetic.get('std_dev')}\n"
                f"Min: {stats_aesthetic.get('min')}, Max: {stats_aesthetic.get('max')}"
            )
            
        return new_cap_list, sample_num_frames, sample_size

    def find_closest_y(self, x, vae_stride_t=4, model_ds_t=1):
        if x < self.min_num_frames:
            return -1  
        for y in range(x, self.min_num_frames - 1, -1):
            if (y - 1) % vae_stride_t == 0 and ((y - 1) // vae_stride_t + 1) % model_ds_t == 0:
                # 4, 8: y in [29, 61, 93, 125, 157, 189, 221, 253, 285, 317, 349, 381, 413, 445, 477, 509, ...]
                # 4, 4: y in [29, 45, 61, 77, 93, 109, 125, 141, 157, 173, 189, 205, 221, 237, 253, 269, 285, 301, 317, 333, 349, 365, 381, 397, 413, 429, 445, 461, 477, 493, 509, ...]
                # 8, 1: y in [33, 41, 49, 57, 65, 73, 81, 89, 97, 105]
                # 8, 2: y in [41, 57, 73, 89, 105]
                # 8, 4: y in [57, 89]
                # 8, 8: y in [57]
                return y
        return -1


class ImageProcesser:
    """Used for image data preprocessing"""

    def __init__(
            self,
            num_frames=16,
            train_pipeline=None,
            image_reader_type="torchvision",
            image_processer_type="image2video",
            dynamic_image_size=False,
            image_size=224,
            min_dynamic_patch=1,
            max_dynamic_patch=6,
            use_thumbnail=False,
            transform_size=None,
            **kwargs,
    ):
        self.num_frames = num_frames
        self.image_transforms = get_transforms(
            is_video=False, train_pipeline=train_pipeline, transform_size=transform_size
        )
        self.video_transforms = get_transforms(
            is_video=True, train_pipeline=train_pipeline, transform_size=transform_size
        )
        self.train_pipeline = train_pipeline
        self.image_reader_type = image_reader_type
        self.image_processer_type = image_processer_type
        self.dynamic_image_size = dynamic_image_size
        self.image_size = image_size
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.use_thumbnail = use_thumbnail

    def __call__(self, image_path, mode="", num_image=1):
        if self.image_processer_type == "image2video":
            image = self.image_to_video(image_path)
        elif self.image_processer_type == "image2image":
            image = self.image_to_image(image_path)
        else:
            raise NotImplementedError(
                f"Unsupported image processer type: {self.image_processer_type}"
            )
        return image

    def image_to_video(self, image_path):
        image = self.image_reader(image_path)
        image = torch.from_numpy(np.array(image))  # [h, w, c]
        image = rearrange(image, "h w c -> c h w").unsqueeze(0)  # [1 c h w]
        image = self.image_transforms(image)
        video = image.repeat(self.num_frames, 1, 1, 1)
        video = video.permute(1, 0, 2, 3)  # TCHW -> CTHW
        return video

    def image_to_image(self, image_path):
        image = self.image_reader(image_path)
        image = torch.from_numpy(np.array(image))  # [h, w, c]
        image = rearrange(image, "h w c -> c h w").unsqueeze(0)  # [1 c h w]
        # [1 C H W] -> num_img [1 C H W]
        if "human_images" in image_path:
            image = self.image_transforms(image)
        else:
            image = self.video_transforms(image)
        # [1 C H W] -> [C 1 H W]
        image = image.permute(1, 0, 2, 3)
        return image

    def image_reader(self, image_path):
        if self.image_reader_type in ["torchvision", "CLIPImageProcessor"]:
            image = pil_loader(image_path)
        elif self.image_reader_type == "Image":
            image = Image.open(image_path).convert("RGB")  # [h, w, c]
        else:
            raise NotImplementedError(
                f"Unsupported image reader type: {self.image_reader_type}"
            )
        return image


class TextProcesser:
    """Used for text data preprocessing"""

    bad_punct_regex = re.compile(
        r"["
        + "#®•©™&@·º½¾¿¡§~"
        + "\)"
        + "\("
        + "\]"
        + "\["
        + "\}"
        + "\{"
        + "\|"
        + "\\"
        + "\/"
        + "\*"
        + r"]{1,}"
    )

    def __init__(
            self,
            tokenizer=None,
            use_clean_caption=True,
            enable_text_preprocessing=True,
            padding_type="max_length",
            support_chinese=False,
            text_preprocess_methods=None,
            cfg=0.1,
    ):
        self.padding = padding_type
        self.tokenizer = tokenizer
        self.use_clean_caption = use_clean_caption
        self.support_chinese = support_chinese
        self.cfg = cfg
        self.enable_text_preprocessing = enable_text_preprocessing
        self.text_preprocess_methods = text_preprocess_methods

    def __call__(self, texts):
        if self.enable_text_preprocessing:
            texts_info = [
                TextProcesser.text_preprocessing(
                    text,
                    self.use_clean_caption,
                    text_preprocess_methods=self.text_preprocess_methods
                )
                for text in texts
            ]
            texts_info = texts_info if random.random() > self.cfg else [""]
        else:
            texts_info = texts

        if not isinstance(self.tokenizer, list):
            text_tokens_and_mask = self.tokenizer(
                texts_info,
                max_length=self.tokenizer.model_max_length,
                padding=self.padding,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            prompt_ids = text_tokens_and_mask["input_ids"]
            prompt_mask = text_tokens_and_mask["attention_mask"]
        else:
            prompt_ids, prompt_mask = [], []
            for tokenizer in self.tokenizer:
                text_tokens_and_mask = tokenizer(
                    texts_info,
                    max_length=tokenizer.model_max_length,
                    padding=self.padding,
                    truncation=True,
                    return_attention_mask=True,
                    add_special_tokens=True,
                    return_tensors="pt"
                )
                prompt_ids.append(text_tokens_and_mask["input_ids"])
                prompt_mask.append(text_tokens_and_mask["attention_mask"])
        return (prompt_ids, prompt_mask)

    @staticmethod
    def text_preprocessing(text, use_clean_caption=True, support_chinese=False, text_preprocess_methods=None):
        if text_preprocess_methods:
            if isinstance(text_preprocess_methods, list):
                for text_preprocess_method in text_preprocess_methods:
                    text = TextProcesser.text_preprocessing(text, text_preprocess_methods=text_preprocess_method)
            else:
                method_name = text_preprocess_methods["method"]
                param = text_preprocess_methods.get("param", None)
                method = getattr(TextProcesser, method_name, None)
                if method:
                    if param:
                        text = method(text, **param)
                    else:
                        text = method(text)
                else:
                    raise NotImplementedError(f"The text preprocessing method {method_name} is not implemented.")
        else:
            if use_clean_caption:
                text = TextProcesser.clean_caption(text, support_chinese=support_chinese)
            else:
                text = text.lower().strip()
        return text

    @staticmethod
    def basic_clean(text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()
    
    @staticmethod
    def whitespace_clean(text):
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    @staticmethod
    def clean_caption(caption, support_chinese=False):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",
            # noqa
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",
            # noqa
            "",
            caption,
        )  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features="html.parser").text

        # @<nickname>
        caption = re.sub(r"@[\w\d]+\b", "", caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        if not support_chinese:
            caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",
            # noqa
            "-",
            caption,
        )

        # Uniform quotation marks
        caption = re.sub(r"[`´«»“”¨]", '"', caption)
        caption = re.sub(r"[‘’]", "'", caption)

        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)

        # ip adresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        # \n
        caption = re.sub(r"\\n", " ", caption)

        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(
            r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption
        )

        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

        caption = re.sub(
            TextProcesser.bad_punct_regex, r" ", caption
        )  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)

        caption = TextProcesser.basic_clean(caption)

        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(
            r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption
        )
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)

        caption = re.sub(
            r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption
        )  # j2d1a2a...

        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)

        caption.strip()

        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)

        return caption.strip()


def get_seed_worker(seed):
    """Deterministic dataloader"""

    def seed_worker(worker_id):
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker


class SingletonMeta(type):
    """
    This is a metaclass for creating singletons.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class DataSetProg(metaclass=SingletonMeta):
    """
    This is a data program for data multithreaded processing.
    """

    def __init__(self):
        self.cap_list = []
        self.elements = []
        self.num_workers = 1
        self.n_elements = 0
        self.worker_elements = dict()
        self.n_used_elements = dict()

    def set_cap_list(self, num_workers, cap_list, n_elements):
        self.num_workers = num_workers
        self.cap_list = cap_list
        self.n_elements = n_elements
        self.elements = list(range(n_elements))
        print(f"n_elements: {len(self.elements)}", flush=True)

    def get_item(self, work_info):
        if work_info is None:
            worker_id = 0
        else:
            worker_id = work_info.id

        idx = self.worker_elements[worker_id][
            self.n_used_elements[worker_id] % len(self.worker_elements[worker_id])
            ]
        self.n_used_elements[worker_id] += 1
        return idx


def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    h, w = arr.shape[2], arr.shape[3]
    arr = arr.squeeze(0)

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    return arr


def pad_last_frame(tensor, num_frames):
    # T, H, W, C
    if len(tensor) < num_frames:
        pad_length = num_frames - len(tensor)
        # Use the last frame to pad instead of zero
        last_frame = tensor[-1]
        pad_tensor = last_frame.unsqueeze(0).expand(pad_length, *tensor.shape[1:])
        padded_tensor = torch.cat([tensor, pad_tensor], dim=0)
        return padded_tensor
    else:
        return tensor[:num_frames]


def filter_resolution(h, w, max_h_div_w_ratio=17 / 16, min_h_div_w_ratio=8 / 16):
    if h / w <= max_h_div_w_ratio and h / w >= min_h_div_w_ratio:
        return True
    return False


def format_numel_str(numel: int) -> str:
    B = 1024 ** 3
    M = 1024 ** 2
    K = 1024
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"


def collate_fn_default(batch):
    use_mask = False
    if "mask" in batch[0] and isinstance(batch[0]["mask"], int):
        masks = [x.pop("mask") for x in batch]
        input_ids = [x.pop("input_ids") for x in batch]
        input_ids = torch.cat(input_ids, dim=-1)
        use_mask = True
    elif "mask" in batch[0] and isinstance(batch[0]["mask"], torch.Tensor):
        masks = [x.pop("mask") for x in batch]
        input_ids = [x.pop("input_ids") for x in batch]
        masks = torch.cat(masks, dim=0)
        input_ids = torch.cat(input_ids, dim=0)
        use_mask = True

    ret = torch.utils.data.default_collate(batch)

    if use_mask:
        ret["mask"] = masks
        ret["input_ids"] = input_ids

    return ret


def preprocess_multimodal(
        sources: Sequence[str],
        is_multimodal,
        mm_use_im_start_end,
) -> Dict:
    """
    Process multimodal sources by handling image tokens.
    """
    image_token = MODEL_CONSTANTS['llava']["IMAGE_TOKEN"]
    img_start_token = MODEL_CONSTANTS['llava']["IMG_START_TOKEN"]
    img_end_token = MODEL_CONSTANTS['llava']["IMG_END_TOKEN"]

    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if image_token in sentence["value"]:
                sentence["value"] = sentence["value"].replace(image_token, "").strip()
                sentence["value"] = image_token + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
            replace_token = image_token
            if mm_use_im_start_end:
                replace_token = img_start_token + replace_token + img_end_token
            sentence["value"] = sentence["value"].replace(image_token, replace_token)

    return sources


def preprocess_v1(
        sources,
        is_multimodal,
        mm_use_im_start_end,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = True
) -> Dict:
    """
    Process sources for llava-v1 of the preprocessing pipeline.
    """
    sources = preprocess_multimodal(sources, is_multimodal, mm_use_im_start_end)

    ignore_index = MODEL_CONSTANTS['llava']["IGNORE_INDEX"]
    image_token_index = MODEL_CONSTANTS['llava']["IMAGE_TOKEN_INDEX"]
    conv = get_conv_template("llava-v1")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversations = get_formatted_conversations(sources, roles, conv)

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, image_token_index=image_token_index, return_tensors="pt")
             for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = ignore_index
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer, image_token_index))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer, image_token_index)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len: cur_len + instruction_len] = ignore_index

            cur_len += round_len
        target[cur_len:] = ignore_index

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = ignore_index
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids[0],
        labels=targets[0],
    )


def preprocess_plain(
        sources: Sequence[str],
        is_multimodal,
        mm_use_im_start_end,
        tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """
    Process plain text sources for preprocessing.
    """
    sources = preprocess_multimodal(sources, is_multimodal, mm_use_im_start_end)

    image_token_index = MODEL_CONSTANTS['llava']["IMAGE_TOKEN_INDEX"]
    image_token = MODEL_CONSTANTS['llava']["IMAGE_TOKEN"]
    ignore_index = MODEL_CONSTANTS['llava']["IGNORE_INDEX"]
    conv = get_conv_template("llava-plain")
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        source[0]["value"] = image_token
        conversation = source[0]["value"] + source[1]["value"] + conv.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, image_token_index=image_token_index, return_tensors="pt")
                 for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer, image_token_index=image_token_index))
        target[:tokenized_len] = ignore_index

    return dict(input_ids=input_ids[0], labels=targets[0])


def preprocess_internlm(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        num_image: int = 1
) -> Dict:
    """
    Process sources for internvl model preprocessing.
    """
    conv = get_conv_template(template_name)
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversations = get_formatted_conversations(sources, roles, conv)

    im_start_token = MODEL_CONSTANTS['internvl']["IMG_START_TOKEN"]
    im_context_token = MODEL_CONSTANTS['internvl']["IMG_CONTEXT_TOKEN"]
    im_end_token = MODEL_CONSTANTS['internvl']["IMG_END_TOKEN"]

    if not text_only:
        new_conversations = []
        for conversation in conversations:
            for i in range(num_image):
                image_tokens = f"{im_start_token}{im_context_token * num_image_token_list[i]}{im_end_token}"
                conversation = conversation.replace("<image>", image_tokens, 1)
            new_conversations.append(conversation)
        conversations = new_conversations

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding=False if group_by_length or use_packed_ds else "max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())  
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID  # <s>
        parts = conversation.split(conv.roles[1])  # [UNUSED_TOKEN_146]assistant\n
        info = parts[0] + conv.roles[1]
        temp_len = len(tokenizer(info).input_ids) - 1 
        target[cur_len: cur_len + temp_len] = IGNORE_TOKEN_ID
        cur_len = cur_len + temp_len

        for index in range(1, len(parts) - 1):
            info = parts[index]
            part1, part2 = info.split(conv.roles[0])
            temp_len = len(tokenizer(part1).input_ids) - 1
            cur_len = cur_len + temp_len
            part = conv.roles[0] + part2 + conv.roles[1]
            temp_len = len(tokenizer(part).input_ids) - 1
            target[cur_len: cur_len + temp_len] = IGNORE_TOKEN_ID
            cur_len = cur_len + temp_len
        last_info = parts[-1]
        temp_len = len(tokenizer(last_info).input_ids) - 1
        cur_len = cur_len + temp_len

        target[cur_len:] = IGNORE_TOKEN_ID
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}.", flush=True)

    return dict(
        input_ids=input_ids[0],
        labels=targets[0],
        attention_mask=input_ids.ne(tokenizer.pad_token_id)[0],
    )


def preprocess_internvl2_5(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        ds_name: str = None,
        num_image: int = 1
) -> Dict:
    if len(sources) != 1:
        raise ValueError('process only the first conversations')
    conversations = sources[0]

    if conversations[0]['from'] == 'system':
        system_prompt = conversations[0]['value']
        conversations = conversations[1:]  # remove system prompt
    else:
        conv = get_conv_template(template_name)
        system_prompt = conv.system_message

    if not text_only:
        IMG_START_TOKEN_ = MODEL_CONSTANTS['internvl2_5']['IMG_START_TOKEN']
        IMG_CONTEXT_TOKEN_ = MODEL_CONSTANTS['internvl2_5']['IMG_CONTEXT_TOKEN']
        IMG_END_TOKEN_ = MODEL_CONSTANTS['internvl2_5']['IMG_END_TOKEN']
        new_conversations = []
        current_image_idx = 0
        for conversation in conversations:
            if conversation['from'] == 'human':
                image_cnt = conversation['value'].count('<image>')
                for _ in range(image_cnt):
                    if current_image_idx == num_image:
                        break
                    image_tokens = f'{IMG_START_TOKEN_}{IMG_CONTEXT_TOKEN_ * num_image_token_list[current_image_idx]}{IMG_END_TOKEN_}'
                    conversation['value'] = conversation['value'].replace('<image>', image_tokens, 1)
                    current_image_idx += 1
            new_conversations.append(conversation)
        conversations = new_conversations
        if current_image_idx != num_image:
            raise ValueError(f"{current_image_idx} != {num_image}")

    batches, roles = [], []
    if system_prompt is not None:
        batches.append(f'<|im_start|>system\n{system_prompt}<|im_end|>\n')
        roles.append('system')
    for conversation in conversations:
        if conversation['from'] == 'human':
            batches.append(f'<|im_start|>user\n{conversation["value"]}<|im_end|>\n')
            roles.append('human')
        elif conversation['from'] == 'gpt':
            batches.append(f'<|im_start|>assistant\n{conversation["value"]}<|im_end|>\n')
            roles.append('gpt')
        else:
            raise NotImplementedError

    add_bos_token = getattr(tokenizer, 'add_bos_token', False)
    if add_bos_token:  # for InternLM series
        batches[0] = tokenizer.bos_token + batches[0]

    # Tokenize conversations
    input_ids = tokenizer(
        batches,
        return_tensors='np',
        padding=False,
        max_length=tokenizer.model_max_length,
        truncation=False,
    ).input_ids

    if add_bos_token:  # for InternLM series
        input_ids = [item[1:] for item in input_ids]

    final_input_ids, final_targets = [], []
    ignore_ids = tokenizer('<|im_start|>assistant\n', return_tensors='np').input_ids[0]
    ignore_len = ignore_ids.shape[0] - 1 if add_bos_token else ignore_ids.shape[0]
    for role, input_id in zip(roles, input_ids):
        final_input_ids.append(input_id)
        if role == 'system' or role == 'human':
            final_targets.append(np.full(input_id.shape, IGNORE_TOKEN_ID))  # ignore
        elif role == 'gpt':
            target = input_id.copy()
            target[:ignore_len] = IGNORE_TOKEN_ID  # ignore loss for `<|im_start|>assistant\n`
            target[-1:] = IGNORE_TOKEN_ID  # ignore loss for `\n`
            final_targets.append(target)
        else:
            raise NotImplementedError
    input_ids = torch.tensor(np.concatenate(final_input_ids))[:tokenizer.model_max_length]
    targets = torch.tensor(np.concatenate(final_targets))[:tokenizer.model_max_length]

    padding = False if group_by_length or use_packed_ds else True
    if padding:
        current_length = input_ids.size(0)
        padding_length = tokenizer.model_max_length - current_length
        input_ids = F.pad(input_ids, (0, padding_length), value=tokenizer.pad_token_id)
        targets = F.pad(targets, (0, padding_length), value=IGNORE_TOKEN_ID)

    input_ids = input_ids.unsqueeze(0)
    targets = targets.unsqueeze(0)

    return dict(
        input_ids=input_ids[0],
        labels=targets[0],
        attention_mask=input_ids.ne(tokenizer.pad_token_id)[0],
    )


def tokenizer_image_token(prompt, tokenizer, image_token_index, return_tensors=None):
    """
    Tokenize prompts with image tokens.
    """
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def get_formatted_conversations(sources, roles, conv):
    """
    Format conversations based on provided roles and conversation template.
    """
    # Apply prompt templates
    conversations = []
    for source in sources:
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            if role != conv.roles[j % 2]:
                raise ValueError(
                    f"Role mismatch at {sentence}, expected {conv.roles[j % 2]}, got {role}")
            sentence["value"] = sentence["value"].strip()
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    return conversations


def preprocess(
        template_name,
        sources,
        tokenizer,
        num_image_token_list,
        group_by_length,
        is_multimodal,
        mm_use_im_start_end,
        num_image: int = 1
):
    """
    Select and run the appropriate preprocessing function based on template name.
    """
    if template_name == "internlm2-chat":
        ret = preprocess_internlm(template_name, sources,
                                  tokenizer, num_image_token_list,
                                  group_by_length=group_by_length,
                                  num_image=num_image)
    elif template_name == "internvl2_5":
        ret = preprocess_internvl2_5(template_name, sources,
                                     tokenizer, num_image_token_list,
                                     group_by_length=group_by_length,
                                     num_image=num_image)
    elif template_name == "llava-v1":
        ret = preprocess_v1(
            sources,
            is_multimodal,
            mm_use_im_start_end,
            tokenizer,
            has_image=True)
    elif template_name == "llava-plain":
        ret = preprocess_plain(
            sources,
            is_multimodal,
            mm_use_im_start_end,
            tokenizer)
    else:
        raise ValueError("%s preprocessor is not implemented" % type(template_name))
    return ret


def build_iterations(train_dl=None, val_dl=None, test_dl=None, iterator_type="cyclic"):

    def _cyclic_iter(dl):
        while True:
            for x in dl:
                yield x
    
    def _get_iterator(dataloader, iter_type=iterator_type):
        """Return dataset iterator."""
        if iter_type == "single":
            return iter(dataloader)
        elif iter_type == "cyclic":
            return iter(_cyclic_iter(dataloader))
        else:
            raise NotImplementedError("unexpected iterator type")
    
    if train_dl is not None:
        train_data_iterator = _get_iterator(train_dl)
    else:
        train_data_iterator = None

    if val_dl is not None:
        valid_data_iterator = _get_iterator(val_dl)
    else:
        valid_data_iterator = None

    if test_dl is not None:
        test_data_iterator = _get_iterator(test_dl)
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator


def get_value_from_args(key: str, default_value=None):
    """
    Get value from global args
    """
    try:
        keys = key.split(".")
        config = get_args()
        for key in keys:
            config = getattr(config, key)
        return config
    except AttributeError as e:
        if default_value is None:
            raise KeyError(f"Configuration key '{key}' not found, please check.") from e
        logger.info(f"Configuration key '{key}' not found, using default value: {default_value}.")
        return default_value


def cal_gradient_accumulation_size():
    args = get_args()
    world_size = torch.distributed.get_world_size()
    acc = int(args.global_batch_size / world_size / args.micro_batch_size * mpu.get_tensor_model_parallel_world_size()
                  * mpu.get_context_parallel_world_size() * mpu.get_pipeline_model_parallel_world_size())
    
    if getattr(args, "dist_train", False):
        from mindspeed.multi_modal.dist_train.parallel_state import is_in_subworld
        from mindspeed.multi_modal.dist_train.config.dist_train_config import get_dist_model_config
        if is_in_subworld("vae"):
            dit_cfg = get_dist_model_config('dit')
            acc = int(
                args.global_batch_size / dit_cfg.world_size / args.micro_batch_size * dit_cfg.tensor_model_parallel_size
                * dit_cfg.context_parallel_size * dit_cfg.pipeline_model_parallel_size)
    return acc
