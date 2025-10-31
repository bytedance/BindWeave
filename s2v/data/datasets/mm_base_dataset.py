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

import numpy as np
import torch
import torchvision
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS

from s2v.data.data_utils.utils import VID_EXTENSIONS, DataFileReader


class MMBaseDataset(Dataset):
    """
    A base mutilmodal dataset,  it's to privide basic parameters and method

    Args: some basic parameters from dataset_param_dict in config.
        data_path(str):  csv/json/parquat file path
        data_folder(str): the root path of multimodal data
    """

    def __init__(
        self,
        data_path: str = "",
        data_folder: str = "",
        return_type: str = "list",
        data_storage_mode: str = "standard",
        **kwargs,
    ):
        self.data_path = data_path
        self.data_folder = data_folder
        self.data_storage_mode = data_storage_mode
        self.get_data = DataFileReader(data_storage_mode=data_storage_mode)
        self.data_samples = self.get_data(self.data_path, return_type=return_type)

    def __len__(self):
        return len(self.data_samples)

    # must be reimplemented in the subclass
    def __getitem__(self, index):
        raise AssertionError("__getitem__() in dataset is required.")

    def get_type(self, path):
        ext = os.path.splitext(path)[-1].lower()
        if ext.lower() in VID_EXTENSIONS:
            return "video"
        elif ext.lower() in IMG_EXTENSIONS:
            return "image"
        else:
            raise NotImplementedError(f"Unsupported file format: {ext}")
