# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: Apache-2.0

import torch
import torch_npu

ALL_PATCHS = dict()

def register_patch(name):
    assert name not in ALL_PATCHS, 'patch name {} already registered!'.format(name)
    def register_fn(fn):
        ALL_PATCHS[name] = fn
        return fn
    return register_fn

@register_patch('wan_s2v_dit')
def patch_wan_r2v_dit():
    from mindspeed_mm.models.predictor.predict_model import PREDICTOR_MODEL_MAPPINGS
    from s2v.models.predictor.wan_dit import WanS2VDiT
    PREDICTOR_MODEL_MAPPINGS['wan_s2v_dit'] = WanS2VDiT

@register_patch('wan_r2v_processor')
def patch_feature_extractor():
    from mindspeed_mm.utils.extra_processor.i2v_processors import I2V_PROCESSOR_MAPPINGS
    from r2v.mindspeed_mm.utils.extra_processor.wan_i2v_processor import WanVideoR2VProcessor
    I2V_PROCESSOR_MAPPINGS['wan_r2v_processor'] = WanVideoR2VProcessor


@register_patch('wan_s2v_pipeline')
def patch_wan_r2v_pipeline():
    from mindspeed_mm.tasks.inference.pipeline import sora_pipeline_dict
    from s2v.tasks.inference.pipeline.wan_pipeline_s2v import WanS2VPipeline
    sora_pipeline_dict['wan_s2v_pipeline'] = WanS2VPipeline


def patch_all():
    for _,fn in ALL_PATCHS.items():
        fn()