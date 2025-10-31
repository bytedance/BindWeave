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
def patch_wan_s2v_dit():
    from mindspeed_mm.models.predictor.predict_model import PREDICTOR_MODEL_MAPPINGS
    from s2v.mindspeed_mm.models.predictor.dits.wan_s2v_dit import WanS2VDiT
    PREDICTOR_MODEL_MAPPINGS['wan_s2v_dit'] = WanS2VDiT

@register_patch('wan_s2v_processor')
def patch_feature_extractor():
    from mindspeed_mm.utils.extra_processor.i2v_processors import I2V_PROCESSOR_MAPPINGS
    from s2v.mindspeed_mm.utils.extra_processor.wan_s2v_processor import WanVideoS2VProcessor
    I2V_PROCESSOR_MAPPINGS['wan_s2v_processor'] = WanVideoS2VProcessor


@register_patch('wan_s2v_pipeline')
def patch_wan_s2v_pipeline():
    from mindspeed_mm.tasks.inference.pipeline import sora_pipeline_dict
    from s2v.mindspeed_mm.tasks.inference.pipeline.wan_pipeline_s2v import WanS2VPipeline
    sora_pipeline_dict['WanS2VPipeline'] = WanS2VPipeline


def patch_all():
    for _,fn in ALL_PATCHS.items():
        fn()
