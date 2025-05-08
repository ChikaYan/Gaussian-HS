#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os
from typing import Optional, Literal
from dataclasses import dataclass

@dataclass
class ModelParams: 
    sh_degree: int = 3
    layer_model: Literal['none', 'bg', 'both'] = 'bg'
    # layer_model: Literal['none', 'bg', 'both'] = 'none'
    layer_encoding: Literal['fourier', 'hash'] = 'hash'
    layer_apply_2d_deform: bool = True
    layer_parser_type: str = 'mlp'
    layer_out_rescale: int = 1
    layer_feature_dim: int = 32
    layer_input_exp_multires: int = -1 # set to -2 to disable layer input
    layer_input_t_multires: int = -2
    layer_input_pose_multires: int = 4 # input camera pose to layer model


@dataclass
class PipelineParams:
    convert_SHs_python = False
    compute_cov3D_python = True # to apply transformation to convariance as well
    debug = False

@dataclass
class OptimizationParams:
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    # layer_bg_lr_init = 1e-4
    # layer_bg_lr_final = 1e-4
    # layer_bg_lr_delay_mult = 1
    # layer_bg_lr_max_steps = 30_000
    # layer_fg_lr_init = 0.0025
    # layer_fg_lr_final = 0.0025
    # layer_fg_lr_delay_mult = 1
    # layer_fg_lr_max_steps = 30_000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    percent_dense: float = 0.01
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_grad_threshold: int = 0.001
    densify_grad_threshold_vgg_warmup: int = 0.001

