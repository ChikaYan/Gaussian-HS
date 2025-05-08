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

import torch
from torch import nn
import numpy as np
from .utils.graphics_utils import getWorld2View2, getProjectionMatrix, getWorld2View2_tensor

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", imgs_size=[512,512],
                 cam_offset=None,
                 ):
        '''
        cam_t_offset: [3] differentiable offset to camera
        '''
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = imgs_size[1]
        self.image_height = imgs_size[0]

        # if gt_alpha_mask is not None:
        #     self.original_image *= gt_alpha_mask.to(self.data_device)
        # else:
        #     self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        # self.world_view_transform = torch.tensor(getWorld2View2_tensor(R, T, trans, scale)).transpose(0, 1).cuda()
        self.world_view_transform = getWorld2View2_tensor(R, T, trans, scale).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.cam_offset = cam_offset

    def project_with_offset(self, xyz, detach_offset=False):
        '''
        Project 3D xyz [B, 3] into pixel coordinate yx [B, 2]. If camera translation offset is given, apply it in a differentaible way 
        
        '''

        if self.cam_offset is None:
            cam_offset = torch.zeros_like(xyz)
        else:
            cam_offset = self.cam_offset

        if detach_offset:
            cam_offset = cam_offset.clone().detach()

        mean3d = torch.cat([xyz + cam_offset[None,:], torch.ones_like(xyz[:, :1])], dim=1) 
        mean3d = mean3d.t()
        xy = self.full_proj_transform.t() @ mean3d
        xy = xy/xy[3]
        xy = torch.nan_to_num(xy, 0, 0, 0)
        xy = xy[:2].T #[N, 2] pixel coordinate of the gaussians
        xy = (xy + 1.) / 2.

        yx = torch.flip(xy, dims=[1]) # [W coord, H coord to H coord W coord]

        return yx
    
    def apply_cam_offset(self, xyz):
        '''
        Apply differentiable cam offset to xyz
        
        '''

        if self.cam_offset is None:
            return xyz
        
        return xyz + self.cam_offset[None,...]


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

