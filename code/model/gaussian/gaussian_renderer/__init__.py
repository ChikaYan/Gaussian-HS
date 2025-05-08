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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from ..gaussian_model import GaussianModel
from ..utils.sh_utils import eval_sh

IDX = 1700


def render(
        viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, 
        override_color = None, transformed_xyz=None, cov_transform=None, 
        xyz_shift=None,
        color_shift=None,
        anchor_xyz=None,
        anchor_rgb=None,
        anchor_rot=None,
        anchor_scale=None,
        anchor_opacity=None,
        train_log=False, # return extra img for logging
        need_separate_gs_alpha=False,
        forward_anchor=False,
        rotate_view_dir=False,
        get_anchor_visibility_mask=False,
        iter=None,
        hide_anchor=False,
        ):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    if transformed_xyz is not None:
        means3D = transformed_xyz
    else:
        means3D = pc.get_xyz

    if xyz_shift is not None:
        means3D_ = means3D.clone()
        means3D += xyz_shift

    means3D = viewpoint_camera.apply_cam_offset(means3D)

    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier, cov_transform)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None

    shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    dir_pp = (means3D - viewpoint_camera.camera_center.detach().clone().repeat(pc.get_features.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    
    # apply rotation to sh dir
    if rotate_view_dir and cov_transform is not None:
        dir_pp_normalized = torch.bmm(cov_transform, dir_pp_normalized[...,None]).squeeze(-1)
        dir_pp_normalized = dir_pp_normalized/dir_pp_normalized.norm(dim=1, keepdim=True)
        

    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    
    if color_shift is not None:
        colors_precomp_ = colors_precomp.clone()
        colors_precomp += color_shift                

    # TODO: might need to rotate the SH: https://github.com/graphdeco-inria/gaussian-splatting/issues/176
        
    if anchor_xyz is not None and not forward_anchor:
        anchor_xyz = viewpoint_camera.apply_cam_offset(anchor_xyz)
        means3D = torch.concat([means3D, anchor_xyz],axis=0)
        opacity = torch.concat([opacity, anchor_opacity],axis=0)
        anchor_cov = pc.covariance_activation(anchor_scale, scaling_modifier, anchor_rot, None)
        cov3D_precomp = torch.concat([cov3D_precomp, anchor_cov],axis=0)
        colors_precomp = torch.concat([colors_precomp, anchor_rgb],axis=0)



    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(means3D, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    means2D = screenspace_points

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth, alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    

    ret = {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "alpha": alpha,
            }
    
    if anchor_xyz is not None and not forward_anchor and get_anchor_visibility_mask:
        try:
            anchor_xyz.retain_grad()
            torch.abs(rendered_image).mean().backward(retain_graph=True)
            anchor_visibility_mask = anchor_xyz.grad.abs().sum(axis=-1) > 1e-8
            ret['anchor_visibility_mask'] = anchor_visibility_mask
        except:
            pass
            

    
    if forward_anchor and anchor_xyz is not None:
        # render anchor gaussians separatly and alpha blend the results
        anchor_xyz = viewpoint_camera.apply_cam_offset(anchor_xyz)
        means3D = anchor_xyz
        opacity = anchor_opacity
        anchor_cov = pc.covariance_activation(anchor_scale, scaling_modifier, anchor_rot, None)
        # opacity = anchor_opacity * 100
        # anchor_cov = pc.covariance_activation(torch.clamp(anchor_scale, 1e-5), scaling_modifier, anchor_rot, None)
        cov3D_precomp = anchor_cov
        colors_precomp = anchor_rgb

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(means3D, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        means2D = screenspace_points

        anchor_render, _, _, anchor_alpha = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = None,
            rotations = None,
            cov3D_precomp = cov3D_precomp)
        
        ## for logging regular gs with white background
        # render_regular_gs = rendered_image + (1.-alpha)


        # alpha blending
        alpha_no_anchor = alpha
        if not hide_anchor:
            # note that the returned render RGB are alread multiplied with alpha to blend with black background
            rendered_image = anchor_render + (1.-anchor_alpha) * rendered_image
            # rendered_image = anchor_alpha * anchor_render + (1.-anchor_alpha) * alpha * rendered_image
            alpha = anchor_alpha + alpha - (anchor_alpha * alpha)

        anchor_render_green = anchor_render + (1.-anchor_alpha) * torch.tensor([0.,1.,0.])[:,None,None].type_as(anchor_render)
        # anchor_render_green = anchor_render + (1.-anchor_alpha)

        ret.update({
            "render": rendered_image,
            "alpha": alpha,
            "render_img_gs": anchor_render_green,
            # "render_regular_gs": render_regular_gs,
            "alpha_no_anchor": alpha_no_anchor,
        })

        # if False:
        #     global IDX

        #     # rendered_image_ = torch.clamp(rendered_image + (1.-alpha), 0., 1,)
        #     # import torchvision; torchvision.utils.save_image(rendered_image_[None,...], f'../paper/extreme_demo/gs_{IDX:05d}.png')
            
        #     # anchor_render = torch.clamp(anchor_render + (1.-anchor_alpha), 0., 1,)
        #     # import torchvision; torchvision.utils.save_image(anchor_render[None,...], '../paper/extreme_demo/anchor.png')

        # #     rendered_image = anchor_render + (1.-anchor_alpha) * rendered_image
        # #     alpha = anchor_alpha + alpha - (anchor_alpha * alpha)
        # #     rendered_image_ = torch.clamp(rendered_image + (1.-alpha), 0., 1,)
        # #     import torchvision; torchvision.utils.save_image(rendered_image_[None,...], '../paper/gs_anchor.png')


            # anchor_cov = pc.covariance_activation(torch.full_like(anchor_scale, 0.01), scaling_modifier, anchor_rot, None)
            # # opacity = anchor_opacity * 100
            # # anchor_cov = pc.covariance_activation(torch.clamp(anchor_scale, 1e-5), scaling_modifier, anchor_rot, None)
            # cov3D_precomp = anchor_cov

            # anchor_render, _, _, anchor_alpha = rasterizer(
            #     means3D = means3D,
            #     means2D = means2D,
            #     shs = None,
            #     colors_precomp = colors_precomp,
            #     opacities = torch.ones_like(opacity),
            #     scales = None,
            #     rotations = None,
            #     cov3D_precomp = cov3D_precomp)
            
            # anchor_render = torch.clamp(anchor_render + (1.-anchor_alpha), 0., 1,)
            # import torchvision; torchvision.utils.save_image(anchor_render[None,...], f'../paper/pose3/anchor_593.png')
            # import torchvision; torchvision.utils.save_image(anchor_render[None,...], f'../paper/extreme_demo/anchor_opaque_{IDX:05d}.png')

        #     IDX += 1

            

        # im = anchor_render + 1 - anchor_alpha

        # im = rendered_image + 1 - (1 - ((1.-anchor_alpha) * (1.-alpha)))

        # import torchvision; torchvision.utils.save_image(torch.clamp(anchor_render[None,...], 0, 1), 'test.png')

    else:
        
        if anchor_xyz is not None and need_separate_gs_alpha:
            # render again without anchors to obtain alpha for regularization
            n_points = means3D.shape[0] - anchor_xyz.shape[0]
            means3D = means3D[:n_points]
            opacity = opacity[:n_points]
            cov3D_precomp = cov3D_precomp[:n_points]
            colors_precomp = colors_precomp[:n_points]

            screenspace_points_ = torch.zeros_like(means3D, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
            try:
                screenspace_points_.retain_grad()
            except:
                pass
            means2D = screenspace_points_

            _, _, _, alpha_no_anchor = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = None,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = None,
                rotations = None,
                cov3D_precomp = cov3D_precomp)
            
            ret['alpha_no_anchor'] = alpha_no_anchor

        
        # import torchvision
        # torchvision.utils.save_image(torch.clamp(render_img_gs[None, ...], 0, 1), 'test.png')
        

        
        with torch.no_grad():
            if train_log:
                # if xyz_shift is not None:
                #     render_img_no_lift, _, _, _ = rasterizer(
                #                     means3D = means3D_,
                #                     means2D = None,
                #                     shs = None,
                #                     colors_precomp = colors_precomp_,
                #                     opacities = opacity,
                #                     scales = scales,
                #                     rotations = rotations,
                #                     cov3D_precomp = cov3D_precomp)
                #     ret['render_img_no_lift'] = render_img_no_lift


                if anchor_xyz is not None:
                    means3D = anchor_xyz
                    opacity = anchor_opacity
                    anchor_cov = pc.covariance_activation(anchor_scale, scaling_modifier, anchor_rot, None)
                    cov3D_precomp = anchor_cov
                    colors_precomp = anchor_rgb

                    render_img_gs, _, _, _ = rasterizer(
                        means3D = means3D,
                        means2D = None,
                        shs = None,
                        colors_precomp = colors_precomp,
                        opacities = opacity,
                        scales = None,
                        rotations = None,
                        cov3D_precomp = cov3D_precomp)
                    ret['render_img_gs'] = render_img_gs
                    
                    # valid_mask = ~((anchor_rgb > 0.9).all(axis=-1))
                    # render_img_gs, _, _, alpha = rasterizer(
                    #     means3D = means3D[valid_mask],
                    #     means2D = None,
                    #     shs = None,
                    #     colors_precomp = colors_precomp[valid_mask],
                    #     opacities = opacity[valid_mask],
                    #     scales = None,
                    #     rotations = None,
                    #     cov3D_precomp = cov3D_precomp[valid_mask])
                    

                    # torchvision.utils.save_image(torch.clamp(alpha[None, ...].repeat([1,3,1,1]), 0, 1), 'test_alpha.png')


    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return ret
