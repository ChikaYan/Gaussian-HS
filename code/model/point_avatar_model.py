import math
from functools import partial

import torch
import torch.nn as nn
from flame.FLAME import FLAME
from pytorch3d.ops import knn_points
from pytorch3d.renderer import (AlphaCompositor,
                                PerspectiveCameras,
                                PointsRasterizationSettings,
                                PointsRasterizer,
                                )
from pytorch3d.structures import Pointclouds

# from functorch import vmap

from model.deformer_network import ForwardDeformer
from model.gaussian.gaussian_model import GaussianModel
from model.gaussian.utils.graphics_utils import BasicPointCloud, focal2fov
from model.gaussian.utils.sh_utils import SH2RGB
import numpy as np
from model.gaussian.gaussian_renderer import render as gs_render
from model.gaussian import arguments as gs_args 
from model.gaussian.cameras import Camera as GsCamera 
# from model.layer.layer_model import LayerModel
# from model.layer.layer_model import DeformLayerNetwork
# from model.layer.lift_deform_model import LiftDeformNetwork
from model.layer.gs_img_model import GsImgNetwork


from torch.profiler import profile, record_function, ProfilerActivity

print_flushed = partial(print, flush=True)

class PointAvatar(nn.Module):
    def __init__(self, conf, shape_params, img_res, canonical_expression, canonical_pose, use_background, gs_model_args: gs_args.ModelParams=None,
                 data_sample=None):
        super().__init__()
        self.conf = conf
        self.FLAMEServer = FLAME('./flame/FLAME2020/generic_model.pkl', './flame/FLAME2020/landmark_embedding.npy',
                                 n_shape=100,
                                 n_exp=50,
                                 shape_params=shape_params,
                                 canonical_expression=canonical_expression,
                                 canonical_pose=canonical_pose).cuda()
        self.FLAMEServer.canonical_verts, self.FLAMEServer.canonical_pose_feature, self.FLAMEServer.canonical_transformations = \
            self.FLAMEServer(expression_params=self.FLAMEServer.canonical_exp, full_pose=self.FLAMEServer.canonical_pose)
        self.FLAMEServer.canonical_verts = self.FLAMEServer.canonical_verts.squeeze(0)
        self.prune_thresh = conf.get_float('prune_thresh', default=0.5)
        self.deformer_network = ForwardDeformer(FLAMEServer=self.FLAMEServer, **conf.get_config('deformer_network'))

        if conf.get_bool('bg_layer_model.enabled', False):
            raise NotImplementedError()
            self.bg_layer_network = DeformLayerNetwork(
                img_res[0], img_res[1], 
                exp_dim=6,
                out_channel=3,
                **conf.get_config('bg_layer_model.args', {})
            ).cuda()
        else:
            self.bg_layer_network = None
            
        if conf.get_bool('lift_deform_model.enabled', False):
            raise NotImplementedError()
            self.lift_deform_model = LiftDeformNetwork(
                exp_dim=6,
                **conf.get_config('lift_deform_model.args', {})
            ).cuda()
        else:
            self.lift_deform_model = None
        self.lift_deform_model_warm_up = False

        if conf.get_bool('gs_img_model.enabled', False):
            self.gs_img_model = GsImgNetwork(
                img_res[0], img_res[1], 
                exp_dim=6,
                **conf.get_config('gs_img_model.args', {})
            )
        else:
            self.gs_img_model = None

        self.forward_anchor = self.conf.get('gs_img_model.forward_anchor', False)
        self.forward_anchor_texture_alpha = self.conf.get('gs_img_model.args.pred_alpha', False)
        self.get_anchor_visibility_mask = self.conf.get('gs_img_model.get_anchor_visibility_mask', False)
        self.rotate_view_dir = self.conf.get('gs_rotate_view_dir', False)
        self.hide_anchor_from_iter = self.conf.get('gs_img_model.hide_anchor_from_iter', None)
        
        self.ghostbone = self.deformer_network.ghostbone
        if self.ghostbone:
            self.FLAMEServer.canonical_transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).float().cuda(), self.FLAMEServer.canonical_transformations], 1)
            
        radius_factor = conf.get_config('point_cloud').get('radius_factor', .5)
        pts = (np.random.random((conf.get_config('point_cloud')['n_init_points'], 3)) - 0.5) * 2. * radius_factor
        shs = np.random.random((pts.shape[0], 3)) / 255.0
        rgb = SH2RGB(shs)

        if self.conf.get('lift_deform_model.feature_im_init', False):
            rgb = np.random.random((pts.shape[0], 3)) / 100.

        pcd = BasicPointCloud(pts, rgb, np.zeros_like(pts))
        self.pc = GaussianModel(sh_degree=gs_model_args.sh_degree, radius_factor=radius_factor)
        self.pc.create_from_pcd(pcd, 1.)
        self.gs_pipe = gs_args.PipelineParams()


        n_points = self.pc.points.shape[0]
        self.img_res = img_res
        self.use_background = use_background
        if self.use_background:
            init_background = torch.zeros(img_res[0] * img_res[1], 3).float().cuda()
            self.background = nn.Parameter(init_background)
        else:
            self.background = torch.ones(img_res[0] * img_res[1], 3).float().cuda()
        self.raster_settings = PointsRasterizationSettings(
            image_size=img_res[0],
            radius=self.pc.radius_factor * (0.75 ** math.log2(n_points / 100)),
            points_per_pixel=10
        )
        # keypoint rasterizer is only for debugging camera intrinsics
        self.raster_settings_kp = PointsRasterizationSettings(
            image_size=self.img_res[0],
            radius=0.007,
            points_per_pixel=1
        )
        self.compositor = AlphaCompositor().cuda()


    def _render(self, point_cloud, cameras, render_kp=False):
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings if not render_kp else self.raster_settings_kp)
        fragments = rasterizer(point_cloud)
        r = rasterizer.raster_settings.radius
        dists2 = fragments.dists.permute(0, 3, 1, 2)
        alphas = 1 - dists2 / (r * r)
        images, weights = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            alphas,
            point_cloud.features_packed().permute(1, 0),
        )
        images = images.permute(0, 2, 3, 1)
        weights = weights.permute(0, 2, 3, 1)


        return images


    def forward(
            self, 
            input, 
            train_log=False, iter=None,
            need_separate_gs_alpha=False, # if ture, need to return an alpha mask only for self.pc gaussians when anchor gaussians are used
            distill_anchor_model=False, # if ture, run a anchor model distillation step. This is only done once at begining of test stage
            distill_anchor_texture=False, # if ture, run a anchor model distillation step. This is only done once at begining of test stage
            initing_anchor=False, # if true, run a anchor initialization step without returning rendering.
            updating_anchor_correspondence=False, # if true, run an anchor corres update step
            pruning_outside_anchor=False,
            clip_texture=False,
            ransac_filter_anchor=False,
            ):
        

        IM_RES = (512, 512) # FIXME: at the moment, we assume the input image is always 512x512
        output = {}
        intrinsics = input["intrinsics"].clone()
        cam_pose = input["cam_pose"].clone()
        R = cam_pose[:, :3, :3]
        T = cam_pose[:, :3, 3]
        flame_pose = input["flame_pose"]
        expression = input["expression"]
        batch_size = flame_pose.shape[0]
        if 'cam_pose_offset' in input:
            cam_pose_offset = input['cam_pose_offset']
        else:
            cam_pose_offset = torch.zeros_like(T)


        # with record_function("FLAME inference"):
        verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose)

        if self.ghostbone:
            # identity transformation for body
            transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float().cuda(), transformations], 1)

        cameras = PerspectiveCameras(device='cuda', R=R, T=T+cam_pose_offset, K=intrinsics)
        # make sure the cameras focal length is logged too
        focal_length = intrinsics[:, [0, 1], [0, 1]]
        cameras.focal_length = focal_length
        cameras.principal_point = cameras.get_principal_point()

        n_points = self.pc.points.shape[0]
        points = self.pc.points

        anchor_inited = self.gs_img_model is not None and (iter is None or iter >= self.gs_img_model.anchor_init_iter)

        if anchor_inited:
            # add anchor points to the transform as well
            anchor_xyz = self.gs_img_model.get_anchor_xyz()

            N_anchor = anchor_xyz.shape[0]
            points = torch.concat([points, anchor_xyz], axis=0)

            n_points = points.shape[0]


        total_points = batch_size * n_points

        
        # with record_function("Transform points"):
        transformed_points, transform_rot = self.transform_pts(pnts_c=points,
                                                pose_feature=pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                betas=expression.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                transformations=transformations.unsqueeze(1).expand(-1, n_points, -1, -1, -1).reshape(total_points, *transformations.shape[1:]),
                                                )
        
        shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(points.detach()) # have to do it again with points detached...

        transformed_points = transformed_points.reshape(batch_size, n_points, 3)
        transform_rot = transform_rot.reshape([n_points, batch_size, 3, 3])

        if anchor_inited:
            transformed_anchors = transformed_points[:, -N_anchor:]
            transformed_points = transformed_points[:, :-N_anchor]
            transform_rot = transform_rot[:-N_anchor]

            n_points = transformed_points.shape[1]


        images = []
        viewspace_point_tensors = []
        visibility_filters = []
        radiis = []

        camera_ts = []

        for batch_id in range(batch_size):
            # convert R from pytorch3D to OpenCV format
            w2c = cam_pose[batch_id].clone()
            c2w = torch.linalg.inv(torch.concat([w2c, torch.tensor([[0,0,0,1]]).type_as(w2c)],axis=0))
            # convert from Pytorch3D (X left Y up Z forward) to OpenCV/Colmap (X right Y down Z forward)
            c2w[:3, 0:2] *= -1

            # get the world-to-camera transform and set R, T
            w2c = torch.linalg.inv(c2w)
            R_gs = w2c[:3,:3].T  # R is stored transposed due to 'glm' in CUDA code

            fx = intrinsics[0,0,0] / 2. * IM_RES[0]
            fy = intrinsics[0,1,1] / 2. * IM_RES[1]
            fovx = focal2fov(fx, IM_RES[1])
            fovy = focal2fov(fy, IM_RES[0])
            gs_T = T[batch_id].clone()
            gs_T[0:2] *= -1.

            cam_offset_gs = cam_pose_offset[batch_id] * torch.tensor([-1., -1., 1.]).type_as(cam_pose_offset)

            camera_ts.append(gs_T)
            gs_cam = GsCamera(None, R_gs, gs_T ,FoVx=fovx, FoVy=fovy, image=None, gt_alpha_mask=None,image_name=None, uid=None, imgs_size=IM_RES, cam_offset=cam_offset_gs)

            if self.lift_deform_model is not None and not self.lift_deform_model_warm_up:
                raise NotImplementedError()
            
            else:
                xyz_lift_shift = None
                color_lift_shift = None


            textured_img = anchor_xyz = anchor_rgb = anchor_rot = anchor_scale = anchor_opacity = None
            anchor_model_output = {}

            if anchor_inited:
                lbs_anchor_xyz = transformed_anchors[batch_id]
                anchor_xyz, anchor_rgb, anchor_opacity, anchor_scale, anchor_rot = self.gs_img_model.get_anchor_data(lbs_anchor_xyz, iter=iter)


            # if iter is not None and self.gs_img_model is not None and iter == self.gs_img_model.anchor_init_iter:
            if initing_anchor:
                lbs_anchor_xyz = self.gs_img_model.end_of_anchor_init(
                    lbs_anchor_xyz=transformed_points[batch_id],
                    gaussian=self.pc,
                    gs_cam=gs_cam,
                    head_mask=input['anchor_head_filter_mask'],
                )
                return

            if self.hide_anchor_from_iter is not None:
                hide_anchor = iter is None or iter > self.hide_anchor_from_iter
            else:
                hide_anchor = False


            # with record_function("Render Gaussian"):
            render_pkg = gs_render(
                gs_cam, 
                self.pc, 
                self.gs_pipe, 
                torch.zeros(3).cuda(), 
                transformed_xyz=transformed_points[batch_id], 
                cov_transform=transform_rot[:,batch_id],
                xyz_shift=xyz_lift_shift, # single batch version
                color_shift=color_lift_shift, # single batch version
                anchor_xyz=anchor_xyz,
                anchor_rgb=anchor_rgb,
                anchor_rot=anchor_rot,
                anchor_scale=anchor_scale,
                anchor_opacity=anchor_opacity,
                train_log=train_log,
                need_separate_gs_alpha=need_separate_gs_alpha,
                forward_anchor=self.forward_anchor,
                rotate_view_dir=self.rotate_view_dir,
                get_anchor_visibility_mask=self.get_anchor_visibility_mask,
                iter=iter,
                hide_anchor=hide_anchor
                )

            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            alpha = render_pkg["alpha"]
            # images = torch.concat([image, alpha], axis=0).permute([1,2,0]).unsqueeze(0)
            images.append(torch.concat([image, alpha], axis=0).permute([1,2,0]))

            viewspace_point_tensors.append(viewspace_point_tensor)
            visibility_filters.append(visibility_filter)
            radiis.append(radii)

            if anchor_inited:
                anchor_visibility_mask = None
                if 'anchor_visibility_mask' in render_pkg:
                    anchor_visibility_mask = render_pkg['anchor_visibility_mask']
                    
                time_input = torch.zeros([batch_size,1]).type_as(transformed_points)
                exp_input = flame_pose[:,:6]
                cam_pose_input = gs_T.clone().unsqueeze(0)

                if self.conf.get('detach_pose_input', False):
                    exp_input = exp_input.clone().detach()
                
                if self.conf.get('detach_cam_input', False):
                    cam_pose_input = cam_pose_input.clone().detach()
                
                body_ldmk = input.get('body_ldmk', None)
                ldmk_drawing = input.get('ldmk_drawing', None)
                frame_noise = input.get('frame_noise', None)
                layer_offset = input.get('layer_offset', None)

                if distill_anchor_model:
                    self.gs_img_model.distill_model(
                        lbs_anchor_xyz=transformed_anchors[batch_id],
                        gs_cam=gs_cam,
                        t=time_input,
                        exp=exp_input, 
                        pose=cam_pose_input, 
                        ldmks=body_ldmk, 
                    )
                    return
                
                if distill_anchor_texture:
                    self.gs_img_model.distill_texture(
                        t=time_input,
                        exp=exp_input, 
                        pose=cam_pose_input, 
                        ldmks=body_ldmk, 
                    )
                    return
                
                if ransac_filter_anchor:
                    self.gs_img_model.ransac_filter_anchor(
                        lbs_anchor_xyz=transformed_anchors[batch_id],
                        gs_cam=gs_cam,
                    )
                    return

                if updating_anchor_correspondence:
                    self.gs_img_model.update_anchor_correspondence(
                        lbs_anchor_xyz=transformed_anchors[batch_id],
                        gs_cam=gs_cam,
                        t=time_input,
                        exp=exp_input, 
                        pose=cam_pose_input, 
                        ldmks=body_ldmk, 
                    )
                    return 
                
                if pruning_outside_anchor:
                    self.gs_img_model.prune_outside_anchor(
                        lbs_anchor_xyz=transformed_anchors[batch_id],
                        gs_cam=gs_cam,
                    )
                    return 


                textured_img, anchor_model_output = self.gs_img_model(
                    lbs_anchor_xyz=transformed_anchors[batch_id],
                    gs_cam=gs_cam,
                    t=time_input,
                    exp=exp_input, 
                    pose=cam_pose_input, 
                    ldmks=body_ldmk, 
                    ldmk_drawing=ldmk_drawing, 
                    frame_noise=frame_noise,
                    iter=iter,
                    layer_offset=layer_offset,
                    anchor_visibility_mask=anchor_visibility_mask,
                    mlp_delta_uv_overwrite=input["smoothed_mlp_delta_uvs"][batch_id] if "smoothed_mlp_delta_uvs" in input else None
                )


                if clip_texture:
                    textured_img = torch.clamp(textured_img, 0., 1.)

                if 'anchor_deform_loss' in anchor_model_output:
                    output['anchor_deform_loss'] = anchor_model_output['anchor_deform_loss']
                if 'anchor_scale_loss' in anchor_model_output:
                    output['anchor_scale_loss'] = anchor_model_output['anchor_scale_loss']
                if 'anchor_opacity_loss' in anchor_model_output:
                    output['anchor_opacity_loss'] = anchor_model_output['anchor_opacity_loss']
                if 'delta_uv_loss' in anchor_model_output:
                    output['delta_uv_loss'] = anchor_model_output['delta_uv_loss']
                if 'mlp_warp_delta_uv' in anchor_model_output:
                    output['mlp_warp_delta_uv'] = anchor_model_output['mlp_warp_delta_uv']
                if 'fine_rgb_reg_loss' in anchor_model_output:
                    output['fine_rgb_reg_loss'] = anchor_model_output['fine_rgb_reg_loss']

                if 'coarse_rgb' in anchor_model_output:
                    output['layer_coarse'] = anchor_model_output['coarse_rgb'].permute([0,2,3,1]).reshape(batch_size, -1, 3)
                    output['layer_fine'] = (anchor_model_output['fine_rgb'][:,:3].permute([0,2,3,1]).reshape(batch_size, -1, 3)) # TODO add support to alpha channel

                output['plain_texture'] = torch.clamp(self.gs_img_model.feature_img.feature_img[:3], 0, 1)

        images = torch.stack(images)
        viewspace_point_tensor = viewspace_point_tensors
        visibility_filter = torch.stack(visibility_filters).any(axis=0)
        radii = torch.stack(radiis).float().mean(axis=0).int()
        camera_ts = torch.stack(camera_ts)



        if not self.training:
            # render landmarks for easier camera format debugging
            landmarks2d, landmarks3d = self.FLAMEServer.find_landmarks(verts, full_pose=flame_pose)
            transformed_verts = Pointclouds(points=landmarks2d, features=torch.ones_like(landmarks2d))
            rendered_landmarks = self._render(transformed_verts, cameras, render_kp=True)


        # with record_function("Blend background"):
        foreground_mask = images[..., 3].reshape(-1, 1)
        if self.bg_layer_network is not None:
            time_input = torch.zeros([batch_size,1]).type_as(images)
            exp_input = flame_pose[:,:6]
            cam_pose_input = camera_ts
            body_ldmk = input.get('body_ldmk', None)
            ldmk_drawing = input.get('ldmk_drawing', None)
            layer_offset = input.get('layer_offset', None)
            frame_noise = input.get('frame_noise', None)

            if self.conf.get('detach_pose_input', False):
                exp_input = exp_input.clone().detach()
            
            if self.conf.get('detach_cam_input', False):
                cam_pose_input = cam_pose_input.clone().detach()


            bg, layer_output = self.bg_layer_network(time_input, exp_input, cam_pose_input, body_ldmk, ldmk_drawing, layer_offset, frame_noise)

            bg = bg.permute([0,2,3,1])


            output['layer_bg'] = bg.reshape(batch_size, -1, 3)
            output['gaussian_only'] = images[..., :3].reshape(batch_size, -1, 3)

            if 'coarse' in layer_output:
                output['layer_coarse'] = layer_output['coarse'].permute([0,2,3,1]).reshape(batch_size, -1, 3)
                output['layer_fine'] = (layer_output['fine'].permute([0,2,3,1]).reshape(batch_size, -1, 3) + 1.) / 2.

            rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bg.reshape(-1, 3)

            rgb_values_detach_bg = images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bg.reshape(-1, 3).clone().detach()
        elif textured_img is not None and self.gs_img_model is not None:
            assert batch_size == 1
            bg = textured_img.permute([0,2,3,1])

            if self.forward_anchor_texture_alpha:
                output['layer_bg'] = (bg[...,:3] * bg[...,3:] + 1. - bg[...,3:]).reshape(batch_size, -1, 3)
            else:
                output['layer_bg'] = bg.reshape(batch_size, -1, 3)
            output['gaussian_only'] = images[..., :3].reshape(batch_size, -1, 3)

            if self.forward_anchor_texture_alpha:
                bg_ = bg.reshape(-1, 4)
                alpha_text = bg_[...,3:]

                rgb_values = alpha_text * bg_[...,:3] \
                    + (1 - alpha_text) * images[..., :3].reshape(-1, 3) \
                    + (1 - foreground_mask) * (1 - alpha_text) * 1.
                
                rgb_values_detach_bg = None

                bg = bg[...,:3] # for bg layer loss

            else:
                rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bg.reshape(-1, 3)
                
                rgb_values_detach_bg = None
        
        else:
            rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
            rgb_values_detach_bg = None
            bg = None
            

        # with record_function("FLAME loss KNN"):
        knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
        flame_distance, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
        index_batch = index_batch.reshape(-1)

        rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)

        if self.bg_layer_network is not None:
            rgb_image_detach_bg = rgb_values_detach_bg.reshape(batch_size, self.img_res[0], self.img_res[1], 3)
        else:
            rgb_image_detach_bg = None

        # training outputs
        output.update({
            'img_res': self.img_res,
            'batch_size': batch_size,
            'predicted_mask': foreground_mask,  # mask loss
            'bg_layer': bg,  # bg layer mask loss
            'rgb_image': rgb_image,
            'rgb_image_detach_bg': rgb_image_detach_bg,
            'canonical_points': pnts_c_flame,
            # for flame loss
            'index_batch': index_batch,
            'posedirs': posedirs,
            'shapedirs': shapedirs,
            'lbs_weights': lbs_weights,
            'flame_posedirs': self.FLAMEServer.posedirs,
            'flame_shapedirs': self.FLAMEServer.shapedirs,
            'flame_lbs_weights': self.FLAMEServer.lbs_weights,
            'viewspace_point_tensor': viewspace_point_tensor,
            'visibility_filter': visibility_filter,
            'radii': radii,
            'xyz_lift_shift': xyz_lift_shift,
            'color_lift_shift': color_lift_shift,
        })

        if 'alpha_no_anchor' in render_pkg:
            output['alpha_no_anchor'] = render_pkg['alpha_no_anchor'].permute([1,2,0]).reshape([-1, 1])


        keys = ['render_img_no_lift', 'render_img_gs', 'render_regular_gs']
        for k in keys:
            if k in render_pkg:
                output[k] = render_pkg[k]


        if not self.training:
            output_testing = {
                'rendered_landmarks': rendered_landmarks.reshape(-1, 3),
                'canonical_verts': self.FLAMEServer.canonical_verts.reshape(-1, 3),
                'deformed_verts': verts.reshape(-1, 3),
                'deformed_points': transformed_points.reshape(batch_size, n_points, 3),
            }
            if self.deformer_network.deform_c:
                output_testing['unconstrained_canonical_points'] = self.pc.points
            output.update(output_testing)

        return output
    

    def transform_pts(self, pnts_c, pose_feature, betas, transformations, deformer_net=None):
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()
        pnts_c.requires_grad_(True)
        total_points = betas.shape[0]
        batch_size = int(total_points / pnts_c.shape[0])
        n_points = pnts_c.shape[0]
        def _func(pnts_c, betas, transformations, pose_feature, deformer_net=None):
            if deformer_net is None:
                deformer_net = self.deformer_network
            shapedirs, posedirs, lbs_weights, pnts_c_flame = deformer_net.query_weights(pnts_c)

            pnts_d, pnts_rot = self.FLAMEServer.forward_pts(pnts_c_flame, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights)
            pnts_d = pnts_d.reshape(-1)
            return pnts_d, pnts_rot

        betas = betas.reshape(batch_size, n_points, *betas.shape[1:]).transpose(0, 1)
        transformations = transformations.reshape(batch_size, n_points, *transformations.shape[1:]).transpose(0, 1)
        pose_feature = pose_feature.reshape(batch_size, n_points, *pose_feature.shape[1:]).transpose(0, 1)
        pnts_d, pnts_rot = _func(pnts_c.squeeze(0), betas.squeeze(1), transformations.squeeze(1), pose_feature.squeeze(1), deformer_net) 

        pnts_d = pnts_d.reshape(-1, batch_size, 3).transpose(0, 1).reshape(-1, 3)

        return pnts_d, pnts_rot
    