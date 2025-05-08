import skimage.measure
import skimage.transform
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .refine_nets.unet import RefineUnetDecoder
import torchvision
import numpy as np
import einops
from .mlp_decoder import MLPDecoder2D
from pytorch3d.ops import sample_farthest_points
from model.gaussian.utils.sh_utils import SH2RGB
from pytorch3d.transforms import euler_angles_to_matrix
import skimage

from ..gaussian.utils.sh_utils import eval_sh, RGB2SH
from scipy import ndimage



def get_embedder(multires, i=1):
    if multires == -1:
        return nn.Identity(), i
    elif multires < -1:
        return lambda x: torch.empty_like(x[..., :0]), 0

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        '''
        Input: [B, C, ...]
        '''
        # return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
    
        inputs = inputs.movedim(1, -1)
        out = torch.cat([fn(inputs) for fn in self.embed_fns], -1)
        return out.movedim(-1, 1)



class RigidDeformNet2D(nn.Module):
    '''
    MLP to predict rigid transformation (rotation and translation) based on head pose and landmarks
    '''
    def __init__(self, D=8, W=256, input_ch=3, output_ch=3, skips=[4], out_rescale=1, act_fn=None):
        super(RigidDeformNet2D, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        self.out_rescale = out_rescale
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input = x
        h = input
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input, h], -1)

        outputs = self.output_linear(h)
        rot_degree = torch.tanh(outputs[:,:1] * self.out_rescale) * 90 # rotation within degree [-90, 90]
        translation = torch.tanh(outputs[:,1:]* self.out_rescale)

        return rot_degree, translation


class FeatureImage(nn.Module):
    def __init__(
            self,
            img_h,
            img_w,
            padding=0,
            feature_dim=3,
            rescale=1.,
            ):
        super(FeatureImage, self).__init__()
        self.img_h = int(img_h * rescale)
        self.img_w = int(img_w * rescale)
        self.rescale = rescale
        # self.feature_h = int(img_h * rescale) + padding*2
        # self.feature_w = int(img_w * rescale) + padding*2
        self.padding = padding
        self.feature_dim = feature_dim
        self.feature_img = nn.Parameter(torch.rand([feature_dim, img_h + padding*2, img_w + padding*2], device='cuda').requires_grad_(True))

        print(f"Creating Feature Image with size {self.feature_img.shape}")

    def forward(self, uv):
        '''
        uv: [B, 2, ...], can go beyond [0, 1]. Values go beyond this range will locate at the padding area
        '''

        B, _,  H, W = uv.shape
        
        # convert from [0,1] to pixel coordinate
        uv = uv * torch.tensor([self.img_h, self.img_w]).type_as(uv)[None,:, None, None] + self.padding

        uv = torch.stack(
            [torch.clamp(uv[:,0], 0, self.img_h + self.padding*2 - 1), 
             torch.clamp(uv[:,1], 0, self.img_w + self.padding*2 - 1)], dim=1)

        x_int = torch.floor(uv[:,1])
        y_int = torch.floor(uv[:,0])

        # Prevent crossing
        x_int = torch.clamp_max(x_int, self.img_h+self.padding*2-2).long()
        y_int = torch.clamp_max(y_int, self.img_w+self.padding*2-2).long()

        x_diff = (uv[:,1] - x_int)[:,None,...]
        y_diff = (uv[:,0] - y_int)[:, None,...]

        out_shape = (B, self.feature_dim, H, W)

        a = self.feature_img[:, y_int.reshape(-1), x_int.reshape(-1)].reshape(out_shape)
        b = self.feature_img[:, y_int.reshape(-1), x_int.reshape(-1)+1].reshape(out_shape)
        c = self.feature_img[:, y_int.reshape(-1)+1, x_int.reshape(-1)].reshape(out_shape)
        d = self.feature_img[:, y_int.reshape(-1)+1, x_int.reshape(-1)+1].reshape(out_shape)

        pixel = a*(1-x_diff)*(1-y_diff) + b*(x_diff) * \
            (1-y_diff) + c*(1-x_diff) * (y_diff) + d*x_diff*y_diff

        return pixel



class MLPDecoder(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=3, skips=[4], out_rescale=1, act_fn=None, init_zero_output=False):
        super(MLPDecoder, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        self.out_rescale = out_rescale
        self.act_fn = act_fn
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        self.output_linear = nn.Linear(W, output_ch)

        if init_zero_output:
            torch.nn.init.constant_(self.output_linear.bias, 0.0)
            torch.nn.init.constant_(self.output_linear.weight, 0.0)

    def forward(self, x):
        
        # input = x.permute([0,2,3,1]).reshape([-1, self.input_ch])
        input = x
        h = input
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input, h], -1)

        outputs = self.output_linear(h)
        # outputs = outputs.reshape([x.shape[0], *x.shape[2:], -1]).permute([0,3,1,2]) * self.out_rescale
        outputs = outputs * self.out_rescale
        if self.act_fn is not None:
            outputs = self.act_fn(outputs)
        return outputs

class GsImgNetwork(nn.Module):
    def __init__(
            self, 
            img_h=512, 
            img_w=512, 
            feature_dim=64, 
            deform_type='mlp',
            apply_rigid_deform=False,
            exp_dim=6,
            uv_multires=10,
            exp_multires=2, # condition on flame parameters (head pose mainly)
            t_multires=-2, # condition on frame id
            pose_multires=2, # condition on camera translation (rot is the same)
            ldmk_multires=10, # condition on openpose/dwpose body landmarks
            exp_multires_color=2, # condition on flame parameters (head pose mainly)
            t_multires_color=-2, # condition on frame id
            pose_multires_color=2, # condition on camera translation (rot is the same)
            ldmk_multires_color=10, # condition on openpose/dwpose body landmarks
            n_ldmk=4, # number of 2D ldmks to condition on
            feature_img_args={},
            use_explict_rgb=True,
            deform_2d_warm_up_iter=2000,
            anchor_init_iter=1000, # init stage to allow anchor gaussians to go to desired places. During this stage, background is not used
            update_base_warping=False,
            pred_alpha=False, # include alpha channel in the texture?
            n_anchor=64, # number of anchors in 1D, total num = n_anchor^2
            init_anchor_radius_range=1.,
            anchor_uniform_scale=False,
            fine_rgb_after_iter=0,
            anchor_init_rescale=1,
            remove_head_anchor=False,
            anchor_opacity_min=0.5,
            apply_affine=False,
            affine_type='projective',
            estimate_affine_ransac_thresh=1.0,
            zero_init_latent=False,
            mlp_warp=True,
            sh_texture=False,
            sh_texture_up_degree_every=10000,
            reg_fine_color_ratio=0.,
            freeze_anchor_rgb=True,
            distill_texture_bbox=None, # use to remove noise in the padding region of the texture
            mask_head_texture_utill_iter=-1,
            deform_net_D=4,
            deform_net_W=128,
            ):
        super(GsImgNetwork, self).__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.uv_embed_fn, uv_embed_ch = get_embedder(uv_multires, 2)
        self.embed_time_fn, time_input_ch = get_embedder(t_multires, 1)
        self.embed_exp_fn, exp_input_ch = get_embedder(exp_multires, exp_dim)
        self.embed_pose_fn, pose_input_ch = get_embedder(pose_multires, 3) # only take camera translation, as rotation is same for all frames
        self.embed_ldmk_fn, ldmk_input_ch = get_embedder(ldmk_multires, n_ldmk * 2)
        self.embed_time_fn_color, time_input_ch_color = get_embedder(t_multires_color, 1)
        self.embed_exp_fn_color, exp_input_ch_color = get_embedder(exp_multires_color, exp_dim)
        self.embed_pose_fn_color, pose_input_ch_color = get_embedder(pose_multires_color, 3) # only take camera translation, as rotation is same for all frames
        self.embed_ldmk_fn_color, ldmk_input_ch_color = get_embedder(ldmk_multires_color, n_ldmk * 2)
        self.apply_rigid_deform = apply_rigid_deform
        self.deform_type = deform_type
        self.N_anchor = n_anchor**2
        self.use_explict_rgb = use_explict_rgb
        self.feature_dim = feature_dim
        self.deform_2d_warm_up_iter = deform_2d_warm_up_iter
        self.anchor_init_iter = anchor_init_iter
        self.update_base_warping = update_base_warping
        self.pred_alpha = pred_alpha
        self.anchor_uniform_scale = anchor_uniform_scale
        self.fine_rgb_after_iter = fine_rgb_after_iter
        self.anchor_init_rescale = anchor_init_rescale
        self.remove_head_anchor = remove_head_anchor
        self.apply_affine = apply_affine
        self.affine_type = affine_type
        self.estimate_affine_ransac_thresh = estimate_affine_ransac_thresh
        self.apply_test_time_affine = False # for reenactment and novel view
        self.zero_init_latent = zero_init_latent
        self.reg_fine_color_ratio = reg_fine_color_ratio # sample a ratio of grid for fine rgb reg
        self.mlp_warp = mlp_warp
        self.sh_texture = sh_texture
        self.sh_texture_up_degree_every = sh_texture_up_degree_every
        self.freeze_anchor_rgb = freeze_anchor_rgb
        self.distill_texture_bbox = distill_texture_bbox
        self.test_time_need_to_update_anchor_correspondence = False
        self.mask_head_texture_utill_iter = mask_head_texture_utill_iter
        if sh_texture:
            # store SH in the texture
            # no neural network parsing is invloved anymore
            assert not use_explict_rgb
            assert feature_dim == 48

            self.sh_activate_degree = 0
            self.sh_max_degree = 3


        self.anchor_uv_texture_target = torch.zeros(self.N_anchor, 2).float().cuda()
        # store it as a parameter so it gets saved and loaded
        self.anchor_uv_texture_target = nn.Parameter(self.anchor_uv_texture_target, requires_grad=False)

        # define a initial view -> texture space warping
        # this will be changed after anchor gaussians are initialized if update_base_warping is True
        # warping_param: [[shift_y, shift_x], [scale_y], [scale_x]]
        self.base_warping_param = nn.Parameter(torch.tensor([[0., 0.], [1., 1.]]).type_as(self.anchor_uv_texture_target.data), requires_grad=False)

        self.anchor_xyz = nn.Parameter(torch.rand([self.N_anchor, 3], device='cuda').requires_grad_(True))
        init_xy = torch.stack(torch.meshgrid(torch.linspace(-1,1,n_anchor), torch.linspace(-1,1,n_anchor))).type_as(self.anchor_xyz)
        init_xy = init_xy.reshape([2, -1]).T * init_anchor_radius_range
        self.anchor_xyz.data[:,:2] = init_xy

        # self.anchor_rgb = nn.Parameter(torch.rand([self.N_anchor, 3], device='cuda').requires_grad_(True))
        self.anchor_rgb = nn.Parameter(torch.full([self.N_anchor, 3], 0.5, device='cuda').requires_grad_(not freeze_anchor_rgb))
        self.opacity_min = anchor_opacity_min
        self.anchor_opacity = nn.Parameter(torch.rand([self.N_anchor, 1], device='cuda').requires_grad_(True))
        self.scaling_interval = [2e-3, 3e-2]
        if anchor_uniform_scale:
            self.anchor_scaling = nn.Parameter(
                torch.log(torch.full([1, 1], 2e-2, device='cuda'))
                .requires_grad_(True))
        else:
            self.anchor_scaling = nn.Parameter(
                torch.log(torch.rand([self.N_anchor, 1], device='cuda') * (self.scaling_interval[1] - self.scaling_interval[0]) + self.scaling_interval[0])
                .requires_grad_(True))


        self.anchor_valid_mask = torch.ones_like(self.anchor_xyz[:,0]).bool()
            
        feature_im_dim = feature_dim
        if use_explict_rgb:
            feature_im_dim += 3

        self.feature_img = FeatureImage(img_h, img_w, feature_dim=feature_im_dim, **feature_img_args)
        if self.zero_init_latent:
            self.feature_img.feature_img.data[...] = 0.
        if use_explict_rgb:
            self.feature_img.feature_img.data[:3,...] = 1.

        if sh_texture:
            # use SH to assign initial white colors
            self.feature_img.feature_img.data[...] = 0.
            self.feature_img.feature_img.data.reshape([3, 16, -1])[:,0,...] += RGB2SH(torch.ones([3]).type_as(self.feature_img.feature_img.data))[:, None]


            
        # pixel coordinate spawning whole image
        self.uv = torch.stack(torch.meshgrid(torch.arange(img_h), torch.arange(img_w))).cuda()[None,...] / max(img_h, img_w) 

        if deform_type == 'mlp':
            if self.apply_rigid_deform:
                raise NotImplementedError
            
            if self.mlp_warp:
                # perform 2d deform
                self.deform_net = MLPDecoder2D(
                    input_ch = uv_embed_ch + time_input_ch + exp_input_ch + pose_input_ch + ldmk_input_ch,
                    output_ch=2, 
                    D=deform_net_D,
                    W=deform_net_W,
                    skips=[],
                    init_zero_output=True,
                    act_fn=None,
                )

            if not self.sh_texture:
                self.parser_net = MLPDecoder2D(
                    input_ch = feature_dim + time_input_ch_color + exp_input_ch_color + pose_input_ch_color + ldmk_input_ch_color,
                    output_ch= 4 if pred_alpha else 3, 
                    D=4,
                    W=128,
                    skips=[],
                    init_zero_output=True,
                    act_fn=None,
                )         
        else:
            raise NotImplementedError()



        self.rgb_activation = torch.sigmoid
        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid

    def base_warping_fn(self, yx):
        '''
        A base warping fn that warps view space pixel coordinate to (intermediate) texture space coordinate
        Should be applied before neural warping field 

        yx: [N, 2, ...] pixel coordinates
        '''
        assert yx.shape[1] == 2 # dim=1 is assume to be 2 digit coordinate
        yx = yx.movedim(1, -1) 
        shift_yx = self.base_warping_param[0]
        scale_yx = self.base_warping_param[1]
        yx = (yx + shift_yx) * scale_yx
        return yx.movedim(-1, 1)
    
    def update_anchor_mask(self):
        '''
        Update anchor mask after the anchor xyz has been loaded from a ckpt
        Needed because pytorch doesn't like storing mask directly as a nn.Parameter ...
        '''

        self.anchor_valid_mask = ~torch.isnan(self.anchor_xyz).all(axis=-1)
        self.N_anchor = torch.count_nonzero(self.anchor_valid_mask)
    

    def get_anchor_xyz(self):
        return self.anchor_xyz[self.anchor_valid_mask]
    
    @torch.no_grad()
    def end_of_anchor_init(self, lbs_anchor_xyz, gaussian, gs_cam, head_mask=None):
        '''
        Re-assign anchor_uv_texture_target and compute base warping fn at the end of anchor initialization stage
        '''


        # randonly sample N_anchor gaussians
        N_gs = gaussian.get_xyz.shape[0]
        assert N_gs >= self.N_anchor
        # ids = torch.randperm(N_gs)[:self.N_anchor].long().to(self.anchor_xyz.device)

        if self.remove_head_anchor:
            # remove gs that located in the head region
            uv = gs_cam.project_with_offset(gaussian.get_xyz)
            assert self.img_h == self.img_w
            uv = torch.clamp(uv * self.img_h, 0, self.img_h-1).long()
            head_mask = head_mask.reshape([self.img_h, self.img_w])

            head_mask_np = head_mask.cpu().detach().numpy()
            head_mask_np = ndimage.binary_dilation(head_mask_np, iterations=32)
            head_mask = torch.from_numpy(head_mask_np).type_as(head_mask)

            # import torchvision;torchvision.utils.save_image(head_mask[None,None,...].float(), 'mask.png')

            mask = ~(head_mask[uv[:,0], uv[:,1]] > 0.1)
        else:
            mask = torch.ones_like(gaussian.get_xyz[:,0]).bool()

        ids = sample_farthest_points(gaussian.get_xyz[mask][None,...], K=self.N_anchor)[1][0]

        self.anchor_xyz.data = gaussian.get_xyz[mask][ids]
        self.anchor_rgb.data = SH2RGB(gaussian.get_features[mask][ids][:,0])
        self.anchor_opacity.data = gaussian._opacity[mask][ids]
        # anchor_scale = gaussian._scaling[mask][ids].mean(axis=-1, keepdim=True)
        anchor_scale = torch.log(torch.exp(gaussian._scaling[mask][ids]).mean(axis=-1, keepdim=True) * self.anchor_init_rescale)

        if self.anchor_uniform_scale:
            self.anchor_scaling.data = anchor_scale.mean().reshape(1,1)
        else:
            self.anchor_scaling.data = anchor_scale


        anchor_uv_view = gs_cam.project_with_offset(lbs_anchor_xyz[mask][ids].detach().clone())


        if self.update_base_warping:
            # define basic warping
            # TODO: maybe we can try more sophasticated warping
            anchor_uv_view_clipped = anchor_uv_view[(anchor_uv_view > 0.).all(axis=-1) & (anchor_uv_view < 1.).all(axis=-1)]
            yx_max = anchor_uv_view_clipped.max(axis=0).values
            yx_min = anchor_uv_view_clipped.min(axis=0).values

            self.base_warping_param.data[0] = -yx_min
            self.base_warping_param.data[1] = 1. / (yx_max - yx_min)

        valid_mask = self.anchor_valid_mask
        # reassign the anchor_uv_texture_target
        self.anchor_uv_texture_target.data[valid_mask] = self.base_warping_fn(anchor_uv_view)
        self.anchor_uv_texture_target.data[~valid_mask] = 0


        return lbs_anchor_xyz # return masked lbs_anchor_xyz for rendering
        

    @torch.no_grad()
    def distill_model(
        self,
        lbs_anchor_xyz, 
        gs_cam, 
        t, 
        exp, 
        pose, 
        ldmks=None, 
        # head_mask=None,
    ):
        '''
        Distill model after training to make things faster
        Fine MLP -> directly goes into coarse texture
        Warp MLP -> update the anchor correspondences, then replaced with projective transformation obtained from 2D correspondences
        '''

        B, _ = t.shape
        assert B == 1

        # fetch embeddings
        transform_net_in = lambda x: x[..., None, None].repeat([1,1,self.img_h, self.img_w])
        deform_in = []
        deform_in.append(transform_net_in(self.embed_exp_fn(exp)))
        deform_in.append(transform_net_in(self.embed_time_fn(t)))
        deform_in.append(transform_net_in(self.embed_pose_fn(pose)))
        if ldmks is not None:
            deform_in.append(transform_net_in(self.embed_ldmk_fn(ldmks)))

        anchor_uv_view = gs_cam.project_with_offset(lbs_anchor_xyz.detach().clone(), detach_offset=True)
        

        # warp this back to texture space
        anchor_uv_view = anchor_uv_view.T[None,:,:,None] # [1, 2, N_anchor, 1]
        anchor_deform_in = [einops.rearrange(x, 'B C H W -> B C (H W) 1')[:,:,:self.N_anchor, :] for x in deform_in]

        anchor_uv_base_warp = self.base_warping_fn(anchor_uv_view)
        anchor_uv_base_warp_original = anchor_uv_base_warp # for MLP input

        
        if self.mlp_warp:
            anchor_deform_in = torch.concat([self.uv_embed_fn(anchor_uv_base_warp_original), *anchor_deform_in], axis=1)
            delta_uv = torch.tanh(self.deform_net(anchor_deform_in))
            anchor_uv_texture = delta_uv + anchor_uv_base_warp
        else:
            anchor_uv_texture = anchor_uv_base_warp
        

        # ### update anchor correspondences ###
        self.anchor_uv_texture_target.data[self.anchor_valid_mask] = anchor_uv_texture.reshape([2, -1]).T.clone()
        self.mlp_warp = False

        anchor_uv_texture = anchor_uv_base_warp


        ### disable anchors out of img ###
        anchor_uv_view = gs_cam.project_with_offset(lbs_anchor_xyz.detach().clone(), detach_offset=True)
        anchor_uv_view_mask = ((anchor_uv_view >= 0.) & (anchor_uv_view <= 1.)).all(dim=1)

        self.anchor_valid_mask[self.anchor_valid_mask.clone()] = anchor_uv_view_mask
        self.N_anchor = torch.count_nonzero(self.anchor_valid_mask)

        

        ### distill fine RGB into coarse ###
        self.distill_texture(t, exp, pose, ldmks)


    @torch.no_grad()
    def ransac_filter_anchor(
        self,
        lbs_anchor_xyz, 
        gs_cam, 
    ):
        
        anchor_uv_view = gs_cam.project_with_offset(lbs_anchor_xyz.detach().clone(), detach_offset=True)

        # warp this back to texture space
        anchor_uv_view = anchor_uv_view.T[None,:,:,None] # [1, 2, N_anchor, 1]

        anchor_uv_base_warp = self.base_warping_fn(anchor_uv_view)



        anchor_uv_texture = anchor_uv_base_warp
        
                
        affine_test_trans, inliers = skimage.measure.ransac(
            (anchor_uv_texture.reshape([2,-1]).T.detach().cpu().numpy(),
            self.anchor_uv_texture_target[self.anchor_valid_mask].detach().cpu().numpy()), 
            skimage.transform.EuclideanTransform, min_samples=100, residual_threshold=0.05, max_trials=1000
            )

        print(f"Numer of inliers: [{np.count_nonzero(inliers)}/{anchor_uv_texture.shape[-2]}]")

        self.anchor_valid_mask[self.anchor_valid_mask.clone()] = torch.from_numpy(inliers).type_as(self.anchor_valid_mask)
        self.N_anchor = torch.count_nonzero(self.anchor_valid_mask)


    @torch.no_grad()
    def distill_texture(
        self,
        t, 
        exp, 
        pose, 
        ldmks=None, 
    ):
        '''
        Distill model after training to make things faster
        Fine MLP -> directly goes into coarse texture, need to mask out noises
        '''

        B, _ = t.shape
        assert B == 1


        ### distill fine RGB into coarse ###
        if self.feature_img.feature_img.data.shape[0] == 3:
            return


        assert not self.sh_texture

        features = self.feature_img.feature_img[None, ...]

        if self.use_explict_rgb:
            corase_rgb = features[:, :3]
            features = features[:, 3:]
        else:
            corase_rgb = 0.

        transform_net_in = lambda x: x[..., None, None].repeat([1,1,features.shape[2], features.shape[3]])
        color_in = []
        color_in.append(transform_net_in(self.embed_exp_fn_color(exp)))
        color_in.append(transform_net_in(self.embed_time_fn_color(t)))
        color_in.append(transform_net_in(self.embed_pose_fn_color(pose)))
        if ldmks is not None:
            color_in.append(transform_net_in(self.embed_ldmk_fn_color(ldmks)))

        parser_in = torch.concat([features, *color_in], axis=1)
        fine_rgb = torch.tanh(self.parser_net(parser_in))


        # ###### deeplab v3 approach ######

        segmentation_model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True).cuda()
        segmentation_model.eval()
        normalize = torchvision.transforms.Compose([
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        texture = torch.clamp(corase_rgb + fine_rgb, 0., 1.)
        texture = normalize(texture)
        seg_out = segmentation_model(texture)['out'].argmax(1,keepdim=True)
        mask = seg_out != 0


        mask_np = mask[0,0].cpu().numpy()
        mask_np = ndimage.binary_dilation(mask_np, iterations=32)

        mask = torch.from_numpy(mask_np)[None,None].type_as(mask)
        mask = ~mask


        ###### bbox ######
        if self.distill_texture_bbox is not None:
            padding = self.feature_img.padding
            uv = torch.stack(torch.meshgrid(torch.arange(-padding, self.img_h+padding), torch.arange(-padding, self.img_h+padding))).cuda()[None,...]
            # self.distill_texture_bbox = [-30, 320, 520, 520]
            bbox = self.distill_texture_bbox
            bbox_mask = (uv[:,:1] < bbox[1]) | (uv[:,1:] < bbox[0]) | (uv[:,:1] > bbox[3]) | (uv[:,1:] > bbox[2])
            mask = bbox_mask | mask


        ##################################################

        texture = corase_rgb+fine_rgb
        texture[mask.repeat([1,3,1,1])] = 1.
        self.feature_img.feature_img = nn.Parameter(texture[0]).requires_grad_(False)


        self.fine_rgb_after_iter = torch.inf

        self.use_explict_rgb = True
        self.feature_dim = 3
        self.feature_img.feature_dim = 3

    @torch.no_grad()
    def update_anchor_correspondence(
        self,
        lbs_anchor_xyz, 
        gs_cam, 
        t, 
        exp, 
        pose, 
        ldmks=None, 
    ):
        B, _ = t.shape
        assert B == 1

        # fetch embeddings
        transform_net_in = lambda x: x[..., None, None].repeat([1,1,self.img_h, self.img_w])
        deform_in = []
        deform_in.append(transform_net_in(self.embed_exp_fn(exp)))
        deform_in.append(transform_net_in(self.embed_time_fn(t)))
        deform_in.append(transform_net_in(self.embed_pose_fn(pose)))
        if ldmks is not None:
            deform_in.append(transform_net_in(self.embed_ldmk_fn(ldmks)))

        anchor_uv_view = gs_cam.project_with_offset(lbs_anchor_xyz.detach().clone(), detach_offset=True)


        # warp this back to texture space
        anchor_uv_view = anchor_uv_view.T[None,:,:,None] # [1, 2, N_anchor, 1]
        anchor_deform_in = [einops.rearrange(x, 'B C H W -> B C (H W) 1')[:,:,:self.N_anchor, :] for x in deform_in]

        anchor_uv_base_warp = self.base_warping_fn(anchor_uv_view)
        anchor_uv_base_warp_original = anchor_uv_base_warp # for MLP input

        
        if self.mlp_warp:
            anchor_deform_in = torch.concat([self.uv_embed_fn(anchor_uv_base_warp_original), *anchor_deform_in], axis=1)
            delta_uv = torch.tanh(self.deform_net(anchor_deform_in))
            anchor_uv_texture = delta_uv + anchor_uv_base_warp
        else:
            anchor_uv_texture = anchor_uv_base_warp
        

        ### update anchor correspondences ###
        self.anchor_uv_texture_target.data[self.anchor_valid_mask] = anchor_uv_texture.reshape([2, -1]).T.clone()



    @torch.no_grad()
    def prune_outside_anchor(
        self,
        lbs_anchor_xyz, 
        gs_cam, 
    ):
        '''
        '''

        anchor_uv_view = gs_cam.project_with_offset(lbs_anchor_xyz.detach().clone(), detach_offset=True)
        anchor_uv_view_mask = ((anchor_uv_view >= 0.) & (anchor_uv_view <= 1.)).all(dim=1)
        self.anchor_valid_mask[self.anchor_valid_mask.clone()] = anchor_uv_view_mask
        self.N_anchor = torch.count_nonzero(self.anchor_valid_mask)

        self.anchor_xyz.data[~self.anchor_valid_mask] = torch.nan



    def forward(
            self, 
            lbs_anchor_xyz, 
            gs_cam, 
            t, 
            exp, 
            pose, 
            ldmks=None, 
            ldmk_drawing=None, 
            layer_offset=None, 
            frame_noise=None, 
            iter=None, 
            anchor_visibility_mask=None,
            mlp_delta_uv_overwrite=None,
            ):
        '''
        t: [B, 1] frame id
        exp: [B, exp_dim] expression latent & flame head pose latent
        pose: [B, 3] camera translation
        ldmks: [B, N_ldmk * 2] 2d landmarks (x,y, confidence)
        ldmk_drawing: [B, H, W, 3] drawing of pose in [0, 1]
        frame_noise: [B, noise_dim] per frame optimizable noises

        '''
        B, _ = t.shape
        assert B == 1

        # fetch embeddings
        transform_net_in = lambda x: x[..., None, None].repeat([1,1,self.img_h, self.img_w])
        deform_in = []
        deform_in.append(transform_net_in(self.embed_exp_fn(exp)))
        deform_in.append(transform_net_in(self.embed_time_fn(t)))
        deform_in.append(transform_net_in(self.embed_pose_fn(pose)))
        if ldmks is not None:
            deform_in.append(transform_net_in(self.embed_ldmk_fn(ldmks)))

            
        if iter is None:
            # assume end of trainig
            iter = torch.inf 


        ret_dict = {}

        if self.deform_type == 'mlp':

            # ### compute reproject loss ###
            if iter > self.deform_2d_warm_up_iter:
                anchor_uv_view = gs_cam.project_with_offset(lbs_anchor_xyz.detach().clone(), detach_offset=True)
                # anchor_uv_view = gs_cam.project_with_offset(lbs_anchor_xyz, detach_offset=True)

                # warp this back to texture space
                anchor_uv_view = anchor_uv_view.T[None,:,:,None] # [1, 2, N_anchor, 1]
                anchor_deform_in = [einops.rearrange(x, 'B C H W -> B C (H W) 1')[:,:,:self.N_anchor, :] for x in deform_in]

                anchor_uv_base_warp = self.base_warping_fn(anchor_uv_view)
                anchor_uv_base_warp_original = anchor_uv_base_warp # for MLP input

                if self.apply_affine:
                    
                    affine = skimage.transform.estimate_transform(
                        self.affine_type, 
                        anchor_uv_base_warp.reshape([2,-1]).T.detach().cpu().numpy(),
                        self.anchor_uv_texture_target[self.anchor_valid_mask].detach().cpu().numpy(), 
                        )
                    

                    affine = np.array(affine) #[:2,:]

                    if affine is not None:
                        affine = torch.tensor(affine).type_as(anchor_uv_base_warp)
                        # view uv -> texture uv affine

                        anchor_uv_base_warp_shape = anchor_uv_base_warp.shape
                        anchor_uv_base_warp_ = anchor_uv_base_warp.reshape([2, -1])
                        anchor_uv_base_warp_ = torch.concat([anchor_uv_base_warp_, torch.ones_like(anchor_uv_base_warp_[:1,:])], axis=0)
                        anchor_uv_base_warp_ = (affine @ anchor_uv_base_warp_)
                        anchor_uv_base_warp = (anchor_uv_base_warp_[:2,:] / anchor_uv_base_warp_[2:,:]).reshape(anchor_uv_base_warp_shape)
                else:
                    affine = None

                
                if self.mlp_warp:
                    anchor_deform_in = torch.concat([self.uv_embed_fn(anchor_uv_base_warp_original), *anchor_deform_in], axis=1)
                    delta_uv = torch.tanh(self.deform_net(anchor_deform_in))
                    anchor_uv_texture = delta_uv + anchor_uv_base_warp
                else:
                    anchor_uv_texture = anchor_uv_base_warp

                if self.apply_test_time_affine:

                    if self.affine_type == 'translation':
                        affine_test = torch.eye(3).type_as(anchor_uv_texture)

                        target = self.anchor_uv_texture_target[self.anchor_valid_mask]
                        source = anchor_uv_texture.reshape([2,-1]).T

                        affine_test[:2,2] =  (target - source).mean(dim=0,keepdim=True)

                    else:
                    
                        affine_test_trans = skimage.transform.estimate_transform(
                            self.affine_type, 
                            anchor_uv_texture.reshape([2,-1]).T.detach().cpu().numpy(), 
                            self.anchor_uv_texture_target[self.anchor_valid_mask].detach().cpu().numpy(), 
                            )
                        

                        affine_test = np.array(affine_test_trans) # [:2,:]

                        
                        affine_test = torch.tensor(affine_test).type_as(anchor_uv_texture)

                    # view uv -> texture uv affine
                    anchor_uv_texture_shape = anchor_uv_texture.shape
                    anchor_uv_texture_ = anchor_uv_texture.reshape([2, -1])
                    anchor_uv_texture_ = torch.concat([anchor_uv_texture_, torch.ones_like(anchor_uv_texture_[:1,:])], axis=0)
                    anchor_uv_texture_ = (affine_test @ anchor_uv_texture_)
                    anchor_uv_texture = (anchor_uv_texture_[:2,:] / anchor_uv_texture_[2:,:]).reshape(anchor_uv_texture_shape)

                if layer_offset is not None:
                    if layer_offset.shape[1] == 8:
                        anchor_uv_texture = perspective_2d(anchor_uv_texture, layer_offset)
                    else:
                        # first rotate, then translate
                        rot_rad = layer_offset[:, 2:3]
                        if layer_offset.shape[1] == 4:
                            shear = layer_offset[:,3:4]
                        else:
                            shear = None
                        anchor_uv_texture = rotate_coord_2d(anchor_uv_texture, angle=torch.rad2deg(rot_rad), shear=shear)

                        anchor_uv_texture = layer_offset[:, :2, None, None] + anchor_uv_texture

                anchor_uv_texture = anchor_uv_texture.reshape([2, -1]).T

                deform_loss = ((self.anchor_uv_texture_target[self.anchor_valid_mask] - anchor_uv_texture)**2)
                if anchor_visibility_mask is not None:
                    deform_loss = deform_loss[anchor_visibility_mask]
                deform_loss = deform_loss.mean()


                ret_dict['anchor_deform_loss'] = deform_loss

            if iter <= self.anchor_init_iter:
                # no texture is used when anchor is still being initialized
                textured_img = torch.ones([B, 3, self.img_h, self.img_w]).type_as(lbs_anchor_xyz)
                if self.pred_alpha:
                    textured_img = torch.concat([
                        textured_img, 
                         torch.zeros([B, 1, self.img_h, self.img_w]).type_as(lbs_anchor_xyz)
                        ], axis=1)

            else:
                uv_view = self.uv.clone()
                uv_view = uv_view.repeat([B, 1, 1, 1])

                uv_base_warp = self.base_warping_fn(uv_view)
                uv_base_warp_original = uv_base_warp

                if iter > self.deform_2d_warm_up_iter:

                    if self.apply_affine and affine is not None:
                        uv_base_warp_shape = uv_base_warp.shape
                        uv_base_warp_ = uv_base_warp.reshape([2, -1])
                        uv_base_warp_ = torch.concat([uv_base_warp_, torch.ones_like(uv_base_warp_[:1,:])], axis=0)
                        uv_base_warp_ = (affine @ uv_base_warp_)
                        uv_base_warp = (uv_base_warp_[:2,:] / uv_base_warp_[2:,:]).reshape(uv_base_warp_shape)

                    if self.apply_rigid_deform:
                        raise NotImplementedError()
                        # apply rigid rotation and translation to the image
                        rigid_deform_in = torch.concat(deform_in, axis=1)[..., 0, 0]
                        rot_degree, trans = self.rigid_deform_net_2d(rigid_deform_in)

                        if self.rebase_rigid_rot:
                            rot_degree[:,0] = rot_degree[:,0] - torch.rad2deg(exp[:,2]) # add FLAME global rotation

                        uv = rotate_coord_2d(uv, angle=rot_degree)
                        if self.apply_rigid_translation:
                            uv = uv + trans[...,None,None]

                    # for querying the deform network
                    # we still use the original view space pixel coordinate
                    # without applying the base_warping
                    if self.mlp_warp:
                        deform_in_ = torch.concat([self.uv_embed_fn(uv_base_warp_original), *deform_in], axis=1)
                        
                        if mlp_delta_uv_overwrite is not None:
                            delta_uv = mlp_delta_uv_overwrite
                        else:
                            delta_uv = torch.tanh(self.deform_net(deform_in_))
                            ret_dict['mlp_warp_delta_uv'] = delta_uv

                        uv_texture = delta_uv + uv_base_warp
                        
                        # uv_texture = uv_base_warp

                        # delta_uv_loss = (delta_uv**2).mean()
                        delta_uv_loss = torch.abs(delta_uv).mean()
                        ret_dict['delta_uv_loss'] = delta_uv_loss

                    else:
                        uv_texture = uv_base_warp
                else:
                    uv_texture = uv_base_warp

                if self.apply_test_time_affine:
                    # uv_texture = uv_texture + affine_test[:,2][None, :, None, None]
                    uv_texture_shape = uv_texture.shape
                    uv_texture_ = uv_texture.reshape([2, -1])
                    uv_texture_ = torch.concat([uv_texture_, torch.ones_like(uv_texture_[:1,:])], axis=0)
                    uv_texture_ = (affine_test @ uv_texture_)
                    uv_texture = (uv_texture_[:2,:] / uv_texture_[2:,:]).reshape(uv_texture_shape)

                if layer_offset is not None:
                    if layer_offset.shape[1] == 8:
                        uv_texture = perspective_2d(uv_texture, layer_offset)
                    else:
                        # first rotate, then translate
                        rot_rad = layer_offset[:, 2:3]
                        if layer_offset.shape[1] == 4:
                            shear = layer_offset[:,3:4]
                        else:
                            shear = None
                        uv_texture = rotate_coord_2d(uv_texture, angle=torch.rad2deg(rot_rad), shear=shear)

                        uv_texture = layer_offset[:, :2, None, None] + uv_texture

                features = self.feature_img(uv_texture)

                if not self.sh_texture:
                    if self.use_explict_rgb:
                        corase_rgb = features[:, :3]
                        features = features[:, 3:]
                    else:
                        corase_rgb = 0.

                    if iter >= self.fine_rgb_after_iter and iter < torch.inf:
                        # gather input for color mlp
                        color_in = []
                        color_in.append(transform_net_in(self.embed_exp_fn_color(exp)))
                        color_in.append(transform_net_in(self.embed_time_fn_color(t)))
                        color_in.append(transform_net_in(self.embed_pose_fn_color(pose)))
                        if ldmks is not None:
                            color_in.append(transform_net_in(self.embed_ldmk_fn_color(ldmks)))

                        parser_in = torch.concat([features, *color_in], axis=1)
                        fine_rgb = torch.tanh(self.parser_net(parser_in))

                        if self.reg_fine_color_ratio > 0.:
                            # sample grid
                            reg_features = self.feature_img.feature_img.reshape([self.feature_img.feature_dim, -1])
                            choices = torch.randperm(reg_features.shape[1])[:int(self.reg_fine_color_ratio*reg_features.shape[1])]
                            reg_features = reg_features[:, choices]

                            if self.use_explict_rgb:
                                reg_features = reg_features[3:]

                            parser_in = torch.concat([
                                reg_features[None,:,:,None], 
                                *[einops.rearrange(x, 'B C H W -> B C (H W) 1')[:,:,:len(choices), :] for x in color_in]
                                ], axis=1)
                            
                            fine_reg_rgb = torch.tanh(self.parser_net(parser_in))

                            ret_dict['fine_rgb_reg_loss'] = torch.abs(fine_reg_rgb).mean()




                    else:
                        fine_rgb = torch.zeros_like(corase_rgb)
                else:
                    if iter != 0 and iter % self.sh_texture_up_degree_every == 0:
                        self.sh_activate_degree = min(self.sh_activate_degree + 1, self.sh_max_degree) 

                    # parse features as SH
                    sh_features = features.permute([0,2,3,1])
                    sh_features = sh_features.reshape([B * sh_features.shape[1]* sh_features.shape[2], 3, 16])
                    # sh_features = sh_features.permute([0,3,4,2,1])
                    # use global rotation as direction
                    head_rot = exp[:,:3]
                    head_mat = euler_angles_to_matrix(head_rot, "XYZ")
                    dir = torch.tensor([[0, 0, -1]]).type_as(head_mat).T
                    dir = (head_mat.inverse() @ dir)[...,0]

                    corase_rgb = eval_sh(self.sh_activate_degree, sh_features, dir.repeat(sh_features.shape[0], 1))
                    corase_rgb = torch.clamp_min(corase_rgb + 0.5, 0.)
                    corase_rgb = corase_rgb.reshape([B, features.shape[-2], features.shape[-1], 3]).permute([0, 3, 1, 2])
                    fine_rgb = torch.zeros_like(corase_rgb)




                ret_dict['coarse_rgb'] = corase_rgb
                ret_dict['fine_rgb'] = fine_rgb

                textured_img = corase_rgb[:,:3,...] + fine_rgb[:,:3,...]
                if self.pred_alpha:
                    alpha = fine_rgb[:,3:,...]

                    textured_img = torch.concat([textured_img, alpha], axis=1)



            if iter >= self.anchor_init_iter:
                    anchor_scaling = torch.clamp(
                        self.scaling_activation(
                            self.anchor_scaling if self.anchor_uniform_scale else self.anchor_scaling[self.anchor_valid_mask]
                            ), self.scaling_interval[0])
                    
                    ret_dict['anchor_scale_loss'] = torch.abs(anchor_scaling).mean()


                    opacity_min = max(self.opacity_min, 1e-6)
                    opacity_min_raw = np.log(opacity_min/(1.-opacity_min))
                    anchor_opacity = torch.clamp(
                        self.anchor_opacity[self.anchor_valid_mask], 
                        opacity_min_raw
                        )
                    ret_dict['anchor_opacity_loss'] = (anchor_opacity - opacity_min_raw).mean() # to make loss > 0


        return textured_img, ret_dict
    
    def get_anchor_data(self, lbs_anchor_xyz, iter):
        '''
        Get xyz, rgb, scale, rot etc for anchor gaussians
        '''
        ### Obtain anchor RGB from deformed texture ###
        ### or, obtain RGB from their hard constraint ###

        anchor_rgb = self.anchor_rgb[self.anchor_valid_mask]

        if iter is None:
            iter = 99999999999
        
        anchor_opacity = torch.clamp_min(self.opacity_activation(self.anchor_opacity[self.anchor_valid_mask]), self.opacity_min) # do not allow opaicty to go below certain value 
        
        if self.anchor_uniform_scale:
            anchor_scaling = self.anchor_scaling.repeat([self.N_anchor, 1])
        else:
            anchor_scaling = self.anchor_scaling[self.anchor_valid_mask]

        anchor_scaling = torch.clamp(
            self.scaling_activation(anchor_scaling), self.scaling_interval[0], self.scaling_interval[1])
        anchor_scaling = anchor_scaling.repeat([1,3])
        anchor_rot = torch.concat([torch.ones_like(lbs_anchor_xyz[:,:1]), torch.zeros_like(lbs_anchor_xyz)], dim=-1)

        return lbs_anchor_xyz, anchor_rgb, anchor_opacity, anchor_scaling, anchor_rot


def rotate_coord_2d(uv: torch.Tensor, angle: torch.Tensor, shear=None):
    '''
    Angle: clock-wise rotation in degrees, shape [B, 1]
    '''
    B, _, h, w  = uv.shape
    origin = torch.tensor([1., 1.]).type_as(uv) / 2.

    c = torch.cos(torch.deg2rad(angle[:,0]))
    s = torch.sin(torch.deg2rad(angle[:,0]))
    rotate_mat = torch.stack([torch.stack([c, -s]),
                   torch.stack([s, c])]).permute([2,0,1])
    
    uv_ = torch.bmm(
        rotate_mat, 
        uv.reshape([B, 2, -1]) - origin[None,..., None]
        ) # same as x_rot = (rot @ x.t()).t() due to rot in O(n) (SO(n) even)
    
    if shear is not None:
        shear_mat = torch.eye(2).type_as(rotate_mat)[None, ...].repeat(B, 1, 1)
        shear_mat[:,0,1] = shear
        uv_ = torch.bmm(shear_mat, uv_)

    uv_ = uv_ + origin[None,..., None]
    
    uv_ = uv_.reshape(uv.shape)

    return uv_


def perspective_2d(uv: torch.Tensor, coeffs: torch.Tensor):
    B, _, h, w  = uv.shape

    x = uv[:,0:1]
    y = uv[:,1:2]

    x_ = (coeffs[:,0]*x + coeffs[:,1]*y + coeffs[:,2]) / (coeffs[:,6]*x + coeffs[:,7]*y + 1)
    y_ = (coeffs[:,3]*x + coeffs[:,4]*y + coeffs[:,5]) / (coeffs[:,6]*x + coeffs[:,7]*y + 1)


    uv_ = torch.concat([x_,y_], axis=1)
    return uv_