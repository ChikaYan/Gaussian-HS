import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gaussian.utils.general_utils import get_expon_lr_func
from .pix2pix import Pix2PixDecoder
# from .refine_nets.unet import RefineUnetDecoder
from .hash_encoding import MultiResHashGrid
from .stylegan2.networks import StyleGan2Gen
import torchvision
import numpy as np
import einops
# from .unet.openaimodel import UNetModel



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

    def forward(self, yx):
        '''
        Bilinearly interpolate an img

        yx: [N, 2] pixel coordinates (height, width)
        img: [C, H, W] feature image

        '''
        img = self.feature_img
        # C, img_h, img_w = img.shape

        img_h, img_w = self.img_h, self.img_w

        # convert from [0,1] to pixel coordinate
        yx = yx * torch.tensor([img_h, img_w]).type_as(yx) + self.padding #[None,:, None, None] 

        yx = torch.stack(
            [torch.clamp(yx[:,0], 0, img_h- 1), 
            torch.clamp(yx[:,1], 0, img_w- 1)], dim=1)

        x_int = torch.floor(yx[:,1])
        y_int = torch.floor(yx[:,0])

        # Prevent crossing
        x_int = torch.clamp_max(x_int, img_h-2).long()
        y_int = torch.clamp_max(y_int, img_w-2).long()

        x_diff = (yx[:,1] - x_int)[None, ...] #[:,None,...]
        y_diff = (yx[:,0] - y_int)[None, ...]#[:, None,...]

        if (y_int < 0).any() or (x_int < 0).any() or (y_int + 1 >= img.shape[1]).any() or (x_int + 1 >= img.shape[2]).any():
            print('hit!')

        a = img[:, y_int.reshape(-1), x_int.reshape(-1)]
        b = img[:, y_int.reshape(-1), x_int.reshape(-1)+1]
        c = img[:, y_int.reshape(-1)+1, x_int.reshape(-1)]
        d = img[:, y_int.reshape(-1)+1, x_int.reshape(-1)+1]

        features = a*(1-x_diff)*(1-y_diff) + b*(x_diff) * \
            (1-y_diff) + c*(1-x_diff) * (y_diff) + d*x_diff*y_diff
        
        return features.T
    
    @torch.no_grad()
    def init_feature(self, img):
        h, w, c = img.shape
        assert h == self.img_h, w == self.img_w

        img = img.permute([2,0,1])
        padding = self.padding
        self.feature_img.data[:c, padding:-padding, padding:-padding] = img



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

class LiftDeformNetwork(nn.Module):
    def __init__(
            self, 
            deform_type='mlp',
            apply_rigid_deform=False,
            exp_dim=6,
            xyz_multires=4,
            exp_multires=2, # condition on flame parameters (head pose mainly)
            t_multires=-2, # condition on frame id
            pose_multires=2, # condition on camera translation (rot is the same)
            ldmk_multires=10, # condition on openpose/dwpose body landmarks
            n_ldmk=4, # number of 2D ldmks to condition on
            pred_rgb_shift=False,
            deform_3d=True, # deform in 3D or 2D
            feature_dim=32,
            use_explict_rgb=False,
            deform_2d_warmup_iter=5000,
            ):
        super(LiftDeformNetwork, self).__init__()
        self.embed_xyz_fn, xyz_input_ch = get_embedder(xyz_multires, 3)
        self.embed_time_fn, time_input_ch = get_embedder(t_multires, 1)
        self.embed_exp_fn, exp_input_ch = get_embedder(exp_multires, exp_dim)
        self.embed_pose_fn, pose_input_ch = get_embedder(pose_multires, 3) # only take camera translation, as rotation is same for all frames
        self.embed_ldmk_fn, ldmk_input_ch = get_embedder(ldmk_multires, n_ldmk * 2)
        self.apply_rigid_deform = apply_rigid_deform
        self.deform_type = deform_type
        self.pred_rgb_shift = pred_rgb_shift
        self.deform_3d = deform_3d
        self.use_explict_rgb = use_explict_rgb
        input_feature_dim = feature_dim # for deform_net input
        if use_explict_rgb:
            feature_dim += 3
        self.feature_dim = feature_dim
        self.deform_2d_warmup_iter = deform_2d_warmup_iter



        deform_out_ch = 3 if deform_3d else 2
        if pred_rgb_shift:
            deform_out_ch += 3   


        if deform_type == 'mlp':
            if self.apply_rigid_deform:
                raise NotImplementedError
                self.rigid_deform_net_2d = RigidDeformNet2D(
                    D=8,
                    W=256,
                    skips=[4],
                    input_ch=time_input_ch + exp_input_ch + pose_input_ch + ldmk_input_ch,
                    # out_rescale=0.01,
                    init_zero_output=True,
                )
            self.deform_net = MLPDecoder(
                input_ch=xyz_input_ch + time_input_ch + exp_input_ch + pose_input_ch + ldmk_input_ch,
                output_ch=deform_out_ch,
                D=8,
                W=128,
                skips=[4],
                # out_rescale=0.01,
                init_zero_output=True,
                act_fn=None,
            )
        elif deform_type == 'draw_unet':
            # use a unet to convert pose drawing to 2D deformation
            # raise NotImplementedError
            assert not use_explict_rgb

            from .unet.unet import ResUnet
            self.feature_net = ResUnet(
                in_channels=3,
                out_channels=feature_dim,
                # out_rescale=1e-2,
                init_zero_output=False,

                down_channels = [64, 128, 256], # /2, /4, /4
                down_attention = [False, True, True],
                mid_attention = True,
                up_channels = [256, 128, 64, 32, 32], # /2, /1, *2, *4, *4
                up_attention = [True, True, False, False, False],
                # **deformer_args,
            )
            self.deform_net = MLPDecoder(
                input_ch=xyz_input_ch + time_input_ch + exp_input_ch + pose_input_ch + ldmk_input_ch + input_feature_dim,
                output_ch=deform_out_ch,
                D=8,
                W=128,
                skips=[4],
                # out_rescale=0.01,
                init_zero_output=True,
                act_fn=None,
            )
        elif deform_type == 'feature':
            # map gaussians to optimizable feature imgs
            img_h = img_w = 512

            layer_encoder_args = {
                'padding': 100
            }
            self.feature_net = FeatureImage(img_h, img_w, feature_dim=feature_dim, **layer_encoder_args)

            if use_explict_rgb:
                self.feature_net.feature_img[:3,...] = 0.

            TWOD_PIXEL_MULTIRES = 32
            self.uv_embed_fn, uv_embed_ch = get_embedder(TWOD_PIXEL_MULTIRES, 2)

            self.deform_net_2d = MLPDecoder(
                D=4,
                W=128,
                input_ch=uv_embed_ch + time_input_ch + exp_input_ch + pose_input_ch + ldmk_input_ch,
                output_ch=2,
                skips=[],
                out_rescale=1e-5,
                init_zero_output=True,
                act_fn=None,
            )

            self.deform_net = MLPDecoder(
                input_ch=xyz_input_ch + time_input_ch + exp_input_ch + pose_input_ch + ldmk_input_ch + input_feature_dim,
                output_ch=deform_out_ch,
                # D=8,
                # W=128,
                # skips=[4],
                D=4,
                W=64,
                skips=[],
                # out_rescale=0.01,
                init_zero_output=True,
                act_fn=None,
            )
        else:
            raise NotImplementedError()

        self.deform_warm_up = True


    def forward(self, xyz, gs_cam, t, exp, pose, ldmks=None, ldmk_drawing=None, layer_offset=None, frame_noise=None, iter=None):
        '''
        xyz: [B, N_pt, 3] Gaussian centers
        t: [B, 1] frame id
        exp: [B, exp_dim] expression latent & flame head pose latent
        pose: [B, 3] camera translation
        ldmks: [B, N_ldmk * 2] 2d landmarks (x,y, confidence)
        ldmk_drawing: [B, H, W, 3] drawing of pose in [0, 1]
        frame_noise: [B, noise_dim] per frame optimizable noises

        returns: [B, N_pt, 3] gaussian centers shifts, [B, N_pt, 3] gaussian rgb shifts (0 if not enabled)
        '''
        B, N, _ = xyz.shape
        assert B == 1, "currently only support single batch"
        # fetch embeddings
        transform = lambda x: x[:, None, :].repeat([1,N,1]).reshape([B*N, -1])
        net_in = []
        xyz = einops.rearrange(xyz, 'B N C -> (B N) C')
        xyz_embed = self.embed_xyz_fn(xyz)
        net_in.append(xyz_embed)
        net_in.append(transform(self.embed_exp_fn(exp)))
        net_in.append(transform(self.embed_time_fn(t)))
        net_in.append(transform(self.embed_pose_fn(pose)))
        if ldmks is not None:
            net_in.append(transform(self.embed_ldmk_fn(ldmks)))

        ret_dict = {}

        coarse_rgb = 0.

        if self.deform_type == 'mlp':
            if self.apply_rigid_deform:
                raise NotImplementedError()
                # apply rigid rotation and translation to the image
                rigid_deform_in = torch.concat(net_in, axis=1)[..., 0, 0]
                rot_degree, trans = self.rigid_deform_net_2d(rigid_deform_in)

                if self.rebase_rigid_rot:
                    rot_degree[:,0] = rot_degree[:,0] - torch.rad2deg(exp[:,2]) # add FLAME global rotation

                uv = rotate_coord_2d(uv, angle=rot_degree)
                if self.apply_rigid_translation:
                    uv = uv + trans[...,None,None]

            net_in = torch.concat([*net_in], axis=-1)

        elif self.deform_type == 'draw_unet':
            # obtain 2D feature images
            deform_in = ldmk_drawing.permute([0,3,1,2]) * 2. - 1.
            feature_img = self.feature_net(deform_in)

            ###### Project 3D gaussian centers to 2D image planes ######
            # xyz[:,2] = -xyz[:,2] 
            mean3d = torch.cat([xyz.detach().clone(), torch.ones_like(xyz[:, :1])], dim=1) 
            mean3d = mean3d.t().clone().detach()
            xy = gs_cam.full_proj_transform.t().detach().clone() @ mean3d
            xy = xy/xy[3]
            xy = torch.nan_to_num(xy, 0, 0, 0)
            xy = xy[:2].T #[N, 2] pixel coordinate of the gaussians

            IMG_SIZE = feature_img.shape[2]

            xy = ((xy + 1.0) * IMG_SIZE - 1.0) * 0.5 / IMG_SIZE

            yx = torch.flip(xy, dims=[1]) # [W coord, H coord to H coord W coord]

            ############################################################
            
            # take feature
            features = biliner(yx, feature_img[0]) # [N, feature_dim]

            net_in = torch.concat([features, *net_in], axis=-1)
        elif self.deform_type == 'feature':
            ###### Project 3D gaussian centers to 2D image planes ######
            # xyz[:,2] = -xyz[:,2] 
            mean3d = torch.cat([xyz.detach().clone(), torch.ones_like(xyz[:, :1])], dim=1) 
            mean3d = mean3d.t()
            xy = gs_cam.full_proj_transform.t().detach().clone() @ mean3d
            xy = xy/xy[3]
            xy = torch.nan_to_num(xy, 0, 0, 0)
            xy = xy[:2].T #[N, 2] pixel coordinate of the gaussians

            IMG_SIZE = self.feature_net.img_h
            xy = ((xy + 1.0) * IMG_SIZE - 1.0) * 0.5 / IMG_SIZE

            yx = torch.flip(xy, dims=[1]) # [W coord, H coord to H coord W coord]

            ############################################################


            # if iter is None or iter > self.deform_2d_warmup_iter:
            if iter is not None and iter > self.deform_2d_warmup_iter:
                # deform yx in 2D
                deform_2d_in = torch.concat([self.uv_embed_fn(yx), *net_in[1:]], axis=1)
                delta_yx = torch.tanh(self.deform_net_2d(deform_2d_in))
                yx = yx + delta_yx

            # fetch feature img
            features = self.feature_net(yx)

            if self.use_explict_rgb:
                coarse_rgb = features[:, :3]
                features = features[:,3:]

            net_in = torch.concat([features, *net_in], axis=-1)

        else:
            raise NotImplementedError()

        # decode feature
        net_out = self.deform_net(net_in)

        if self.pred_rgb_shift:
            xyz_shift = net_out[:, 3:]

            if self.use_explict_rgb:
                fine_rgb = net_out[:, :3]
                color_shift = fine_rgb + coarse_rgb
            else:
                color_shift = net_out[:, :3]

            color_shift = torch.tanh(color_shift)

        else:
            xyz_shift = net_out
            color_shift = torch.zeros([B*N, 3]).type_as(xyz_shift)

        if not self.deform_3d:
            # add 0 shift to z
            xyz_shift = torch.concat([xyz_shift, torch.zeros_like(xyz_shift[:,:1])], axis=-1)
                

        xyz_shift = xyz_shift.reshape([B, N, 3])
        color_shift = color_shift.reshape([B, N, 3])

        # # if iter is not None and iter > self.deform_2d_warmup_iter:
        xyz_shift = torch.zeros_like(xyz_shift)

        return xyz_shift, color_shift


def biliner(yx, img):
    '''
    Bilinearly interpolate an img

    yx: [N, 2] pixel coordinates (height, width)
    img: [C, H, W] feature image

    '''
    C, img_h, img_w = img.shape

    # convert from [0,1] to pixel coordinate
    yx = yx * torch.tensor([img_h, img_w]).type_as(yx)#[None,:, None, None]

    yx = torch.stack(
        [torch.clamp(yx[:,0], 0, img_h- 1), 
        torch.clamp(yx[:,1], 0, img_w- 1)], dim=1)

    x_int = torch.floor(yx[:,1])
    y_int = torch.floor(yx[:,0])

    # Prevent crossing
    x_int = torch.clamp_max(x_int, img_h-2).long()
    y_int = torch.clamp_max(y_int, img_w-2).long()

    x_diff = (yx[:,1] - x_int)[None, ...] #[:,None,...]
    y_diff = (yx[:,0] - y_int)[None, ...]#[:, None,...]


    a = img[:, y_int.reshape(-1), x_int.reshape(-1)]
    b = img[:, y_int.reshape(-1), x_int.reshape(-1)+1]
    c = img[:, y_int.reshape(-1)+1, x_int.reshape(-1)]
    d = img[:, y_int.reshape(-1)+1, x_int.reshape(-1)+1]

    features = a*(1-x_diff)*(1-y_diff) + b*(x_diff) * \
        (1-y_diff) + c*(1-x_diff) * (y_diff) + d*x_diff*y_diff
    
    return features.T