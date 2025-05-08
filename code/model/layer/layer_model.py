import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp_decoder import MLPDecoder2D
# from .refine_nets.unet import RefineUnetDecoder
from .hash_encoding import MultiResHashGrid
from .stylegan2.networks import StyleGan2Gen
import torchvision
import numpy as np
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

        # TODO: replace with grid_sample?

        return pixel


class DeformLayerNetwork(nn.Module):
    def __init__(
            self, 
            img_h=512, 
            img_w=512, 
            feature_dim=64, 
            apply_deform=False,
            apply_rigid_deform=True, 
            apply_rigid_translation=True,
            rebase_rigid_rot=True, # base rigid rotation on global rotation of FLAME model
            parser_type='mlp', 
            deform_type='mlp', 
            layer_parser_rescale=1, 
            out_channel=3,
            exp_dim=64,
            exp_multires=-2, # condition on flame parameters
            t_multires=-2, # condition on frame id
            pose_multires=4, # condition on camera translation (rot is the same)
            n_ldmk=4,
            ldmk_multires=-2, # condition on openpose/dwpose body landmarks
            layer_encoding='fourier',
            feature_img_as_coarse=True, # use :3 of feature img as coarse img
            shrink_uv=1., # shrink the coordinate queries from [0, 1] to [shrink_uv/2, 1-shrink_uv/2]
            parser_args={},
            deformer_args={},
            layer_encoder_args={},
            use_per_frame_noise=False,
            frame_noise_dim=16,
            ):
        super(DeformLayerNetwork, self).__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.feature_dim = feature_dim
        self.embed_time_fn, time_input_ch = get_embedder(t_multires, 1)
        self.embed_exp_fn, exp_input_ch = get_embedder(exp_multires, exp_dim)
        self.embed_pose_fn, pose_input_ch = get_embedder(pose_multires, 3) # only take camera translation, as rotation is same for all frames
        self.embed_ldmk_fn, ldmk_input_ch = get_embedder(ldmk_multires, n_ldmk * 2)
        self.apply_deform = apply_deform
        self.layer_encoding = layer_encoding
        self.feature_img_as_coarse = feature_img_as_coarse
        self.apply_rigid_deform = apply_rigid_deform
        self.apply_rigid_translation = apply_rigid_translation
        self.rebase_rigid_rot = rebase_rigid_rot
        self.deform_type = deform_type
        self.use_per_frame_noise = use_per_frame_noise
        self.frame_noise_dim = frame_noise_dim

        # [1, 2, H, W]
        self.uv = torch.stack(torch.meshgrid(torch.arange(img_h), torch.arange(img_w))).cuda()[None,...] / max(img_h, img_w) 
        self.uv = (self.uv - 0.5) * shrink_uv + 0.5 # leave some space in the edge

        self.uv_embed_fn, uv_embed_ch = get_embedder(feature_dim, 2) # abuse feature_dim for PE frequency
        if layer_encoding == 'fourier':
            feature_dim = uv_embed_ch
            self.feature_dim = uv_embed_ch
            self.feature_img_fn = self.uv_embed_fn
        elif layer_encoding == 'hash':
            self.feature_img_fn = MultiResHashGrid(2, n_features_per_level=feature_dim // layer_encoder_args.get('n_levels', 16), **layer_encoder_args)
        elif layer_encoding == 'feature':
            # self.feature_img = nn.Parameter(torch.rand([img_h, img_w, feature_dim], device='cuda').requires_grad_(True))
            self.feature_img_fn = FeatureImage(img_h, img_w, feature_dim=feature_dim, **layer_encoder_args)
        else:
            raise NotImplementedError(layer_encoding)

        if apply_deform:
            if deform_type == 'mlp':
                if self.apply_rigid_deform:
                    self.rigid_deform_net_2d = RigidDeformNet2D(
                        D=8,
                        W=256,
                        skips=[4],
                        input_ch=time_input_ch + exp_input_ch + pose_input_ch + ldmk_input_ch,
                        # out_rescale=0.01,
                        init_zero_output=True,
                    )
                self.deform_net_2d = MLPDecoder2D(
                    D=4,
                    W=128,
                    input_ch=uv_embed_ch + time_input_ch + exp_input_ch + pose_input_ch + ldmk_input_ch,
                    output_ch=2,
                    skips=[],
                    out_rescale=0.01,
                    # init_zero_output=True,
                    act_fn=None,
                )
                
            else:
                raise NotImplementedError()

        self.deform_warm_up = True

        # input_channel = uv_embed_ch + feature_dim + time_input_ch + exp_input_ch + pose_input_ch + ldmk_input_ch
        input_channel = feature_dim + time_input_ch + exp_input_ch + pose_input_ch + ldmk_input_ch

        if layer_encoding != 'fourier' and feature_img_as_coarse:
            # apply coarse fine split, first three digit of feature img is corase RGB
            input_channel = input_channel - 3

        self.parser_type = parser_type
        if parser_type == 'mlp':
            self.network = MLPDecoder2D(
                input_ch=input_channel, 
                out_rescale=layer_parser_rescale,
                output_ch=out_channel,
                act_fn=torch.sigmoid,
                **parser_args,
                )

        elif parser_type == 'none':
            # directly take feature img as RGB without going through MLP
            self.network = lambda feature_img: feature_img[:,:3,...]
        else:
            raise NotImplementedError(f'refine_parser_type {parser_type} not supported')


    def forward(self, t, exp, pose, ldmks=None, ldmk_drawing=None, layer_offset=None, frame_noise=None):
        '''
        t: [B, 1] frame id
        exp: [B, exp_dim] expression latent & flame head pose latent
        pose: [B, 3] camera translation
        ldmks: [B, N_ldmk, 2] 2d landmarks (x,y, confidence)
        ldmk_drawing: [B, H, W, 3] drawing of pose in [0, 1]
        frame_noise: [B, noise_dim] per frame optimizable noises

        returns: img in [B, out_channel, H, W], and another img with coarse_img detached (for VGG loss)
        '''
        B = t.shape[0]
        # fetch embeddings
        # transform = lambda x: x.T.unsqueeze(1).repeat([1, self.img_h, self.img_w])
        transform = lambda x: x[..., None, None].repeat([1,1,self.img_h, self.img_w])
        net_in = []
        net_in.append(transform(self.embed_exp_fn(exp)))
        net_in.append(transform(self.embed_time_fn(t)))
        net_in.append(transform(self.embed_pose_fn(pose)))
        if ldmks is not None:
            # net_in.append(transform(self.embed_ldmk_fn(ldmks[...,:2].reshape(B, -1))))
            net_in.append(transform(self.embed_ldmk_fn(ldmks)))

        ret_dict = {}

        if self.parser_type == 'stylegan2':
            raise NotImplementedError
            net_in = torch.concat(net_in, axis=1)[:,:,0,0]

            img = self.network(net_in, noise_mode='none')
        elif self.parser_type == 'draw_unet' or self.parser_type == 'draw_swin':
            net_in = ldmk_drawing.permute([0,3,1,2]) * 2. - 1.
            _, _, draw_h, draw_w = net_in.shape

            # deform_uv = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w))).cuda()[None,...] / max(h, w) 
            # deform_uv = deform_uv.repeat([B, 1, 1, 1])
            # net_in = torch.concat([self.uv_embed_fn(deform_uv), net_in], axis=1)
            
            if self.use_per_frame_noise:
                if frame_noise is None:
                    frame_noise = torch.zeros([B, self.frame_noise_dim]).type_as(net_in)
                
                net_in = torch.concat([frame_noise[...,None,None].repeat([1,1,draw_h,draw_w]), net_in], axis=1)


            img = self.network(net_in)
            img = torch.sigmoid(img)

        else:
            uv = self.uv.clone()
            uv = uv.repeat([B, 1, 1, 1])

            if self.apply_deform and not self.deform_warm_up:
                if self.deform_type == 'mlp':
                    if self.apply_rigid_deform:
                        # apply rigid rotation and translation to the image
                        rigid_deform_in = torch.concat(net_in, axis=1)[..., 0, 0]
                        rot_degree, trans = self.rigid_deform_net_2d(rigid_deform_in)

                        if self.rebase_rigid_rot:
                            rot_degree[:,0] = rot_degree[:,0] - torch.rad2deg(exp[:,2]) # add FLAME global rotation

                        uv = rotate_coord_2d(uv, angle=rot_degree)
                        if self.apply_rigid_translation:
                            uv = uv + trans[...,None,None]

                    deform_in = torch.concat([self.uv_embed_fn(uv), *net_in], axis=1)

                elif self.deform_type == 'draw_unet' or self.deform_type == 'draw_swin':
                    deform_in = ldmk_drawing.permute([0,3,1,2]) #* 2. - 1.
                    # _, _, h, w = deform_in.shape
                    # deform_uv = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w))).cuda()[None,...] / max(h, w) 
                    # deform_uv = deform_uv.repeat([B, 1, 1, 1])
                    # deform_in = torch.concat([self.uv_embed_fn(deform_uv), deform_in], axis=1)

                delta_uv = torch.tanh(self.deform_net_2d(deform_in))
                uv = delta_uv + uv


            if layer_offset is not None:
                # first rotate, then translate
                rot_rad = layer_offset[:, 2:]
                uv = rotate_coord_2d(uv, angle=torch.rad2deg(rot_rad))

                uv = layer_offset[:, :2, None, None] + uv

            if self.layer_encoding != 'feature':
                uv = torch.clamp(uv, 0., 1.)

            feature_img = self.feature_img_fn(uv)

            # coarse_img = 0.
            if self.layer_encoding == 'fourier' or not self.feature_img_as_coarse:
                coarse_img = torch.zeros_like(feature_img[:,:3,...])
            else:
                coarse_img = feature_img[:,:3,...]
                feature_img = feature_img[:,3:,...]

            # torchvision.utils.save_image(feature_img, 'test.png')
            # ###
            # import imageio
            # gt_im = imageio.imread('/home/tw554/pointavatar_gs/data/datasets/walter_all/walter_all/all/image/1.png')
            # gt_im = torch.from_numpy(gt_im / 255.).cuda()
            # padding = self.feature_img_fn.padding
            # self.feature_img_fn.feature_img.data[:, padding:padding+self.img_h, padding:padding+self.img_w] = gt_im.movedim(-1, 0)
            # ###

            # # additional add original pixel coordinate for infering pose dependent appearance changes
            # original_uv = self.uv.clone().repeat([B, 1, 1, 1])
            # net_in = torch.concat([self.uv_embed_fn(original_uv), feature_img, *net_in], axis=1)
            net_in = torch.concat([feature_img, *net_in], axis=1)

            # if self.parser_type == 'pix2pix':
            #     net_in = self.pre_network(net_in)

            if self.parser_type == 'none':
                fine_img = 0.
            else:
                fine_img = self.network(net_in)

            img = fine_img + coarse_img

            ret_dict.update({
                'fine': fine_img,
                'coarse': coarse_img,
            })

        return img, ret_dict



def rotate_coord_2d(uv: torch.Tensor, angle: torch.Tensor):
    '''
    Angle: clock-wise rotation in degrees, shape [B, 1]
    '''
    B, _, h, w  = uv.shape
    origin = torch.tensor([1., 1.]).type_as(uv) / 2.

    c = torch.cos(torch.deg2rad(angle[:,0]))
    s = torch.sin(torch.deg2rad(angle[:,0]))
    rotate_mat = torch.stack([torch.stack([c, -s]),
                   torch.stack([s, c])]).permute([2,0,1])
    
    uv_rot = torch.bmm(
        rotate_mat, 
        uv.reshape([B, 2, -1]) - origin[None,..., None]
        ) + origin[None,..., None] # same as x_rot = (rot @ x.t()).t() due to rot in O(n) (SO(n) even)
    
    uv_rot = uv_rot.reshape(uv.shape)

    return uv_rot
