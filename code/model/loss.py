import torch
from torch import nn
from model.vgg_feature import VGGPerceptualLoss
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)



class Loss(nn.Module):
    def __init__(
            self, 
            mask_weight=0.0,
            var_expression=None, 
            lbs_weight=0,
            sdf_consistency_weight=0, 
            eikonal_weight=0, 
            vgg_feature_weight=0, 
            vgg_detach_bg_weight=0, # VGG loss applied to image where corase layer model is detached
            vgg_detach_coarse_layer_weight=0, # VGG loss applied to image where corase layer model is detached
            head_mask_weight=0., 
            layer_alpha_weight = 0., # encourage bg to be white if occluded by Gaussian
            rgb_l2_weight=0., 
            rgb_l1_weight=1.,
            rgb_l1_detach_bg_weight=0.,
            rgb_dssim_weight=0.,
            bg_layer_mask_loss_weight=0.,
            anchor_deform_loss_weight=1.,
            anchor_scale_loss_weight=0.,
            anchor_opacity_loss_weight=0.,
            texture_neural_uv_delta_loss_weight=0.,
            fine_rgb_reg_loss_weight=0.,
            **kwargs,
            ):
        super().__init__()
        self.mask_weight = mask_weight
        self.lbs_weight = lbs_weight
        self.sdf_consistency_weight = sdf_consistency_weight
        self.eikonal_weight = eikonal_weight
        self.vgg_feature_weight = vgg_feature_weight
        self.vgg_detach_bg_weight = vgg_detach_bg_weight
        self.vgg_detach_coarse_layer_weight = vgg_detach_coarse_layer_weight
        self.var_expression = var_expression
        self.head_mask_weight = head_mask_weight
        self.layer_alpha_weight = layer_alpha_weight
        self.bg_layer_mask_loss_weight = bg_layer_mask_loss_weight
        self.rgb_l2_weight = rgb_l2_weight
        self.rgb_l1_weight = rgb_l1_weight
        self.rgb_l1_detach_bg_weight = rgb_l1_detach_bg_weight
        self.rgb_dssim_weight = rgb_dssim_weight
        self.anchor_deform_loss_weight = anchor_deform_loss_weight
        self.anchor_scale_loss_weight = anchor_scale_loss_weight
        self.anchor_opacity_loss_weight = anchor_opacity_loss_weight
        self.fine_rgb_reg_loss_weight = fine_rgb_reg_loss_weight
        self.texture_neural_uv_delta_loss_weight = texture_neural_uv_delta_loss_weight
        if self.var_expression is not None:
            self.var_expression = self.var_expression.unsqueeze(1).expand(1, 3, -1).reshape(1, -1).cuda()
        print("Expression variance: ", self.var_expression)

        # if self.vgg_feature_weight > 0 or self.vgg_detach_coarse_layer_weight > 0. or self.vgg_detach_bg_weight > 0.:
        self.get_vgg_loss = VGGPerceptualLoss().cuda()

        self.l1_loss = nn.L1Loss(reduction='mean')
        self.l2_loss = nn.MSELoss(reduction='none')

    def get_rgb_loss(self, rgb_values, rgb_gt, weight=None):
        if weight is not None:
            rgb_loss = self.l1_loss(rgb_values.reshape(-1, 3) * weight.reshape(-1, 1), rgb_gt.reshape(-1, 3) * weight.reshape(-1, 1))
        else:
            rgb_loss = self.l1_loss(rgb_values.reshape(-1, 3), rgb_gt.reshape(-1, 3))
        return rgb_loss
    
    def get_rgb_l2_loss(self, rgb_values, rgb_gt, weight=None):
        if weight is not None:
            rgb_loss = self.l2_loss(rgb_values.reshape(-1, 3) * weight.reshape(-1, 1), rgb_gt.reshape(-1, 3) * weight.reshape(-1, 1))
        else:
            rgb_loss = self.l2_loss(rgb_values.reshape(-1, 3), rgb_gt.reshape(-1, 3))
        return rgb_loss.mean()
    
    def get_dssim_loss(self, pred, gt):
        B = pred.shape[0]
        dssim_losses = []
        for i in range(B):
            dssim_losses.append(1.0 - ssim(pred[i], gt[i]))

        dssim_loss = torch.stack(dssim_losses).mean()
        return dssim_loss

    def get_lbs_loss(self, lbs_weight, gt_lbs_weight, use_var_expression=False):
        # the same function is used for lbs, shapedirs, posedirs.
        if use_var_expression and self.var_expression is not None:
            lbs_loss = torch.mean(self.l2_loss(lbs_weight, gt_lbs_weight) / self.var_expression / 50)
        else:
            lbs_loss = self.l2_loss(lbs_weight, gt_lbs_weight).mean()
        return lbs_loss

    def get_mask_loss(self, predicted_mask, object_mask):
        mask_loss = self.l1_loss(predicted_mask.reshape(-1).float(), object_mask.reshape(-1).float())
        return mask_loss
    
    def get_head_mask_loss(self, predicted_alpha, head_mask):
        '''
        A soft regularization to encourage gaussians to only model head
        Only penalize gaussians that are outside of the mask
        '''
        predicted_alpha = predicted_alpha.reshape(-1).float()
        head_mask = head_mask.reshape(-1).float()
        head_mask_loss = torch.where(
            predicted_alpha > head_mask, 
            (predicted_alpha - head_mask)**2, torch.zeros_like(predicted_alpha)).mean()
        return head_mask_loss
    
    def get_layer_alpha_loss(self, predicted_alpha, bg_layer):
        predicted_alpha = predicted_alpha.reshape(-1, 1).float()
        bg_layer = bg_layer.reshape(-1, 3).float()
        layer_alpha_loss = torch.where(
            predicted_alpha > 0.1, 
            (bg_layer - 1.)**2, torch.zeros_like(bg_layer)).mean()
        return layer_alpha_loss
    
    def get_bg_layer_mask_loss(self, pred_bg, bg_mask):
        pred_bg = pred_bg.reshape(-1, 3).float()
        bg_mask = bg_mask.reshape(-1, 1).float()

        empty_bg_color = 1.
        bg_lasyer_mask = (empty_bg_color - (pred_bg * (1-bg_mask) + bg_mask))**2

        # torchvision.utils.save_image(bg_mask.reshape([1,512,512,1]).permute([0,3,1,2]), 'test.png')
        # torchvision.utils.save_image((pred_bg * (1-bg_mask) + bg_mask).reshape([1,512,512,3]).permute([0,3,1,2]), 'test.png')

        return bg_lasyer_mask.mean()


    def get_gt_blendshape(self, index_batch, flame_lbs_weights, flame_posedirs, flame_shapedirs, ghostbone):
        if ghostbone:
            gt_lbs_weight = torch.zeros(len(index_batch), 6).cuda()
            gt_lbs_weight[:, 1:] = flame_lbs_weights[index_batch, :]
        else:
            gt_lbs_weight = flame_lbs_weights[index_batch, :]

        gt_shapedirs = flame_shapedirs[index_batch, :, 100:]
        gt_posedirs = torch.transpose(flame_posedirs.reshape(36, -1, 3), 0, 1)[index_batch, :, :]

        output = {
            'gt_lbs_weights': gt_lbs_weight,
            'gt_posedirs': gt_posedirs,
            'gt_shapedirs': gt_shapedirs,
        }
        return output

    def get_sdf_consistency_loss(self, sdf_values):
        return torch.mean(sdf_values * sdf_values)

    def get_eikonal_loss(self, grad_theta):
        assert grad_theta.shape[1] == 3
        assert len(grad_theta.shape) == 2
        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def forward(self, model_outputs, ground_truth):
        predicted_mask = model_outputs['predicted_mask']
        object_mask = ground_truth['object_mask']
        mask_loss = self.get_mask_loss(predicted_mask, object_mask)

        rgb_loss = self.get_rgb_loss(model_outputs['rgb_image'], ground_truth['rgb'])
        loss = self.rgb_l1_weight * rgb_loss + self.mask_weight * mask_loss

        out = {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'mask_loss': mask_loss,
        }

        if self.rgb_l1_detach_bg_weight > 0.:
            rgb_l1_detach_bg_loss = self.get_rgb_loss(model_outputs['rgb_image_detach_bg'], ground_truth['rgb'])
            loss += self.rgb_l1_detach_bg_weight * rgb_l1_detach_bg_loss
            out['rgb_l1_detach_bg_loss'] = rgb_l1_detach_bg_loss

        if self.rgb_l2_weight > 0.:
            rgb_l2_loss = self.get_rgb_l2_loss(model_outputs['rgb_image'], ground_truth['rgb'])
            loss += self.rgb_l2_weight * rgb_l2_loss
            out['rgb_l2_loss'] = rgb_l2_loss

        if self.head_mask_weight > 0.:
            gs_alpha = predicted_mask
            if 'alpha_no_anchor' in model_outputs:
                gs_alpha = model_outputs['alpha_no_anchor']
            head_mask_loss = self.get_head_mask_loss(gs_alpha, ground_truth['head_mask'])
            loss += self.head_mask_weight * head_mask_loss
            out['head_mask_loss'] = head_mask_loss

        if self.layer_alpha_weight > 0. and model_outputs.get('bg_layer', None) is not None:
            gs_alpha = predicted_mask
            if 'alpha_no_anchor' in model_outputs:
                gs_alpha = model_outputs['alpha_no_anchor']
            layer_alpha_loss = self.get_layer_alpha_loss(gs_alpha, model_outputs['bg_layer'])
            loss += self.layer_alpha_weight * layer_alpha_loss
            out['layer_alpha_loss'] = layer_alpha_loss

        if self.bg_layer_mask_loss_weight > 0.:
            bg_layer_mask_loss = self.get_bg_layer_mask_loss(model_outputs['bg_layer'], ground_truth['bg_layer_mask'])
            loss += self.bg_layer_mask_loss_weight * bg_layer_mask_loss
            out['bg_layer_mask_loss'] = bg_layer_mask_loss

        if self.vgg_feature_weight > 0:
            bz = model_outputs['batch_size']
            img_res = model_outputs['img_res']
            gt = ground_truth['rgb'].reshape(bz, img_res[0], img_res[1], 3).permute(0, 3, 1, 2)

            predicted = model_outputs['rgb_image'].reshape(bz, img_res[0], img_res[1], 3).permute(0, 3, 1, 2)

            vgg_loss = self.get_vgg_loss(predicted, gt)
            out['vgg_loss'] = vgg_loss
            out['loss'] += vgg_loss * self.vgg_feature_weight

        if self.rgb_dssim_weight > 0:
            bz = model_outputs['batch_size']
            img_res = model_outputs['img_res']
            gt = ground_truth['rgb'].reshape(bz, img_res[0], img_res[1], 3).permute(0, 3, 1, 2)

            predicted = model_outputs['rgb_image'].reshape(bz, img_res[0], img_res[1], 3).permute(0, 3, 1, 2)

            dssim_loss = self.get_dssim_loss(predicted, gt)
            out['dssim_loss'] = dssim_loss
            out['loss'] += dssim_loss * self.rgb_dssim_weight

        if 'anchor_deform_loss' in model_outputs:
            out['anchor_deform_loss'] = model_outputs['anchor_deform_loss']
            out['loss'] += model_outputs['anchor_deform_loss'] * self.anchor_deform_loss_weight
        if 'anchor_scale_loss' in model_outputs:
            out['anchor_scale_loss'] = model_outputs['anchor_scale_loss']
            out['loss'] += model_outputs['anchor_scale_loss'] * self.anchor_scale_loss_weight
        if 'anchor_opacity_loss' in model_outputs:
            out['anchor_opacity_loss'] = model_outputs['anchor_opacity_loss']
            out['loss'] += model_outputs['anchor_opacity_loss'] * self.anchor_opacity_loss_weight
        if 'delta_uv_loss' in model_outputs:
            out['texture_neural_uv_delta_loss'] = model_outputs['delta_uv_loss']
            out['loss'] += model_outputs['delta_uv_loss'] * self.texture_neural_uv_delta_loss_weight
        if 'fine_rgb_reg_loss' in model_outputs:
            out['fine_rgb_reg_loss'] = model_outputs['fine_rgb_reg_loss']
            out['loss'] += model_outputs['fine_rgb_reg_loss'] * self.fine_rgb_reg_loss_weight

        if self.vgg_detach_coarse_layer_weight > 0:
            raise NotImplementedError('no longer supported!')
            bz = model_outputs['batch_size']
            img_res = model_outputs['img_res']
            gt = ground_truth['rgb'].reshape(bz, img_res[0], img_res[1], 3).permute(0, 3, 1, 2)

            predicted = model_outputs['rgb_image_detach_bg_c'].reshape(bz, img_res[0], img_res[1], 3).permute(0, 3, 1, 2)

            vgg_loss = self.get_vgg_loss(predicted, gt)
            out['vgg_detach_bg_c_loss'] = vgg_loss
            out['loss'] += vgg_loss * self.vgg_detach_coarse_layer_weight

        if self.vgg_detach_bg_weight > 0:
            bz = model_outputs['batch_size']
            img_res = model_outputs['img_res']
            gt = ground_truth['rgb'].reshape(bz, img_res[0], img_res[1], 3).permute(0, 3, 1, 2)

            predicted = model_outputs['rgb_image_detach_bg'].reshape(bz, img_res[0], img_res[1], 3).permute(0, 3, 1, 2)

            vgg_loss = self.get_vgg_loss(predicted, gt)
            out['vgg_detach_bg_loss'] = vgg_loss
            out['loss'] += vgg_loss * self.vgg_detach_bg_weight

        if self.sdf_consistency_weight > 0:
            assert self.eikonal_weight > 0
            sdf_consistency_loss = self.get_sdf_consistency_loss(model_outputs['sdf_values'])
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_thetas'])
            out['loss'] += sdf_consistency_loss * self.sdf_consistency_weight + eikonal_loss * self.eikonal_weight
            out['sdf_consistency'] = sdf_consistency_loss
            out['eikonal'] = eikonal_loss

        if self.lbs_weight != 0:
            num_points = model_outputs['lbs_weights'].shape[0]
            ghostbone = model_outputs['lbs_weights'].shape[-1] == 6
            outputs = self.get_gt_blendshape(model_outputs['index_batch'], model_outputs['flame_lbs_weights'],
                                             model_outputs['flame_posedirs'], model_outputs['flame_shapedirs'],
                                             ghostbone)

            lbs_loss = self.get_lbs_loss(model_outputs['lbs_weights'].reshape(num_points, -1),
                                             outputs['gt_lbs_weights'].reshape(num_points, -1),
                                             )

            out['loss'] += lbs_loss * self.lbs_weight * 0.1
            out['lbs_loss'] = lbs_loss

            gt_posedirs = outputs['gt_posedirs'].reshape(num_points, -1)
            posedirs_loss = self.get_lbs_loss(model_outputs['posedirs'].reshape(num_points, -1) * 10,
                                              gt_posedirs* 10,
                                              )
            out['loss'] += posedirs_loss * self.lbs_weight * 10.0
            out['posedirs_loss'] = posedirs_loss

            gt_shapedirs = outputs['gt_shapedirs'].reshape(num_points, -1)
            shapedirs_loss = self.get_lbs_loss(model_outputs['shapedirs'].reshape(num_points, -1)[:, :50*3] * 10,
                                               gt_shapedirs * 10,
                                               use_var_expression=True,
                                               )
            out['loss'] += shapedirs_loss * self.lbs_weight * 10.0
            out['shapedirs_loss'] = shapedirs_loss

        return out