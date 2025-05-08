import os
import torch

import sys
sys.path.append('../code')
sys.path.append('./')

import utils.general as utils
import utils.plots as plt

from functools import partial
from model.point_avatar_model import PointAvatar
from model.gaussian import arguments as gs_args 
from pathlib import Path
import numpy as np
from tqdm import tqdm
import wandb
from skimage.io import imread
import json
import cv2
import os.path as osp
import pandas as pd

import imageio



import argparse


from utils.metrics import img_mse, perceptual, img_ssim, img_psnr

from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--pred_dir', type=str)
parser.add_argument('--gt_dir', type=str) # /home/tw554/pointavatar_gs/data/datasets/Turnbull/Turnbull
opt = parser.parse_args()



gt_dir = Path(opt.gt_dir)
pred_img_dir = Path(opt.pred_dir)


os.system(f"cd /home/tw554/IMavatar/preprocess/submodules/face-parsing.PyTorch; python test.py --dspth {str(gt_dir)} --respth {str(gt_dir / '..' / 'semantic')}")





def cal_metrics(output_dir, gt_dir, pred_file_name, load_npz=False, no_cloth=False, no_head=False):
    res = 512
    files = os.listdir(os.path.join(output_dir))


    use_mask = False
    only_face_interior = False
    no_cloth_mask = no_cloth
    use_mask = use_mask or no_cloth or only_face_interior or no_head

    def _load_img(imgpath):
        image = imread(imgpath).astype(np.float32)
        if image.shape[-2] != res:
            image = cv2.resize(image, (res, res))
        image = image / 255.
        if image.ndim >= 3:
            if image.shape[2] > 3:
                image = image[:, :, :3] * image[:, :, 3:] + (1 - image[:, :, 3:])
            image = image[:, :, :3]
        # 256, 256, 3
        return image

    def _to_tensor(image):
        if image.ndim == 3:
            image = image.transpose(2, 0, 1)
        image = torch.as_tensor(image).unsqueeze(0)
        # 1, 3, 256, 256
        return image

    mse_l = np.zeros(0)
    rmse_l = np.zeros(0)
    mae_l = np.zeros(0)
    perceptual_l = np.zeros(0)
    ssim_l = np.zeros(0)
    psnr_l = np.zeros(0)
    l1_l = np.zeros(0)

    # Keep track of where the images come from
    result_subfolders = list()
    result_filenames = list()

    instance_dir = gt_dir

    

    files = sorted(os.listdir(os.path.join(output_dir)), key=lambda x: int(Path(x).stem))

    for i in tqdm(range(len(files))):
        filename = files[i]
        pred_path = os.path.join(output_dir, filename)

        gt_path = os.path.join(gt_dir, filename)
        mask_path = osp.join(os.path.join(gt_dir, "..", "back", filename))
        mask_path = osp.join(os.path.join(gt_dir, "..", "mask", filename))



        pred = _load_img(pred_path)
        gt = _load_img(gt_path)
        mask = _load_img(mask_path)[...,None]

        # Our prediction has white background, so do the same for GT
        gt_masked = gt * mask + 1.0 * (1 - mask)
        gt = gt_masked

        if no_cloth_mask:
            def load_semantic(path, img_res):
                img = imageio.imread(path, as_gray=True)
                img = cv2.resize(img, (int(img_res), int(img_res)))
                return img
            semantic_path = osp.join(os.path.join(gt_dir, "..", "semantic", filename))
            semantics = load_semantic(semantic_path, img_res=res)
            mask_cloth = np.logical_or(semantics == 16, semantics == 15)
            mask[mask_cloth] = 0.
        elif no_head:
            def load_semantic(path, img_res):
                img = imageio.imread(path, as_gray=True)
                img = cv2.resize(img, (int(img_res), int(img_res)))
                return img
            semantic_path = osp.join(os.path.join(gt_dir, "..", "semantic", filename))
            semantics = load_semantic(semantic_path, img_res=res)
            mask_head = (semantics >= 1) & (semantics <= 13) | (semantics == 17)
            mask[mask_head] = 0.
        w, h, d = gt.shape
        gt = gt.reshape(-1, d)
        # gt[np.sum(gt, 1) == 0., :] = 1 # if background is black, change to white
        gt = gt.reshape(w, h, d)

        pred = _to_tensor(pred)
        gt = _to_tensor(gt)
        mask = _to_tensor(mask)
        mask = mask[:, [0], :, :]


        l1, error_mask = img_mse(pred, gt, mask=mask, error_type='l1', use_mask=use_mask, return_all=True)

        mse = img_mse(pred, gt, mask=mask, error_type='mse', use_mask=use_mask)
        rmse = img_mse(pred, gt, mask=mask, error_type='rmse', use_mask=use_mask)
        mae = img_mse(pred, gt, mask=mask, error_type='mae', use_mask=use_mask)
        perc_error = perceptual(pred, gt, mask, use_mask=use_mask)

        assert mask.size(1) == 1
        if use_mask:
            mask = mask.bool()
            pred_masked = pred.clone()
            gt_masked = gt.clone()
            pred_masked[~mask.expand_as(pred_masked)] = 0
            gt_masked[~mask.expand_as(gt_masked)] = 0

            ssim = img_ssim(pred_masked, gt_masked)
            psnr = img_psnr(pred_masked, gt_masked, rmse=rmse)

        else:
            # import torchvision; torchvision.utils.save_image(gt, 'gt.png')
            # import torchvision; torchvision.utils.save_image(pred, 'pred.png')
            ssim = img_ssim(pred, gt)
            psnr = img_psnr(pred, gt, rmse=rmse)

        if i % 200 == 0:
            print("{}\t{}\t{}\t{}".format(np.mean(mae_l), np.mean(perceptual_l), np.mean(ssim_l), np.mean(psnr_l)))
        mse_l = np.append(mse_l, mse)
        rmse_l = np.append(rmse_l, rmse)
        mae_l = np.append(mae_l, mae)
        perceptual_l = np.append(perceptual_l, perc_error)
        ssim_l = np.append(ssim_l, ssim)
        psnr_l = np.append(psnr_l, psnr)
        l1_l = np.append(l1_l, l1)


        result_filenames.append(filename)

    result = {
        "filenames": result_filenames,
        "mse_l": mse_l.copy(),
        "rmse_l": rmse_l.copy(),
        "mae_l": mae_l.copy(),
        "perceptual_l": perceptual_l.copy(),
        "ssim_l": ssim_l.copy(),
        "psnr_l": psnr_l.copy(),
        "l1_l": l1_l.copy(),
    }
    base_result_name = "results"
    if no_cloth_mask:
        base_result_name = "results_no_cloth"
    elif no_head:
        base_result_name = "results_no_head"
    path_result_npz = os.path.join(output_dir, '..', "{}_{}.npz".format(base_result_name, pred_file_name))
    path_result_csv = os.path.join(output_dir, '..', "{}_{}.csv".format(base_result_name, pred_file_name))
    # np.savez(path_result_npz, **result)
    df = pd.DataFrame.from_dict(result)
    df = df.sort_values('filenames', key=lambda x: x.str.strip('.png').astype(int))
    df.to_csv(path_result_csv)
    print("Written result to ", path_result_npz)

    print("{}\t{}\t{}\t{}".format(np.mean(mae_l), np.mean(perceptual_l), np.mean(ssim_l), np.mean(psnr_l)))

    json_name = "metrics.json"
    if no_cloth:
        json_name = "metrics_no_cloth.json"
    elif no_head:
        json_name = "metrics_no_head.json"

    avg_metrics = {
        "mse": np.mean(mse_l),
        "psnr": np.mean(psnr_l),
        "ssim": np.mean(ssim_l),
        "lpips": np.mean(perceptual_l),
    }
    with open(os.path.join(output_dir, '..', json_name), 'w') as f:
        json.dump(avg_metrics, f)

    return avg_metrics


metrics = cal_metrics(output_dir=str(pred_img_dir), gt_dir=str(gt_dir), pred_file_name='')

metrics_no_cloth = cal_metrics(output_dir=str(pred_img_dir), gt_dir=str(gt_dir), pred_file_name='', no_cloth=True)


