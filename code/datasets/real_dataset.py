import os
import torch
import numpy as np
import cv2
import json
import imageio
import skimage
from tqdm import tqdm
from pathlib import Path
import json
from scipy import ndimage
import math
import matplotlib

class FaceDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_folder,
                 subject_name,
                 json_name,
                 sub_dir,
                 img_res,
                 is_eval,
                 subsample=1,
                 hard_mask=False,
                 hard_head_mask=False, # for head mask loss
                 only_json=False,
                 use_mean_expression=False,
                 use_var_expression=False,
                 use_background=False,
                 load_images=False,
                 load_body_ldmk=False,
                 load_ldmk_drawing=False,
                 draw_hand=False,
                 ldmk_drawing_downscale=1,
                 draw_point_size=8,
                 draw_line_size=6,
                 blur_drawing=True,
                 smooth_ldmk=True,
                 smooth_ldmk_savgol = False,
                 frame_interval=[0,999999],
                 canonical_frame_id=0, # a canonical frame for anchor initialization and update, this is id after subsampling
                 ):
        """
        sub_dir: list of scripts/testing subdirectories for the subject, e.g. [MVI_1810, MVI_1811]
        Data structure:
            RGB images in data_folder/subject_name/subject_name/sub_dir[i]/image
            foreground masks in data_folder/subject_name/subject_name/sub_dir[i]/mask
            json files containing FLAME parameters in data_folder/subject_name/subject_name/sub_dir[i]/json_name
        json file structure:
            frames: list of dictionaries, which are structured like:
                file_path: relative path to image
                world_mat: camera extrinsic matrix (world to camera). Camera rotation is actually the same for all frames,
                           since the camera is fixed during capture.
                           The FLAME head is centered at the origin, scaled by 4 times.
                expression: 50 dimension expression parameters
                pose: 15 dimension pose parameters
                flame_keypoints: 2D facial keypoints calculated from FLAME
            shape_params: 100 dimension FLAME shape parameters, shared by all scripts and testing frames of the subject
            intrinsics: camera focal length fx, fy and the offsets of the principal point cx, cy
        img_res: a list containing height and width, e.g. [256, 256] or [512, 512]
        subsample: subsampling the images to reduce frame rate, mainly used for inference and evaluation
        hard_mask: whether to use boolean segmentation mask or not
        only_json: used for testing, when there is no GT images or masks. If True, only load json.
        use_background: if False, replace with white background. Otherwise, use original background
        load_images: if True, load images at the beginning instead of at each iteration
        use_mean_expression: if True, use mean expression of the training set as the canonical expression
        use_var_expression: if True, blendshape regularization weight will depend on the variance of expression
                            (more regularization if variance is small in the training set.)
        """
        sub_dir = [str(dir) for dir in sub_dir]
        self.img_res = img_res
        self.use_background = use_background
        self.load_images = load_images
        self.hard_mask = hard_mask
        self.load_body_ldmk = load_body_ldmk
        self.load_ldmk_drawing = load_ldmk_drawing
        self.canonical_frame_id = canonical_frame_id
        self.hard_head_mask = hard_head_mask

        self.data = {
            "image_paths": [],
            "mask_paths": [],
            # camera extrinsics
            "world_mats": [],
            # FLAME expression and pose parameters
            "expressions": [],
            "flame_pose": [],
            # saving image names and subdirectories
            "img_name": [],
            "sub_dir": [],
        }

        if self.load_body_ldmk:
            self.data['body_ldmks'] = []
            self.data['dwposes'] = []


        if smooth_ldmk:
            smoother = OneEuroFilter(freq=30, mincutoff=1., beta=0.)
            smoother.reset()

        for dir in sub_dir:
            instance_dir = os.path.join(data_folder, subject_name, subject_name, dir)
            assert os.path.exists(instance_dir), "Data directory {} is empty".format(instance_dir)

            cam_file = '{0}/{1}'.format(instance_dir, json_name)

            with open(cam_file, 'r') as f:
                camera_dict = json.load(f)
            for frame in tqdm(camera_dict['frames'], desc='loading annotations'):
                # world to camera matrix
                world_mat = np.array(frame['world_mat']).astype(np.float32)
                # camera to world matrix
                self.data["world_mats"].append(world_mat)
                self.data["expressions"].append(np.array(frame['expression']).astype(np.float32))
                self.data["flame_pose"].append(np.array(frame['pose']).astype(np.float32))
                self.data["sub_dir"].append(dir)
                image_path = '{0}/{1}.png'.format(instance_dir, frame["file_path"])
                self.data["image_paths"].append(image_path)
                self.data["mask_paths"].append(image_path.replace('image', 'mask'))
                self.data["img_name"].append(int(frame["file_path"].split('/')[-1]))

                if self.load_body_ldmk:
                    img_name = Path(frame['file_path']).name

                    dwpose_path = Path(image_path).parent.parent / 'dwpose' / f"{img_name}.npy"

                    if dwpose_path.exists():
                        dwpose = np.load(str(dwpose_path), allow_pickle=True).item()
                        # smooth dwpose dict
                        all_poses = np.concatenate([dwpose['bodies']['candidate'][:18].reshape(-1), dwpose['hands'][:2].reshape(-1)])
                        all_poses = smoother(all_poses, (len(self.data["img_name"])-1) / smoother.init_freq)
                        dwpose['bodies']['candidate'] = all_poses[:36].reshape(dwpose['bodies']['candidate'][:18].shape)
                        dwpose['hands'] = all_poses[36:].reshape(dwpose['hands'][:2].shape)

                        ldmks = dwpose['bodies']['candidate']
                        ldmks = np.concatenate([ldmks, np.ones_like(ldmks[:,:1])], axis=-1) # add fake confidence

                    else:
                        raise NotImplementedError('no longer supported')
                        assert not use_dwpose, "do not allow part dwpose and part openpose"

                        openpose_path = Path(image_path).parent.parent / 'openpose_json' / f"{img_name}_keypoints.json"
                        with openpose_path.open('r') as f:
                            openpose = json.load(f)
                        ldmks = np.array(openpose['people'][0]['pose_keypoints_2d']).reshape([-1,3]) # x,y,confidence


                        if self.load_ldmk_drawing:
                            raise NotImplementedError

                    # we only take landmarks for shoulders
                    body_ldmks = ldmks[[0, 1, 2, 5]]
                    # body_ldmks = ldmks[[0, 1, 2, 3, 5, 6]]
                    # # replace ldmks with very low confidence, only needed for openpose ldmks
                    # for i in range(body_ldmks.shape[0]):
                    #     if body_ldmks[i, 2] < 0.1:
                    #         body_ldmks[i] = self.data['body_ldmks'][-1][i]
                    self.data['body_ldmks'].append(np.array(body_ldmks)[:,:2].reshape([-1]))
                    self.data['dwposes'].append(dwpose)

        if self.load_body_ldmk and smooth_ldmk_savgol:
            from scipy.signal import savgol_filter
            ldmks_np = np.stack(self.data["body_ldmks"], 0)
            ldmks_np = savgol_filter(ldmks_np, 13, 5,axis=0)
            self.data["body_ldmks"] = ldmks_np

            # from scipy.ndimage import gaussian_filter1d
            # ldmks_np = np.stack(self.data["body_ldmks"], 0)
            # ldmks_np = gaussian_filter1d(ldmks_np, 1, axis=0, mode='nearest')
            # self.data["body_ldmks"] = ldmks_np
            

        self.gt_dir = instance_dir
        self.shape_params = torch.tensor(camera_dict['shape_params']).float().unsqueeze(0)
        focal_cxcy = camera_dict['intrinsics']

        # train test split for dataset that store frames together
        if "split.txt" in frame_interval:
            # read split from file
            assert len(sub_dir) == 1
            split_path = Path(data_folder) / subject_name / subject_name / dir / 'split.txt'
            if split_path.exists():
                split = int(np.loadtxt(str(split_path)).item())
            else:
                split = -1000
            if frame_interval[0] == 'split.txt':
                frame_interval[0] = split
            if frame_interval[1] == 'split.txt':
                frame_interval[1] = split
                
        self.frame_interval = frame_interval
        for k, v in self.data.items():
            self.data[k] = v[frame_interval[0]:frame_interval[1]]

        if isinstance(subsample, int) and subsample > 1:
            for k, v in self.data.items():
                self.data[k] = v[::subsample]
        elif isinstance(subsample, list):
            if len(subsample) == 2:
                subsample = list(range(subsample[0], subsample[1]))
            for k, v in self.data.items():
                self.data[k] = [v[s] for s in subsample]

        self.data["expressions"] = torch.from_numpy(np.stack(self.data["expressions"], 0))
        self.data["flame_pose"] = torch.from_numpy(np.stack(self.data["flame_pose"], 0))
        self.data["world_mats"] = torch.from_numpy(np.stack(self.data["world_mats"], 0)).float()
        if self.load_body_ldmk:
            self.data["body_ldmks"] = torch.from_numpy(np.stack(self.data["body_ldmks"], 0)).float()

        # construct intrinsic matrix
        intrinsics = np.zeros((4, 4))

        # from whatever camera convention to pytorch3d
        intrinsics[0, 0] = focal_cxcy[0] * 2
        intrinsics[1, 1] = focal_cxcy[1] * 2
        intrinsics[0, 2] = (focal_cxcy[2] * 2 - 1.0) * -1
        intrinsics[1, 2] = (focal_cxcy[3] * 2 - 1.0) * -1

        intrinsics[3, 2] = 1.
        intrinsics[2, 3] = 1.
        self.intrinsics = intrinsics

        if intrinsics[0, 0] < 0:
            intrinsics[:, 0] *= -1
            self.data["world_mats"][:, 0, :] *= -1
        self.data["world_mats"][:, :3, 2] *= -1
        self.data["world_mats"][:, 2, 3] *= -1

        if use_mean_expression:
            self.mean_expression = torch.mean(self.data["expressions"], 0, keepdim=True)
        else:
            self.mean_expression = torch.zeros_like(self.data["expressions"][[0], :])
        if use_var_expression:
            self.var_expression = torch.var(self.data["expressions"], 0, keepdim=True)
        else:
            self.var_expression = None

        self.intrinsics = torch.from_numpy(self.intrinsics).float()
        self.only_json = only_json

        if self.load_body_ldmk:
            subset_mask = np.zeros(18)
            for dwpose in self.data['dwposes']:
                subset = dwpose['bodies']['subset'][0]
                subset_mask += (subset > -1)

            subset_mask = subset_mask / len(self.data['dwposes']) > 0.95

            dwpose_subset = np.arange(18).astype(float)
            dwpose_subset[~subset_mask] = -1
            dwpose_subset = dwpose_subset[None, ...]

        images = []
        masks = []
        head_masks = []
        anchor_head_filter_masks = []
        # bg_layer_masks = []
        ldmk_drawings = []
        if load_images and not only_json:
            print("Loading all images, this might take a while.")
            for idx in tqdm(range(len(self.data["image_paths"]))):
                rgb = torch.from_numpy(load_rgb(self.data["image_paths"][idx], self.img_res).reshape(3, -1).transpose(1,0)).float()
                object_mask = torch.from_numpy(load_mask(self.data["mask_paths"][idx], self.img_res).reshape(-1))
                if not self.use_background:
                    if not hard_mask:
                        rgb = rgb * object_mask.unsqueeze(1).float() + (1 - object_mask.unsqueeze(1).float())
                    else:
                        rgb = rgb * (object_mask.unsqueeze(1) > 0.5) + ~(object_mask.unsqueeze(1) > 0.5)
                images.append(rgb)
                masks.append(object_mask)

                # load semantic mask to obtain head mask
                semantic_mask = load_semantic(self.data["mask_paths"][idx].replace('mask', 'semantic'), self.img_res)

                if not self.hard_head_mask:
                    mask_cloth = np.logical_or(semantic_mask == 16, semantic_mask == 15)
                    mask_cloth = torch.from_numpy(mask_cloth).reshape(-1)
                    head_mask = object_mask * (~mask_cloth)
                else:
                    head_mask = (semantic_mask >= 1) & (semantic_mask <= 14) | (semantic_mask == 17)
                    head_mask = torch.from_numpy(head_mask).reshape(-1)

                head_masks.append(head_mask)

                

                anchor_head_filter_mask = ((semantic_mask >= 1) & (semantic_mask <= 13) | (semantic_mask == 17)) 
                # anchor_head_filter_mask = ndimage.binary_dilation(anchor_head_filter_mask, iterations=32)
                anchor_head_filter_mask = torch.from_numpy(anchor_head_filter_mask).reshape(-1)
                anchor_head_filter_masks.append(anchor_head_filter_mask)


                # # mask for bg layer, include cloth & neck
                # no_neck = ndimage.binary_dilation((semantic_mask <= 13) & (semantic_mask > 0) | (semantic_mask == 17), iterations=10).reshape(-1)
                # bg_layer_mask = (object_mask > 0.1) & (~no_neck)
                # # bg_layer_mask = ndimage.binary_dilation(bg_layer_mask, iterations=2)
                # # bg_layer_mask = torch.from_numpy(bg_layer_mask).reshape(-1)
                # bg_layer_masks.append(bg_layer_mask)


                # ldmk_drawing_path = Path(self.data["image_paths"][idx]).parent.parent / 'dwpose' / f"{img_name}_body.png"

                # dwpose = self.data["dwposes"][idx]
                # candidate = dwpose['bodies']['candidate'].copy()
                # # do not draw face part
                # # candidate[[14, 15, 16, 17],:] = -1

                # # candidate[[0, 1, 2, 5]]
                # candidate[[3, 6,7,8,9,10,11,12,13,14, 15, 16, 17],:] = -1
                # # candidate[[4],:] = -1

                # ldmk_drawing = draw_bodypose(
                #     # np.zeros(shape=(img_res[0]//ldmk_drawing_downscale, img_res[1]//ldmk_drawing_downscale, 3), dtype=np.uint8) + 255, 
                #     (rgb.reshape([512,512,3]).numpy() * 255).astype(np.uint8), 
                #     candidate, 
                #     # dwpose['bodies']['subset'],
                #     dwpose_subset,
                #     point_size=draw_point_size, line_size=draw_line_size)
                # imageio.imwrite(str(ldmk_drawing_path), ldmk_drawing)
                # print(str(ldmk_drawing_path))
                # ldmk_drawing = ldmk_drawing / 255.



                # if self.load_ldmk_drawing:
                #     img_name = Path(self.data["image_paths"][idx]).stem
                #     ldmk_drawing_path = Path(self.data["image_paths"][idx]).parent.parent / 'dwpose' / f"{img_name}_body.png"
                #     # if ldmk_drawing_path.exists():
                #     if False:
                #         ldmk_drawing = imageio.imread(str(ldmk_drawing_path)) / 255.
                #     else:
                #         dwpose = self.data["dwposes"][idx]
                #         candidate = dwpose['bodies']['candidate'].copy()
                #         # do not draw face part
                #         candidate[[14, 15, 16, 17],:] = -1
                #         # candidate[[4],:] = -1

                #         ldmk_drawing = draw_bodypose(
                #             np.zeros(shape=(img_res[0]//ldmk_drawing_downscale, img_res[1]//ldmk_drawing_downscale, 3), dtype=np.uint8), 
                #             candidate, 
                #             # dwpose['bodies']['subset'],
                #             dwpose_subset,
                #             point_size=draw_point_size, line_size=draw_line_size)
                #         if draw_hand:
                #             ldmk_drawing = draw_handpose(ldmk_drawing, dwpose['hands'],
                #                                          point_size=draw_point_size, line_size=draw_line_size)
                #         if blur_drawing:
                #             ldmk_drawing = cv2.GaussianBlur(ldmk_drawing, (9,9), 1)
                #             # blur = cv2.GaussianBlur(ldmk_drawing, (9,9), 1)
                #         imageio.imwrite(str(ldmk_drawing_path), ldmk_drawing)
                #         ldmk_drawing = ldmk_drawing / 255.

                #     ldmk_drawings.append(torch.from_numpy(ldmk_drawing).float())
                

        self.data['images'] = images
        self.data['masks'] = masks
        self.data['head_masks'] = head_masks
        self.data['anchor_head_filter_masks'] = anchor_head_filter_masks
        # self.data['bg_layer_masks'] = bg_layer_masks
        if len(ldmk_drawings) > 0:
            self.data['ldmk_drawings'] =  ldmk_drawings
            
            # import torchvision; torchvision.io.write_video('body_smooth.mp4', torch.from_numpy(np.stack(ldmk_drawings, 0)).float() *255, fps=30, video_codec='h264')

    def __len__(self):
        return len(self.data["image_paths"])

    def __getitem__(self, idx):
        sample = {
            "idx": torch.LongTensor([idx]),
            "img_name": torch.LongTensor([self.data["img_name"][idx]]),
            "sub_dir": self.data["sub_dir"][idx],
            "intrinsics": self.intrinsics,
            "expression": self.data["expressions"][idx],
            "flame_pose": self.data["flame_pose"][idx],
            "cam_pose": self.data["world_mats"][idx],
            }
        
        if self.load_body_ldmk:
            sample['body_ldmk'] = self.data['body_ldmks'][idx]
            # sample['full_ldmk'] = self.data['full_ldmks'][idx]


        ground_truth = {}

        if not self.only_json:
            if not self.load_images:
                # raise NotImplementedError('No longer supported!')
                ground_truth["object_mask"] = torch.from_numpy(load_mask(self.data["mask_paths"][idx], self.img_res).reshape(-1))
                rgb = torch.from_numpy(load_rgb(self.data["image_paths"][idx], self.img_res).reshape(3, -1).transpose(1, 0)).float()
                if not self.use_background:
                    if not self.hard_mask:
                        ground_truth['rgb'] = rgb * ground_truth["object_mask"].unsqueeze(1).float() + (1 - ground_truth["object_mask"].unsqueeze(1).float())
                    else:
                        ground_truth['rgb'] = rgb * (ground_truth["object_mask"].unsqueeze(1) > 0.5) + ~(ground_truth["object_mask"].unsqueeze(1) > 0.5)
                else:
                    ground_truth['rgb'] = rgb

                # load semantic mask to obtain head mask
                semantic_mask = load_semantic(self.data["mask_paths"][idx].replace('mask', 'semantic'), self.img_res)
                object_mask = ground_truth["object_mask"]
                if not self.hard_head_mask:
                    mask_cloth = np.logical_or(semantic_mask == 16, semantic_mask == 15)
                    mask_cloth = torch.from_numpy(mask_cloth).reshape(-1)
                    head_mask = object_mask * (~mask_cloth)
                else:
                    head_mask = (semantic_mask >= 1) & (semantic_mask <= 14) | (semantic_mask == 17)
                    hair_mask = semantic_mask == 17
                    head_mask = head_mask | ndimage.binary_dilation(hair_mask, iterations=32)
                    head_mask = torch.from_numpy(head_mask).reshape(-1).float()
                    head_mask = object_mask * head_mask
                    head_mask = (head_mask > 0.05).float()
                    
                # import torchvision; torchvision.utils.save_image(head_mask.reshape([1,1,512,512]), 'm1.png')
                # import torchvision; torchvision.utils.save_image(head_mask.reshape([1,1,512,512]), 'm2.png')
                # import torchvision; torchvision.utils.save_image(rgb.T.reshape([1,3,512,512]), 'rgb.png')
                ground_truth['head_mask'] = head_mask

                anchor_head_filter_mask = ((semantic_mask >= 1) & (semantic_mask <= 13) | (semantic_mask == 17)) 
                # anchor_head_filter_mask = ndimage.binary_dilation(anchor_head_filter_mask, iterations=32)
                anchor_head_filter_mask = torch.from_numpy(anchor_head_filter_mask).reshape(-1)
                sample['anchor_head_filter_mask'] = anchor_head_filter_mask

            else:
                ground_truth = {
                    'rgb': self.data['images'][idx],
                    'object_mask': self.data['masks'][idx],
                    'head_mask': self.data['head_masks'][idx],
                    # 'bg_layer_mask': self.data['bg_layer_masks'][idx],
                }
                sample['anchor_head_filter_mask'] = self.data['anchor_head_filter_masks'][idx]
            if self.load_ldmk_drawing:
                sample['ldmk_drawing'] = self.data['ldmk_drawings'][idx]
            # sample['head_mask'] = ground_truth['head_mask']

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # this function is borrowed from https://github.com/lioryariv/idr/blob/main/code/datasets/scene_dataset.py
        # get list of dictionaries and returns sample, ground_truth as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    try:
                        ret[k] = torch.stack([obj[k] for obj in entry])
                    except:
                        ret[k] = [obj[k] for obj in entry]
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)
    
    def get_canonical_frame_inputs(self):
        _, can_inputs, _ = self.collate_fn([self[self.canonical_frame_id]])

        return can_inputs

def load_rgb(path, img_res):
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)

    img = cv2.resize(img, (int(img_res[0]), int(img_res[1])))
    img = img.transpose(2, 0, 1)
    return img


def load_mask(path, img_res):
    alpha = imageio.imread(path, as_gray=True)
    alpha = skimage.img_as_float32(alpha)

    alpha = cv2.resize(alpha, (int(img_res[0]), int(img_res[1])))
    object_mask = alpha / 255

    return object_mask


def load_semantic(path, img_res):
    img = imageio.imread(path, as_gray=True)
    img = cv2.resize(img, tuple(img_res))
    return img





def draw_bodypose(canvas, candidate, subset, point_size=4, line_size=4):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    # colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
    #           [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
    #           [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [170, 255, 0], [85, 255, 0], [255, 170, 0] , \
              [0, 255, 85], [0, 255, 170], [0, 170, 255], [0, 85, 255], [255, 85, 0], [85, 0, 255], \
              [170, 0, 255],  [255, 0, 170], [255, 0, 85]]

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)

            if (candidate[index.astype(int), 0] == -1).any() or (candidate[index.astype(int), 1] == -1).any():
                continue

            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), line_size), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), point_size, colors[i], thickness=-1)

    return canvas

def draw_handpose(canvas, all_hand_peaks, point_size=4, line_size=4):
    eps = 0.01

    H, W, C = canvas.shape

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for peaks in all_hand_peaks:
        peaks = np.array(peaks)

        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255, thickness=line_size//2)

        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), point_size, (0, 0, 255), thickness=-1)
    return canvas



# OneEuroFilter.py -
#
# Authors: 
# Nicolas Roussel (nicolas.roussel@inria.fr)
# Géry Casiez https://gery.casiez.net
#
# Copyright 2019 Inria
# 
# BSD License https://opensource.org/licenses/BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this list of conditions
# and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
# and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or
# promote products derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

class LowPassFilter(object):

    def __init__(self, alpha):
        self.__setAlpha(alpha)
        self.__y = self.__s = None

    def __setAlpha(self, alpha):
        # alpha = float(alpha)
        if (alpha<=0).any() or (alpha>1.0).any():
            raise ValueError("alpha (%s) should be in (0.0, 1.0]"%alpha)
        self.__alpha = alpha

    def __call__(self, value, timestamp=None, alpha=None):        
        if alpha is not None:
            self.__setAlpha(alpha)
        if self.__y is None:
            s = value
        else:
            s = self.__alpha*value + (1.0-self.__alpha)*self.__s
        self.__y = value
        self.__s = s
        return s

    def lastValue(self):
        return self.__y
    
    def lastFilteredValue(self):
        return self.__s

# ----------------------------------------------------------------------------

class OneEuroFilter(object):

    def __init__(self, freq, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        if freq<=0:
            raise ValueError("freq should be >0")
        if mincutoff<=0:
            raise ValueError("mincutoff should be >0")
        if dcutoff<=0:
            raise ValueError("dcutoff should be >0")
        # self.freq = torch.tensor(freq, dtype=torch.float32)
        # self.__mincutoff = torch.tensor(mincutoff, dtype=torch.float32)
        # self.__beta = torch.tensor(beta, dtype=torch.float32)
        # self.__dcutoff = torch.tensor(dcutoff, dtype=torch.float32)
        self.init_freq = np.array(freq)
        self.freq = np.array(freq)
        self.__mincutoff = np.array(mincutoff)
        self.__beta = np.array(beta)
        self.__dcutoff = np.array(dcutoff)
        self.__x = LowPassFilter(self.__alpha(self.__mincutoff))
        self.__dx = LowPassFilter(self.__alpha(self.__dcutoff))
        self.__lasttime = None

    def reset(self):
        self.__lasttime = None
        self.freq = self.init_freq
        self.__x = LowPassFilter(self.__alpha(self.__mincutoff))
        self.__dx = LowPassFilter(self.__alpha(self.__dcutoff))
        
    def __alpha(self, cutoff):
        te    = 1.0 / self.freq
        tau   = 1.0 / (2*math.pi*cutoff)
        return  1.0 / (1.0 + tau/te)

    def __call__(self, x, timestamp=None):
        # ---- update the sampling frequency based on timestamps
        if self.__lasttime and timestamp:
            self.freq = 1.0 / (timestamp-self.__lasttime)
        self.__lasttime = timestamp
        # ---- estimate the current variation per second
        prev_x = self.__x.lastFilteredValue()
        dx = np.zeros_like(x) if prev_x is None else (x-prev_x)*self.freq # FIXME: 0.0 or value?
        edx = self.__dx(dx, timestamp, alpha=self.__alpha(self.__dcutoff))
        # ---- use it to update the cutoff frequency
        cutoff = self.__mincutoff + self.__beta* np.abs(edx)
        # ---- filter the given value
        return self.__x(x, timestamp, alpha=self.__alpha(cutoff))
