import os
from pyhocon import ConfigFactory
import sys
import torch

import utils.general as utils
import utils.plots as plt

from functools import partial
from model.point_avatar_model import PointAvatar
from model.gaussian import arguments as gs_args 
from pathlib import Path
import torchvision
import numpy as np
from tqdm import tqdm
import wandb
from scipy.signal import savgol_filter


print = partial(print, flush=True)
class ReenactRunner():
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)
        self.conf = kwargs['conf']
        self.conf.put('dataset.test.subsample', 1)
        default_methodname = Path(kwargs['conf_path']).stem
        if Path(kwargs['conf_path']).parent.name != 'confs':
            default_methodname = f"{Path(kwargs['conf_path']).parent.name}/{Path(kwargs['conf_path']).stem}"
        self.conf['train']['methodname'] = self.conf['train'].get_string('methodname', default_methodname)

        if kwargs['subject'] is not None:
            self.conf['dataset']['subject_name'] = kwargs['subject']

        self.quick_eval = kwargs['quick_eval']
        if self.quick_eval:
            self.conf.put('dataset.test.subsample', 20)

        self.exps_folder_name = self.conf.get_string('train.exps_folder')
        self.subject = self.conf.get_string('dataset.subject_name')
        self.methodname = self.conf.get_string('train.methodname')

        self.expdir = os.path.join(self.exps_folder_name, self.subject, self.methodname)
        train_split_name = utils.get_split_name(self.conf.get_list('dataset.train.sub_dir'))

        self.eval_dir = os.path.join(self.expdir, train_split_name, 'eval')
        if self.quick_eval:
            self.eval_dir = os.path.join(self.expdir, train_split_name, 'eval_quick')

        self.train_dir = os.path.join(self.expdir, train_split_name, 'train')

        if kwargs['load_path'] != '':
            load_path = kwargs['load_path']
        else:
            load_path = self.train_dir
        assert os.path.exists(load_path), load_path

        utils.mkdir_ifnotexists(self.eval_dir)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        self.use_background = self.conf.get_bool('dataset.use_background', default=False)

        load_body_ldmk = True
        load_ldmk_drawing = False


        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(data_folder=self.conf.get_string('dataset.data_folder'),
                                                                                          subject_name=self.conf.get_string('dataset.subject_name'),
                                                                                          json_name=self.conf.get_string('dataset.json_name'),
                                                                                          use_mean_expression=self.conf.get_bool('dataset.use_mean_expression', default=False),
                                                                                          use_background=self.use_background,
                                                                                          is_eval=False,
                                                                                          only_json=True,
                                                                                          load_body_ldmk=load_body_ldmk,
                                                                                          load_ldmk_drawing=load_ldmk_drawing,
                                                                                          **self.conf.get_config('dataset.train'))
        
        if self.conf.get_string('dataset.test_reenact_subject', None) is not None:
            plot_subject_name = self.conf.get_string('dataset.test_reenact_subject')
        else:
            plot_subject_name = self.conf.get_string('dataset.subject_name')

        self.plot_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(data_folder=self.conf.get_string('dataset.data_folder'),
                                                                                         subject_name=plot_subject_name,
                                                                                         json_name=self.conf.get_string('dataset.json_name'),
                                                                                         only_json=kwargs['only_json'],
                                                                                         use_background=self.use_background,
                                                                                         is_eval=True,
                                                                                         load_body_ldmk=load_body_ldmk,
                                                                                         smooth_ldmk_savgol=True,
                                                                                         load_ldmk_drawing=load_ldmk_drawing,
                                                                                         **self.conf.get_config('dataset.test'))
        

        print('Finish loading data ...')

        self.gs_model_args = gs_args.ModelParams(**self.conf.get_config('gs_model', {}))
        self.model = PointAvatar(conf=self.conf.get_config('model'),
                                shape_params=self.plot_dataset.shape_params,
                                img_res=self.plot_dataset.img_res,
                                canonical_expression=self.train_dataset.mean_expression,
                                canonical_pose=self.conf.get_float(
                                    'dataset.canonical_pose',
                                    default=0.2),
                                use_background=self.use_background,
                                gs_model_args=self.gs_model_args,
                                )
        self.simulate_fast_anchor = kwargs.get('run_fast_test', False)
        if self.model.gs_img_model is None:
            self.simulate_fast_anchor = False

        if self.simulate_fast_anchor:
            self.eval_dir += '_simulate_fast'

        if torch.cuda.is_available():
            self.model.cuda()
        old_checkpnts_dir = os.path.join(load_path, 'checkpoints')
        self.checkpoints_path = old_checkpnts_dir
        assert os.path.exists(old_checkpnts_dir)
        
        self.model_ckpt_path = os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth")
        saved_model_state = torch.load(self.model_ckpt_path)
        # n_points = saved_model_state["model_state_dict"]['pc.points'].shape[0]
        # self.model.pc.init(n_points)
        # self.model.pc = self.model.pc.cuda()

        self.model.pc.load_ply(os.path.join(old_checkpnts_dir, 'GaussianPly', str(kwargs['checkpoint']) + ".ply"))

        self.model.raster_settings.radius = saved_model_state['radius']

        self.model.load_state_dict(saved_model_state["model_state_dict"]) #, strict=False
        if self.model.bg_layer_network is not None:
            self.model.bg_layer_network.deform_warm_up = False
        if self.model.gs_img_model is not None:
            self.model.gs_img_model.update_anchor_mask()
            
        # at test time, do not optimize pose, cam with layer
        # as this tend to break the head part
        self.model.conf['detach_pose_input'] = True
        self.model.conf['detach_cam_input'] = True
        self.start_epoch = saved_model_state['epoch']





        self.eval_dir = self.eval_dir + f"_reenact_{plot_subject_name}"



        self.plot_dataloader = torch.utils.data.DataLoader(self.plot_dataset,
                                                           batch_size=min(int(self.conf.get_int('train.max_points_training') /self.model.pc.points.shape[0]),self.conf.get_int('train.max_batch',default='10')),
                                                           shuffle=False,
                                                           collate_fn=self.plot_dataset.collate_fn
                                                           )
        self.optimize_tracking = False
        

        self.img_res = self.plot_dataset.img_res

    def run(self):
        self.model.eval() # TODO: change this if warping field needs to be updated
        self.model.training = False

        eval_all = True
        is_first_batch = True
        imgs = []
        _, base_input, _ = self.plot_dataset.collate_fn([self.plot_dataset[0]])

        # TEST_TIME_AFFINE = False
        TEST_TIME_AFFINE = True

        if self.model.gs_img_model is not None and self.model.gs_img_model.apply_affine:
            TEST_TIME_AFFINE = False

        if self.simulate_fast_anchor:
            # self.model.gs_img_model.affine_type = 'similarity'
            TEST_TIME_AFFINE = False
            distill_inputs = self.train_dataset.get_canonical_frame_inputs()
            self.model.gs_img_model.apply_test_time_affine = True
            # _, distill_inputs, _ = self.plot_dataset.collate_fn([self.plot_dataset[0]])
            for k, v in distill_inputs.items():
                try:
                    distill_inputs[k] = v.cuda()
                except:
                    distill_inputs[k] = v
            self.model(distill_inputs, distill_anchor_model=True)


        if TEST_TIME_AFFINE and self.model.gs_img_model is not None:
            self.model.gs_img_model.apply_test_time_affine = True
            self.model.gs_img_model.affine_type = 'euclidean'
            # self.model.gs_img_model.affine_type = 'translation'

            self.eval_dir += f'_test_{self.model.gs_img_model.affine_type}'

            model_input = self.train_dataset.get_canonical_frame_inputs()
            for k, v in model_input.items():
                try:
                    model_input[k] = v.cuda()
                except:
                    model_input[k] = v
            with torch.no_grad():
                self.model(
                    model_input, 
                    updating_anchor_correspondence=True,
                )
                self.model(
                    model_input, 
                    pruning_outside_anchor=True,
                )
                self.model(
                    model_input, 
                    distill_anchor_texture=True,
                )


        self.mlp_warp_smooth = False

        with torch.no_grad():
            if self.mlp_warp_smooth:
                mlp_delta_uvs = []
                eval_iterator = iter(self.plot_dataloader)

                for img_index in tqdm(range(len(self.plot_dataloader)), desc='Smoothing'):
                    indices, model_input, ground_truth = next(eval_iterator)
                    batch_size = model_input['expression'].shape[0]
                    for k, v in model_input.items():
                        try:
                            model_input[k] = v.cuda()
                        except:
                            model_input[k] = v
                    

                    model_outputs = self.model(model_input, clip_texture=self.conf.get('train.train_texture_only_clip', False))

                    if "mlp_warp_delta_uv" in model_outputs:
                        mlp_delta_uvs.append(model_outputs["mlp_warp_delta_uv"].detach().cpu())

                mlp_delta_uvs = torch.concat(mlp_delta_uvs, axis=0).cpu().numpy()
                delta_uv_smooth = savgol_filter(mlp_delta_uvs, 3, 2, axis=0)
                smoothed_mlp_delta_uvs = torch.from_numpy(delta_uv_smooth)

        
        eval_iterator = iter(self.plot_dataloader)
        for img_index in tqdm(range(len(self.plot_dataloader)), desc='Rendering'):
            indices, reenact_input, ground_truth = next(eval_iterator)

            # indices, reenact_input, ground_truth =  self.plot_dataset.collate_fn([self.plot_dataset[50]])

            # _, model_input, _ = self.train_dataset.collate_fn([self.train_dataset[0]])
            model_input = self.train_dataset.get_canonical_frame_inputs()

            batch_size = model_input['expression'].shape[0]
            for k, v in model_input.items():
                try:
                    model_input[k] = v.cuda()
                except:
                    model_input[k] = v
            for k, v in reenact_input.items():
                try:
                    reenact_input[k] = v.cuda()
                except:
                    reenact_input[k] = v
            for k, v in base_input.items():
                try:
                    base_input[k] = v.cuda()
                except:
                    base_input[k] = v

            for k, v in ground_truth.items():
                try:
                    ground_truth[k] = v.cuda()
                except:
                    ground_truth[k] = v



            model_input['expression'] = reenact_input['expression']
            model_input['flame_pose'] = reenact_input['flame_pose']
            
             

            if 'body_ldmk' in reenact_input:
                reenact_ldmks = reenact_input['body_ldmk'].reshape([-1, 2])
                original_ldmks = model_input['body_ldmk'].reshape([-1, 2])
                reenact_canon_ldmks = base_input['body_ldmk'].reshape([-1, 2])

                original_center = original_ldmks[0]
                reenact_canon_center = reenact_canon_ldmks[0]
                reenact_center = reenact_ldmks[0]

                reenact_movement = reenact_ldmks - reenact_canon_ldmks
                model_input['body_ldmk'] = (original_ldmks + reenact_movement).reshape(1, -1)




            if self.mlp_warp_smooth:
                model_input["smoothed_mlp_delta_uvs"] = smoothed_mlp_delta_uvs[indices].cuda()


            # model_input['layer_offset'] = offset
            model_outputs = self.model(model_input, clip_texture=self.conf.get('train.train_texture_only_clip', False))
            for k, v in model_outputs.items():
                try:
                    model_outputs[k] = v.detach()
                except:
                    model_outputs[k] = v
            plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], f'epoch_{str(self.start_epoch)}_test') for i in range(len(model_input['sub_dir']))]

            img_names = reenact_input['img_name'][:,0].cpu().numpy()
            utils.mkdir_ifnotexists(os.path.join(self.eval_dir, model_input['sub_dir'][0]))
            if eval_all:
                for dir in plot_dir:
                    utils.mkdir_ifnotexists(dir)
            img = plt.plot(img_names,
                    model_outputs,
                    ground_truth,
                    plot_dir,
                    self.start_epoch,
                    self.img_res,
                    is_eval=eval_all,
                    first=is_first_batch,
                    save_compare=(img_index % 10)==0,
                    )
            imgs.append(torch.clamp(model_outputs['rgb_image'], 0, 1)[0].detach().cpu().numpy() * 255)

            is_first_batch = False
            del model_outputs, ground_truth

        # compress to video
        imgs = np.stack(imgs)
        torchvision.io.write_video(os.path.join(self.eval_dir, 'test.mp4'), imgs, fps=10, video_codec='h264')

