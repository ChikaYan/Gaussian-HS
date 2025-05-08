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


print = partial(print, flush=True)
class DistillRunner():
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)
        self.conf = kwargs['conf']
        default_methodname = Path(kwargs['conf_path']).stem
        if Path(kwargs['conf_path']).parent.name != 'confs':
            default_methodname = f"{Path(kwargs['conf_path']).parent.name}/{Path(kwargs['conf_path']).stem}"
        self.conf['train']['methodname'] = self.conf['train'].get_string('methodname', default_methodname)

        if kwargs['subject'] is not None:
            self.conf['dataset']['subject_name'] = kwargs['subject']

        self.exps_folder_name = self.conf.get_string('train.exps_folder')
        self.subject = self.conf.get_string('dataset.subject_name')
        self.methodname = self.conf.get_string('train.methodname')

        self.expdir = os.path.join(self.exps_folder_name, self.subject, self.methodname)
        train_split_name = utils.get_split_name(self.conf.get_list('dataset.train.sub_dir'))


        self.eval_dir = os.path.join(self.expdir, train_split_name, 'eval')
        self.train_dir = os.path.join(self.expdir, train_split_name, 'train')

        if kwargs['load_path'] != '':
            load_path = kwargs['load_path']
        else:
            load_path = self.train_dir
        assert os.path.exists(load_path), load_path


        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        self.use_background = self.conf.get_bool('dataset.use_background', default=False)

        load_body_ldmk = True
        load_ldmk_drawing = False
        # load_body_ldmk = self.conf.get_int('model.bg_layer_model.args.ldmk_multires', -2) > -2
        # load_ldmk_drawing = self.conf.get('model.bg_layer_model.args.parser_type', 'mlp').startswith('draw_')
        # load_ldmk_drawing = load_ldmk_drawing or self.conf.get('model.bg_layer_model.args.deform_type', 'mlp').startswith('draw_')
        # load_ldmk_drawing = load_ldmk_drawing or self.conf.get('model.lift_deform_model.args.deform_type', 'mlp').startswith('draw_')


        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(data_folder=self.conf.get_string('dataset.data_folder'),
                                                                                          subject_name=self.conf.get_string('dataset.subject_name'),
                                                                                          json_name=self.conf.get_string('dataset.json_name'),
                                                                                          use_mean_expression=self.conf.get_bool('dataset.use_mean_expression', default=False),
                                                                                          use_background=self.use_background,
                                                                                          is_eval=False,
                                                                                          only_json=False,
                                                                                          load_body_ldmk=load_body_ldmk,
                                                                                          load_ldmk_drawing=load_ldmk_drawing,
                                                                                          **self.conf.get_config('dataset.train'))
        
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=1,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn,
                                                            num_workers=4,
                                                            )


        print('Finish loading data ...')

        self.gs_model_args = gs_args.ModelParams(**self.conf.get_config('gs_model', {}))
        self.model = PointAvatar(conf=self.conf.get_config('model'),
                                shape_params=self.train_dataset.shape_params,
                                img_res=self.train_dataset.img_res,
                                canonical_expression=self.train_dataset.mean_expression,
                                canonical_pose=self.conf.get_float(
                                    'dataset.canonical_pose',
                                    default=0.2),
                                use_background=self.use_background,
                                gs_model_args=self.gs_model_args,
                                )
        if torch.cuda.is_available():
            self.model.cuda()
        old_checkpnts_dir = os.path.join(load_path, 'checkpoints')
        self.checkpoints_path = old_checkpnts_dir
        assert os.path.exists(old_checkpnts_dir)
        
        self.model_ckpt_path = os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth")
        saved_model_state = torch.load(self.model_ckpt_path)

        self.model.pc.load_ply(os.path.join(old_checkpnts_dir, 'GaussianPly', str(kwargs['checkpoint']) + ".ply"))

        self.model.raster_settings.radius = saved_model_state['radius']

        self.model.load_state_dict(saved_model_state["model_state_dict"]) #, strict=False)
        if self.model.gs_img_model is not None:
            self.model.gs_img_model.update_anchor_mask()


        self.start_iter= saved_model_state['epoch']

        
        self.train_iter = self.conf.get('distill.train_iter')
        
        from model.loss import Loss
        self.loss = Loss(**self.conf.get('loss', {}))

        self.img_res = self.train_dataset.img_res


    def run(self):


        iteration = 0

        distill_inputs = self.train_dataset.get_canonical_frame_inputs()
        for k, v in distill_inputs.items():
            try:
                distill_inputs[k] = v.cuda()
            except:
                distill_inputs[k] = v
        self.model(distill_inputs, distill_anchor_model=True)

        self.model.gs_img_model.feature_img.feature_img.requires_grad_(True)
        self.optimizer = torch.optim.Adam(list(self.model.gs_img_model.feature_img.parameters()), 
                                          lr=self.conf.get('distill.learning_rate', self.conf.get('train.learning_rate_bg'))
                                          )

        while iteration < self.train_iter:
            for data_index, (indices, model_input, ground_truth) in tqdm(enumerate(self.train_dataloader),total=self.train_iter):
                iteration += 1

                for k, v in model_input.items():
                    try:
                        model_input[k] = v.cuda()
                    except:
                        model_input[k] = v
                for k, v in ground_truth.items():
                    try:
                        ground_truth[k] = v.cuda()
                    except:
                        ground_truth[k] = v

                model_outputs = self.model(
                    model_input, 
                    iter=iteration+self.start_iter,
                    clip_texture=True,
                    )

                loss_output = self.loss(model_outputs, ground_truth)

                loss = loss_output['loss']

                self.optimizer.zero_grad()  
                loss.backward()
                self.optimizer.step()

                if iteration % 50 == 0:
                    print(f"MSE: {loss_output['rgb_l2_loss'].item()}")


                if iteration > self.train_iter:
                    break

        # save texture
        torch.save(self.model.gs_img_model.feature_img.feature_img.data, str(Path(self.train_dir) / 'checkpoints' / 'ModelParameters' / 'distilled_feature_img.pth'))


