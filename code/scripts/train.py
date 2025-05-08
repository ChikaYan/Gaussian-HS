import os
from pyhocon import ConfigFactory
import torch

import sys
sys.path.append('./')

import utils.general as utils
import utils.plots as plt

import wandb
from functools import partial
from model.point_avatar_model import PointAvatar
from model.loss import Loss
from model.gaussian import arguments as gs_args 
from pathlib import Path
from torch.profiler import profile, record_function, ProfilerActivity
import torchvision
from model.scheduler import ConstantSchedule, LinearSchedule, ExpSchedule, SequentialSchedule

print = partial(print, flush=True)

import numpy as np


np.random.seed(0)
torch.manual_seed(0)

# torch.autograd.set_detect_anomaly(True)


class TrainRunner():
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

        self.gs_model_args = gs_args.ModelParams(**self.conf.get_config('gs_model', {}))
        self.gs_opt = gs_args.OptimizationParams(**self.conf.get_config('gs_opt', {}))
        self.max_batch = self.conf.get_int('train.max_batch', default='8')
        self.batch_size = min(int(self.conf.get_int('train.max_points_training') / self.conf.get_int('model.point_cloud.n_init_points')), self.max_batch)
        self.nepochs = self.conf.get_int('train.nepochs', kwargs['nepochs'])
        self.exps_folder_name = self.conf.get_string('train.exps_folder')
        self.optimize_expression = self.conf.get_bool('train.optimize_expression')
        self.optimize_pose = self.conf.get_bool('train.optimize_camera')
        self.optimize_ldmk = self.conf.get_bool('train.optimize_ldmk')
        self.subject = self.conf.get_string('dataset.subject_name')
        self.methodname = self.conf.get_string('train.methodname')
        self.log_train_img_every = self.conf.get_int('train.log_train_img_every', 5000)
        self.start_input_opt_iter = self.conf.get_int('train.start_input_opt_iter', 30_000)
        self.train_iter = self.conf.get_int('train.train_iter', 60_000)
        self.gs_sh_up_from_iter = self.conf.get_int('train.gs_sh_up_from_iter', 0)
        self.train_texture_only_from = None

        os.environ['WANDB_DIR'] = os.path.join(self.exps_folder_name)
        wandb.init(
            project=kwargs['wandb_workspace'], 
            name=self.methodname, 
            group=self.subject, 
            config=self.conf, 
            tags=kwargs['wandb_tags'],
            mode=kwargs['wandb_mode'],
            )

        self.optimize_inputs = self.optimize_expression or self.optimize_pose or self.optimize_ldmk
        self.expdir = os.path.join(self.exps_folder_name, self.subject, self.methodname)
        train_split_name = utils.get_split_name(self.conf.get_list('dataset.train.sub_dir'))

        # write wandb id for eval logging
        wandb_id_path = Path(os.path.join(self.expdir, train_split_name)) / 'wandb_id.txt'
        wandb_id_path.parent.mkdir(exist_ok=True, parents=True)
        with wandb_id_path.open('w') as f:
            f.write(wandb.run.id)


        self.eval_dir = os.path.join(self.expdir, train_split_name, 'eval')
        self.train_dir = os.path.join(self.expdir, train_split_name, 'train')

        if kwargs['is_continue']:
            if kwargs['load_path'] != '':
                load_path = kwargs['load_path']
            else:
                load_path = self.train_dir
            if os.path.exists(os.path.join(load_path)):
                is_continue = True
            else:
                is_continue = False
        else:
            is_continue = False

        utils.mkdir_ifnotexists(self.train_dir)
        utils.mkdir_ifnotexists(self.eval_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.train_dir, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_params_subdir = "SchedulerParameters"
        self.gaussian_subdir = "GaussianPly"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.gaussian_subdir))

        if self.optimize_inputs:
            self.optimizer_inputs_subdir = "OptimizerInputs"
            self.input_params_subdir = "InputParameters"

            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_inputs_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.input_params_subdir))

        os.system("""cp -r {0} "{1}" """.format(kwargs['conf_path'], os.path.join(self.train_dir, 'runconf.conf')))
        self.file_backup()

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
                                                                                          use_var_expression=self.conf.get_bool('dataset.use_var_expression', default=False),
                                                                                          use_background=self.use_background,
                                                                                          load_body_ldmk=load_body_ldmk,
                                                                                          is_eval=False,
                                                                                          load_ldmk_drawing=load_ldmk_drawing,
                                                                                          **self.conf.get_config('dataset.train'))
        self.plot_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(data_folder=self.conf.get_string('dataset.data_folder'),
                                                                                         subject_name=self.conf.get_string('dataset.subject_name'),
                                                                                         json_name=self.conf.get_string('dataset.json_name'),
                                                                                         use_background=self.use_background,
                                                                                         load_body_ldmk=load_body_ldmk,
                                                                                         is_eval=True,
                                                                                         load_ldmk_drawing=load_ldmk_drawing,
                                                                                         **self.conf.get_config('dataset.test'),
                                                                                         )
        print('Finish loading data ...')

        self.model = PointAvatar(conf=self.conf.get_config('model'),
                                shape_params=self.train_dataset.shape_params,
                                img_res=self.train_dataset.img_res,
                                canonical_expression=self.train_dataset.mean_expression,
                                canonical_pose=self.conf.get_float('dataset.canonical_pose', default=0.2),
                                use_background=self.use_background,
                                gs_model_args=self.gs_model_args,
                                data_sample = self.train_dataset[0],
                                )
        
        self.update_anchor_correspondence_every_iter = self.conf.get('model.gs_img_model.update_anchor_correspondence_every_iter', None)
        self.prune_outside_anchor_every_iter = self.conf.get('model.gs_img_model.prune_outside_anchor_every_iter', None)
        if self.model.gs_img_model is None:
            self.update_anchor_correspondence_every_iter = None
            self.prune_outside_anchor_every_iter = None
        
        if self.model.lift_deform_model is not None and self.conf.get_bool('model.lift_deform_model.feature_im_init', False):
            # initialize the feature img with first gt in model
            im = self.train_dataset[0][2]['rgb'].cuda()

            eps = 1e-6
            im = torch.clamp(im, eps, 1-eps)

            # # inverse sigmoid
            # im = torch.log(im/(1-im))

            # inverse tanh
            im = torch.atanh(im)

            im = im.reshape([*self.train_dataset.img_res, -1])
            self.model.lift_deform_model.feature_net.init_feature(im)

            # img = self.model.lift_deform_model.feature_net.feature_img.data[:3].permute([1,2,0])
            # img = torch.sigmoid(img)


        self.lift_deform_warm_up_iter = self.conf.get_int('model.lift_deform_model.warm_up', 0)
        if self.lift_deform_warm_up_iter > 0:
            self.model.lift_deform_model_warm_up = True 

        
        self.model.pc.training_setup(self.gs_opt)
        # self.model.anchor.training_setup(self.gs_opt)
        self.bg_layer_deform_warm_up = self.conf.get_int('model.bg_layer_model.deform_warm_up', 3000)

        self._init_dataloader()
        if torch.cuda.is_available():
            self.model.cuda()

        self.loss = Loss(**self.conf.get_config('loss'), var_expression=self.train_dataset.var_expression)

        self.vgg_loss_schedule = LinearSchedule(
            self.conf.get('loss.vgg_init_weight', self.loss.vgg_feature_weight),
            self.loss.vgg_feature_weight,
            self.conf.get_int('loss.vgg_step', 0),
        )

        self.vgg_loss_warm_up = self.conf.get('loss.vgg_loss_warm_up', 0)
        if self.vgg_loss_warm_up > 0:
            self.vgg_loss_schedule = SequentialSchedule([
                ConstantSchedule(0, 0, self.vgg_loss_warm_up),
                self.vgg_loss_schedule
            ])
            
        self.loss.vgg_feature_weight = self.vgg_loss_schedule(0)

        self.lr = self.conf.get_float('train.learning_rate')
        self.lr_bg = self.conf.get_float('train.learning_rate_bg', self.lr)
        self.lr_lift_deform = self.conf.get_float('train.learning_rate_lift_deform', self.lr)

        # self.optimizer = torch.optim.Adam([
        #     {'name': 'deformer_network', 'params': list(self.model.deformer_network.parameters()), 'lr': self.lr},
        # ], lr=self.lr)

        # build parameter dict for optimizer
        param_dict = [
            # {'name': 'FLAME', 'params': list(self.model.FLAMEServer.parameters()), 'lr': self.lr},
            {'name': 'deformer_network', 'params': list(self.model.deformer_network.parameters()), 'lr': self.lr},
        ]
        if self.model.bg_layer_network is not None:
            param_dict.append(
                {'name': 'bg_layer_network', 'params': list(self.model.bg_layer_network.parameters()), 'lr': self.lr_bg},
            )
            # param_dict += [
            #     {'name': 'bg_layer_network', 'params': list(self.model.bg_layer_network.deform_net_2d.parameters()), 'lr': 1e-5},
            #     {'name': 'bg_layer_network', 'params': list(self.model.bg_layer_network.feature_img_fn.parameters()), 'lr': self.lr_bg},
            #     {'name': 'bg_layer_network', 'params': list(self.model.bg_layer_network.network.parameters()), 'lr': self.lr_bg},
            # ]

        if self.model.lift_deform_model is not None:
            param_dict.append(
                {'name': 'lift_deform_model', 'params': list(self.model.lift_deform_model.parameters()), 'lr': self.lr_lift_deform},
            )

        if self.model.gs_img_model is not None:
            # param_dict.append(
            #     {'name': 'gs_img_model', 'params': list(self.model.gs_img_model.parameters()), 'lr': self.lr_bg},
            # )

            # param_dict.append(
            #     {'name': 'deformer_network_anchor', 'params': list(self.model.deformer_network_anchor.parameters()), 'lr': self.lr_bg},
            # )
            self.optimizer_anchor = torch.optim.Adam(list(self.model.gs_img_model.parameters()), lr=self.lr_bg)
            self.train_texture_only_from = self.model.hide_anchor_from_iter
            self.hide_anchor_allow_gs_opacity_update = self.conf.get('model.gs_img_model.hide_anchor_allow_gs_opacity_update', False)
            self.hide_anchor_allow_gs_color_update = self.conf.get('model.gs_img_model.hide_anchor_allow_gs_color_update', False)
        else:
            self.optimizer_anchor = None

        if self.train_texture_only_from is None:
            self.train_texture_only_from = np.inf


        self.optimizer = torch.optim.Adam(param_dict, lr=self.lr)

        self.sched_milestones = self.conf.get_list('train.sched_milestones', default=[])
        self.sched_factor = self.conf.get_float('train.sched_factor', default=0.0)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.sched_milestones, gamma=self.sched_factor)
        # self.upsample_freq = self.conf.get_int('train.upsample_freq', default=5)
        # settings for input parameter optimization
        if self.optimize_inputs:
            num_training_frames = len(self.train_dataset)
            param = []
            if self.optimize_expression:
                init_expression = torch.cat((self.train_dataset.data["expressions"], torch.randn(self.train_dataset.data["expressions"].shape[0], max(self.model.deformer_network.num_exp - 50, 0)).float()), dim=1)
                self.expression = torch.nn.Embedding(num_training_frames, self.model.deformer_network.num_exp, _weight=init_expression, sparse=True).cuda()
                param += list(self.expression.parameters())

            if self.optimize_pose:
                self.flame_pose = torch.nn.Embedding(num_training_frames, 15, _weight=self.train_dataset.data["flame_pose"], sparse=True).cuda()
                self.camera_pose = torch.nn.Embedding(num_training_frames, 3, _weight=torch.zeros_like(self.train_dataset.data["world_mats"][:, :3, 3]), sparse=True).cuda()
                param += list(self.flame_pose.parameters()) + list(self.camera_pose.parameters())

                # data = torch.load('/home/tw554/pointavatar/data/experiments/soubhik/rerun/train/train/checkpoints/InputParameters/latest.pth')
                # self.flame_pose.load_state_dict(data["flame_pose_state_dict"])
                # self.camera_pose.load_state_dict(data["camera_pose_state_dict"])
            if self.optimize_ldmk:
                self.ldmks_opted = torch.nn.Embedding(num_training_frames, self.train_dataset.data["body_ldmks"].shape[-1],
                                                     _weight=self.train_dataset.data["body_ldmks"],
                                                     sparse=True).cuda()
                param += list(self.ldmks_opted.parameters())

            self.optimizer_cam = torch.optim.SparseAdam(param, self.conf.get_float('train.learning_rate_cam'))


        self.use_per_frame_noise = self.conf.get_bool('model.bg_layer_model.args.use_per_frame_noise', False)

        if self.use_per_frame_noise:
            noise_dim = self.conf.get_int('model.bg_layer_model.args.frame_noise_dim')
            self.per_frame_noises = torch.nn.Embedding(num_training_frames, noise_dim,
                                                     _weight=torch.zeros(num_training_frames, noise_dim),
                                                     sparse=True).cuda()
            
            self.optimizer_noise = torch.optim.SparseAdam(list(self.per_frame_noises.parameters()), self.conf.get_float('train.learning_rate_noise'))
        
        self.start_noise_opt_ep = self.conf.get_float('train.start_noise_opt_ep', 10)

        self.start_epoch = 0
        if is_continue:
            raise NotImplementedError()
            old_checkpnts_dir = os.path.join(load_path, 'checkpoints')
            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.start_epoch = saved_model_state['epoch']
            n_points = saved_model_state["model_state_dict"]['pc.points'].shape[0]

            batch_size = min(int(self.conf.get_int('train.max_points_training') / n_points), self.max_batch)
            if self.batch_size != batch_size:
                self.batch_size = batch_size
                self._init_dataloader()
            self.model.pc.init(n_points)
            self.model.pc = self.model.pc.cuda()

            self.model.load_state_dict(saved_model_state["model_state_dict"], strict=False)

            self.model.raster_settings.radius = saved_model_state['radius']

            self.optimizer = torch.optim.Adam([
                {'params': list(self.model.parameters())},
            ], lr=self.lr)

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

            if self.optimize_inputs:
                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.optimizer_inputs_subdir, str(kwargs['checkpoint']) + ".pth"))
                try:
                    self.optimizer_cam.load_state_dict(data["optimizer_cam_state_dict"])
                except:
                    print("input and camera optimizer parameter group doesn't match")
                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.input_params_subdir, str(kwargs['checkpoint']) + ".pth"))
                try:
                    if self.optimize_expression:
                        self.expression.load_state_dict(data["expression_state_dict"])
                    if self.optimize_pose:
                        self.flame_pose.load_state_dict(data["flame_pose_state_dict"])
                        self.camera_pose.load_state_dict(data["camera_pose_state_dict"])
                except:
                    print("expression or pose parameter group doesn't match")

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn,
                                                            num_workers=4,
                                                            )
        self.n_batches = len(self.train_dataloader)
        self.img_res = self.plot_dataset.img_res
        self.plot_freq = self.conf.get_int('train.plot_freq_iter', 5000)
        if self.plot_freq == 1:
            # change from ep to iter
            self.plot_freq = 5000
        self.save_freq = self.conf.get_int('train.save_freq_iter', 10000)

        self.GT_lbs_milestones = self.conf.get_list('train.GT_lbs_milestones', default=[])
        self.GT_lbs_factor = self.conf.get_float('train.GT_lbs_factor', default=0.5)
        for acc in self.GT_lbs_milestones:
            if self.start_epoch > acc:
                self.loss.lbs_weight = self.loss.lbs_weight * self.GT_lbs_factor
        # if len(self.GT_lbs_milestones) > 0 and self.start_epoch >= self.GT_lbs_milestones[-1]:
        #    self.loss.lbs_weight = 0.

    def _init_dataloader(self):
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn,
                                                            num_workers=4,
                                                            )
        self.plot_dataloader = torch.utils.data.DataLoader(self.plot_dataset,
                                                           batch_size=min(self.batch_size, 10),
                                                        #    batch_size=6,
                                                           shuffle=False,
                                                           collate_fn=self.plot_dataset.collate_fn,
                                                           num_workers=4,
                                                           )
        self.n_batches = len(self.train_dataloader)
    def save_checkpoints(self, epoch, only_latest=False):
        if not only_latest:
            torch.save(
                {"epoch": epoch, "radius": self.model.raster_settings.radius,
                 "model_state_dict": self.model.state_dict()},
                os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
                os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
            self.model.pc.save_ply(os.path.join(self.checkpoints_path, self.gaussian_subdir, str(epoch) + ".ply"))

        torch.save(
            {"epoch": epoch, "radius": self.model.raster_settings.radius,
                 "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))
        self.model.pc.save_ply(os.path.join(self.checkpoints_path, self.gaussian_subdir, "latest.ply"))

        if self.optimize_inputs:
            if not only_latest:
                torch.save(
                    {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                    os.path.join(self.checkpoints_path, self.optimizer_inputs_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_inputs_subdir, "latest.pth"))\

            dict_to_save = {}
            dict_to_save["epoch"] = epoch
            if self.optimize_expression:
                dict_to_save["expression_state_dict"] = self.expression.state_dict()
            if self.optimize_pose:
                dict_to_save["flame_pose_state_dict"] = self.flame_pose.state_dict()
                dict_to_save["camera_pose_state_dict"] = self.camera_pose.state_dict()
            if self.optimize_ldmk:
                dict_to_save["ldmks_state_dict"] = self.ldmks_opted.state_dict()
            if self.use_per_frame_noise:
                dict_to_save["frame_noise_dict"] = self.per_frame_noises.state_dict()
            if not only_latest:
                torch.save(dict_to_save, os.path.join(self.checkpoints_path, self.input_params_subdir, str(epoch) + ".pth"))
            torch.save(dict_to_save, os.path.join(self.checkpoints_path, self.input_params_subdir, "latest.pth"))


    def file_backup(self):
        from shutil import copyfile
        dir_lis = [
            './model', 
            './scripts', 
            './utils', 
            './flame', 
            './datasets', 
            './model/layer', 
            './model/gaussian', 
            './model/gaussian/arguments', 
            './model/gaussian/gaussian_renderer'
            ]
        os.makedirs(os.path.join(self.train_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.train_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        # copyfile(self.conf_path, os.path.join(self.train_dir, 'recording', 'config.conf'))

    def run(self):
        acc_loss = {}
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        iteration = 0

        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:

        # for epoch in range(self.start_epoch, self.nepochs + 1):
        while iteration < self.train_iter:
            start_time.record()

            # # Prunning
            # if epoch != self.start_epoch and self.model.raster_settings.radius >= 0.006:
            #     self.model.pc.prune(self.model.visible_points)
            #     self.optimizer = torch.optim.Adam([
            #         {'params': list(self.model.parameters())},
            #     ], lr=self.lr)
            # # Upsampling
            # if epoch % self.upsample_freq == 0:
            #     if epoch != 0 and self.model.pc.points.shape[0] <= self.model.pc.max_points / 2:
            #         self.upsample_points()
            #         batch_size = min(int(self.conf.get_int('train.max_points_training') / self.model.pc.points.shape[0]), self.max_batch)
            #         if batch_size != self.batch_size:
            #             self.batch_size = batch_size
            #             self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
            #                                                                 batch_size=self.batch_size,
            #                                                                 shuffle=True,
            #                                                                 collate_fn=self.train_dataset.collate_fn,
            #                                                                 num_workers=4,
            #                                                                 )
            #             self.n_batches = len(self.train_dataloader)

            # # re-init visible point tensor each epoch
            # self.model.visible_points = torch.zeros(self.model.pc.points.shape[0]).bool().cuda()

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                iteration += 1

                if iteration % self.save_freq == 0 and iteration != 0:
                    self.save_checkpoints(iteration, only_latest=True)
                    
                if iteration in self.GT_lbs_milestones:
                    self.loss.lbs_weight = self.loss.lbs_weight * self.GT_lbs_factor
                if len(self.GT_lbs_milestones) > 0 and iteration >= self.GT_lbs_milestones[-1]:
                    self.loss.lbs_weight = 0.


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

                if self.optimize_inputs:
                    if self.optimize_expression:
                        model_input['expression'] = self.expression(model_input["idx"]).squeeze(1)
                    if self.optimize_pose:
                        model_input['flame_pose'] = self.flame_pose(model_input["idx"]).squeeze(1)
                        # model_input['cam_pose'][:, :3, 3] = self.camera_pose(model_input["idx"]).squeeze(1)
                        model_input['cam_pose_offset'] = self.camera_pose(model_input["idx"]).squeeze(1)
                    if self.optimize_ldmk:
                        # import pdb; pdb.set_trace()
                        model_input['body_ldmk'] = self.ldmks_opted(model_input["idx"]).squeeze(1)

                if self.use_per_frame_noise:
                    model_input['frame_noise'] = self.per_frame_noises(model_input["idx"]).squeeze(1)

                self.model.pc.update_learning_rate(iteration)
                # self.model.anchor.update_learning_rate(iteration)
                # Every 1000 its we increase the levels of SH up to a maximum degree
                if iteration > self.gs_sh_up_from_iter and iteration % 1000 == 0:
                    self.model.pc.oneupSHdegree()

                if iteration == self.train_texture_only_from+1:
                    # freeze anchors
                    self.model.gs_img_model.anchor_xyz.requires_grad = False
                    self.model.gs_img_model.anchor_opacity.requires_grad = False
                    self.model.gs_img_model.anchor_rgb.requires_grad = False
                    self.model.gs_img_model.anchor_scaling.requires_grad = False

                    if self.hide_anchor_allow_gs_opacity_update or self.hide_anchor_allow_gs_color_update:
                        self.model.pc._xyz.requires_grad = False
                        self.model.pc._scaling.requires_grad = False
                        self.model.pc._rotation.requires_grad = False
                        self.model.pc._opacity.requires_grad = self.hide_anchor_allow_gs_opacity_update
                        self.model.pc._features_dc.requires_grad = self.hide_anchor_allow_gs_color_update
                        self.model.pc._features_rest.requires_grad = self.hide_anchor_allow_gs_color_update


                if self.model.gs_img_model is not None:
                    if iteration == self.model.gs_img_model.anchor_init_iter:
                        anchor_init_inputs = self.train_dataset.get_canonical_frame_inputs()
                        for k, v in anchor_init_inputs.items():
                            try:
                                anchor_init_inputs[k] = v.cuda()
                            except:
                                anchor_init_inputs[k] = v
                        with torch.no_grad():
                            self.model(
                                anchor_init_inputs, 
                                initing_anchor=True,
                            )

                    if iteration > self.model.gs_img_model.anchor_init_iter and \
                    self.update_anchor_correspondence_every_iter is not None and \
                    iteration % self.update_anchor_correspondence_every_iter == 0 and \
                    iteration <= self.train_texture_only_from:
                        anchor_init_inputs = self.train_dataset.get_canonical_frame_inputs()
                        for k, v in anchor_init_inputs.items():
                            try:
                                anchor_init_inputs[k] = v.cuda()
                            except:
                                anchor_init_inputs[k] = v
                        with torch.no_grad():
                            self.model(
                                anchor_init_inputs, 
                                updating_anchor_correspondence=True,
                            )

                    if iteration > self.model.gs_img_model.anchor_init_iter and \
                    self.prune_outside_anchor_every_iter is not None and \
                    iteration % self.prune_outside_anchor_every_iter == 0 and \
                    iteration <= self.train_texture_only_from:
                        anchor_init_inputs = self.train_dataset.get_canonical_frame_inputs()
                        for k, v in anchor_init_inputs.items():
                            try:
                                anchor_init_inputs[k] = v.cuda()
                            except:
                                anchor_init_inputs[k] = v
                        with torch.no_grad():
                            self.model(
                                anchor_init_inputs, 
                                pruning_outside_anchor=True,
                            )

                    



                # with record_function("Model inference"):
                model_outputs = self.model(
                    model_input, 
                    train_log=iteration % self.log_train_img_every == 0, 
                    iter=iteration,
                    # need_separate_gs_alpha=self.loss.head_mask_weight > 0.,
                    clip_texture=(iteration > self.train_texture_only_from) and \
                        self.conf.get('train.train_texture_only_clip', False)
                    )
                

                if self.model.gs_img_model is not None:
                    if iteration < self.model.gs_img_model.anchor_init_iter:
                        self.loss.head_mask_weight = 0.
                    elif iteration == self.model.gs_img_model.anchor_init_iter:
                        self.loss.head_mask_weight = self.conf.get('loss.head_mask_weight', 0.)

                # with record_function("Loss computation"):
                loss_output = self.loss(model_outputs, ground_truth)

                loss = loss_output['loss']

                if self.optimizer_anchor is not None:
                    self.optimizer_anchor.zero_grad()
                if iteration <= self.train_texture_only_from:
                    self.optimizer.zero_grad()
                    # if self.model.get_anchor_visibility_mask:
                    #     self.model.pc.optimizer.zero_grad(set_to_none = True) 
                    if self.optimize_inputs and iteration > self.start_input_opt_iter:
                        self.optimizer_cam.zero_grad()
                    if self.use_per_frame_noise and iteration > self.start_noise_opt_ep:
                        raise NotImplementedError()
                        self.optimizer_noise.zero_grad()
                    
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                # torch.nn.utils.clip_grad_norm_(self.model.pc.points, 1.)
                if self.optimizer_anchor is not None:
                    self.optimizer_anchor.step()
                if iteration <= self.train_texture_only_from:
                    self.optimizer.step()
                    if self.model.get_anchor_visibility_mask:
                        self.optimizer.zero_grad()
                        if self.optimizer_anchor is not None:
                            self.optimizer_anchor.zero_grad()

                    if self.optimize_inputs and iteration > self.start_input_opt_iter:
                        self.optimizer_cam.step()
                    if self.use_per_frame_noise and iteration > self.start_noise_opt_ep:
                        raise NotImplementedError()
                        self.optimizer_noise.step()

                for k, v in loss_output.items():
                    loss_output[k] = v.detach().item()
                    if k not in acc_loss:
                        acc_loss[k] = [v]
                    else:
                        acc_loss[k].append(v)

                if iteration > self.bg_layer_deform_warm_up and self.model.bg_layer_network is not None:
                    self.model.bg_layer_network.deform_warm_up = False

                if iteration > self.lift_deform_warm_up_iter:
                    self.model.lift_deform_model_warm_up = False 


                self.loss.vgg_feature_weight = self.vgg_loss_schedule(iteration)

                # if iteration > self.vgg_loss_warm_up :
                #     self.loss.vgg_feature_weight = self.conf.get_float('loss.vgg_feature_weight', 0.)

                if iteration > self.conf.get('loss.head_mask_loss_util', np.inf):
                    self.loss.head_mask_weight = 0.

                if iteration > self.conf.get('loss.anchor_opacity_loss_util', np.inf):
                    self.loss.anchor_opacity_loss_weight = 0.

                if iteration > self.conf.get('loss.texture_neural_uv_delta_loss_until', np.inf):
                    self.loss.texture_neural_uv_delta_loss_weight = 0.

                ####### gs density control #######
                with torch.no_grad():
                    viewspace_point_tensor = model_outputs['viewspace_point_tensor']
                    visibility_filter = model_outputs['visibility_filter']
                    radii = model_outputs['radii']
                    gaussians = self.model.pc

                    grad_thresh = self.gs_opt.densify_grad_threshold
                    if iteration < self.vgg_loss_warm_up and self.conf.get_float('loss.vgg_feature_weight', 0.) > 0.:
                        # reduce grad threshold if vgg loss is set, but current iter is in no vgg warm up period
                        grad_thresh = self.gs_opt.densify_grad_threshold_vgg_warmup

                    # mask out extra gaussian
                    N_gs = gaussians.get_xyz.shape[0]
                    # viewspace_point_tensor = [x[:N_gs] for x in viewspace_point_tensor]
                    visibility_filter = visibility_filter[:N_gs]
                    radii = radii[:N_gs]

                    # Densification
                    if iteration < self.gs_opt.densify_until_iter:
                        # Keep track of max radii in image-space for pruning
                        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                        # gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                        viewspace_point_tensor_grad = torch.stack(list(map(lambda x: x.grad[:N_gs], viewspace_point_tensor))) #.mean(axis=0)
                        gaussians.add_densification_stats_grad(viewspace_point_tensor_grad, visibility_filter)

                        if iteration > self.gs_opt.densify_from_iter and iteration % self.gs_opt.densification_interval == 0:
                            size_threshold = 20 if iteration > self.gs_opt.opacity_reset_interval else None
                            CAMERA_EXTEND = 1
                            gaussians.densify_and_prune(grad_thresh, 0.005, CAMERA_EXTEND, size_threshold)
                            print(f"new points: {gaussians._xyz.shape[0]}")
                            # with (Path(dataset.model_path) / 'point_number.txt').open('a') as f:
                            #     f.write(f"Iter [{iteration}], Point: {gaussians._xyz.shape[0]}\n")

                        WHITE_BACKGROUND = True
                        if iteration % self.gs_opt.opacity_reset_interval == 0 or (WHITE_BACKGROUND and iteration == self.gs_opt.densify_from_iter):
                            gaussians.reset_opacity()
                ####### gs density control end #######
                    if iteration <= self.train_texture_only_from or self.hide_anchor_allow_gs_opacity_update or self.hide_anchor_allow_gs_color_update:
                        self.model.pc.optimizer.step()
                        self.model.pc.optimizer.zero_grad(set_to_none = True)

                    # self.model.anchor.optimizer.step()
                    # self.model.anchor.optimizer.zero_grad(set_to_none = True)

                # acc_loss['visible_percentage'] = (torch.sum(self.model.visible_points)/self.model.pc.points.shape[0]).unsqueeze(0)
                if iteration % 50 == 0:
                    for k, v in acc_loss.items():
                        acc_loss[k] = sum(v) / len(v)
                    print_str = '{0} [{1}] ({2}/{3}): '.format(self.methodname, data_index, iteration, self.train_iter)
                    for k, v in acc_loss.items():
                        print_str += '{}: {:.3g} '.format(k, v)
                    print(print_str)
                    acc_loss['num_points'] = self.model.pc.points.shape[0]
                    acc_loss['radius'] = self.model.raster_settings.radius

                    if model_outputs['xyz_lift_shift'] is not None:
                        acc_loss['lift_deform/xyz_max'] = model_outputs['xyz_lift_shift'].max().item()
                        acc_loss['lift_deform/xyz_min'] = model_outputs['xyz_lift_shift'].min().item()
                        acc_loss['lift_deform/xyz_mean'] = model_outputs['xyz_lift_shift'].abs().mean().item()

                    if model_outputs['color_lift_shift'] is not None:
                        acc_loss['lift_deform/color_max'] = model_outputs['color_lift_shift'].max().item()
                        acc_loss['lift_deform/color_min'] = model_outputs['color_lift_shift'].min().item()
                        acc_loss['lift_deform/color_mean'] = model_outputs['color_lift_shift'].abs().mean().item()

                    acc_loss['lr'] = self.scheduler.get_last_lr()[0]
                    wandb.log(acc_loss, step=iteration)
                    acc_loss = {}

                if iteration % self.log_train_img_every == 0:
                    imgs_log_dir = Path(self.eval_dir)/ 'train_log'
                    imgs_log_dir.mkdir(exist_ok=True, parents=True)
                    torchvision.utils.save_image(model_outputs['rgb_image'].permute([0,3,1,2]), 
                                                 str(imgs_log_dir / f"rgb_image_{iteration:05d}.png"))
                    
                    if 'layer_bg' in model_outputs:
                        torchvision.utils.save_image(model_outputs['layer_bg'].reshape(model_outputs['rgb_image'].shape).permute([0,3,1,2]), 
                                                    str(imgs_log_dir / f"layer_bg_{iteration:05d}.png"))
                        
                    if 'render_img_gs' in model_outputs:
                        torchvision.utils.save_image(model_outputs['render_img_gs'][None,...], 
                                                    str(imgs_log_dir / f"im_gs_{iteration:05d}.png"))
                    if 'render_img_no_lift' in model_outputs:
                        torchvision.utils.save_image(model_outputs['render_img_no_lift'][None,...], 
                                                    str(imgs_log_dir / f"no_lift_{iteration:05d}.png"))
                        
                    if 'plain_texture' in model_outputs:
                        torchvision.utils.save_image(model_outputs['plain_texture'][None,...], 
                                                    str(imgs_log_dir / f"plain_texture_{iteration:05d}.png"))
                        

                ##### Validation #####
                if (iteration % self.plot_freq == 0):
                    self.model.eval()
                    if self.optimize_inputs:
                        if self.optimize_expression:
                            self.expression.eval()
                        if self.optimize_pose:
                            self.flame_pose.eval()
                            self.camera_pose.eval()
                        if self.optimize_ldmk:
                            self.ldmks_opted.eval()
                        if self.use_per_frame_noise:
                            self.per_frame_noises.eval()
                    eval_iterator = iter(self.plot_dataloader)
                    start_time.record()
                    for batch_index in range(len(self.plot_dataloader)):
                        indices, model_input, ground_truth = next(eval_iterator)
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

                        model_outputs = self.model(model_input, iter=iteration)
                        for k, v in model_outputs.items():
                            try:
                                model_outputs[k] = v.detach()
                            except:
                                model_outputs[k] = v
                        plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'iter_'+str(iteration)) for i in range(len(model_input['sub_dir']))]
                        img_names = model_input['img_name'][:, 0].cpu().numpy()
                        print("Plotting images: {}".format(img_names))
                        utils.mkdir_ifnotexists(os.path.join(self.eval_dir, model_input['sub_dir'][0]))

                        plt.plot(img_names,
                                model_outputs,
                                ground_truth,
                                plot_dir,
                                iteration,
                                self.img_res,
                                is_eval=False,
                                first=(batch_index==0),
                                )
                        del model_outputs, ground_truth

                    end_time.record()
                    torch.cuda.synchronize()
                    print("Plot time per image: {} ms".format(start_time.elapsed_time(end_time) / len(self.plot_dataset)))
                    self.model.train()
                    if self.optimize_inputs:
                        if self.optimize_expression:
                            self.expression.train()
                        if self.optimize_pose:
                            self.flame_pose.train()
                            self.camera_pose.train()
                        if self.optimize_ldmk:
                            self.ldmks_opted.train()
                        if self.use_per_frame_noise:
                            self.per_frame_noises.train()

                ##### Validation End #####


                if iteration > self.train_iter:
                    break


                self.scheduler.step() # need to step for every iteration, since we are switching to iter based decay

            end_time.record()
            torch.cuda.synchronize()
            wandb.log({"timing_epoch": start_time.elapsed_time(end_time)}, step=iteration)
            print("Epoch time: {} s".format(start_time.elapsed_time(end_time)/1000))

        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
        self.save_checkpoints(iteration + 1, only_latest=True)



