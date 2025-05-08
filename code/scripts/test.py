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
class TestRunner():
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)
        self.conf = kwargs['conf']
        self.conf.put('dataset.test.subsample', 1)
        self.conf.put('dataset.train.load_images', False)
        self.conf.put('dataset.test.load_images', False)
        default_methodname = Path(kwargs['conf_path']).stem
        if Path(kwargs['conf_path']).parent.name != 'confs':
            default_methodname = f"{Path(kwargs['conf_path']).parent.name}/{Path(kwargs['conf_path']).stem}"
        self.conf['train']['methodname'] = self.conf['train'].get_string('methodname', default_methodname)

        if kwargs['subject'] is not None:
            self.conf['dataset']['subject_name'] = kwargs['subject']

        self.quick_eval = kwargs['quick_eval']
        if self.quick_eval:
            self.conf.put('dataset.test.subsample', 5)

        self.exps_folder_name = self.conf.get_string('train.exps_folder')
        self.subject = self.conf.get_string('dataset.subject_name')
        self.methodname = self.conf.get_string('train.methodname')

        self.expdir = os.path.join(self.exps_folder_name, self.subject, self.methodname)
        train_split_name = utils.get_split_name(self.conf.get_list('dataset.train.sub_dir'))

        self.no_metrics = kwargs['no_metrics']
        self.apply_extra_euc = kwargs['apply_extra_euc']
        self.mlp_warp_smooth = kwargs['mlp_warp_smooth']
        self.smooth_test_opt_render = kwargs['smooth_test_opt_render']

        # read wandb id
        self.wandb_debug_log = self.conf.get_bool('test.wandb_debug_log', False)
        wandb_id_path = Path(self.expdir) / train_split_name / 'wandb_id.txt'
        if self.wandb_debug_log:
            wandb.init(
                project=kwargs['wandb_workspace'], 
                name="[test]"+self.methodname, 
                group=self.subject, 
                config=self.conf, 
                mode=kwargs['wandb_mode'],
                tags=kwargs['wandb_tags'],
            )
            self.log_wandb = True
        else:
            if wandb_id_path.exists():
                with wandb_id_path.open('r') as f:
                    wandb_id = f.read()
                    wandb.init(
                        project=kwargs['wandb_workspace'],
                        id=wandb_id,
                        resume='must',
                        mode=kwargs['wandb_mode'],
                    )

                self.log_wandb = True
            else:
                self.log_wandb = False

        self.eval_dir = os.path.join(self.expdir, train_split_name, 'eval')
        if self.quick_eval:
            self.eval_dir = os.path.join(self.expdir, train_split_name, 'eval_quick')

        self.simulate_fast_anchor = kwargs.get('run_fast_test', False)
        if self.simulate_fast_anchor:
            self.eval_dir += '_simulate_fast'

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
                                                                                        #   only_json=not self.simulate_fast_anchor,
                                                                                          load_body_ldmk=load_body_ldmk,
                                                                                          load_ldmk_drawing=load_ldmk_drawing,
                                                                                          **self.conf.get_config('dataset.train'))

        self.plot_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(data_folder=self.conf.get_string('dataset.data_folder'),
                                                                                         subject_name=self.conf.get_string('dataset.subject_name'),
                                                                                         json_name=self.conf.get_string('dataset.json_name'),
                                                                                         only_json=kwargs['only_json'],
                                                                                         use_background=self.use_background,
                                                                                         is_eval=True,
                                                                                         load_body_ldmk=load_body_ldmk,
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
        if self.model.bg_layer_network is not None:
            self.model.bg_layer_network.deform_warm_up = False
        if self.model.gs_img_model is not None:
            self.model.gs_img_model.update_anchor_mask()
            
        self.model.conf['detach_pose_input'] = True
        self.model.conf['detach_cam_input'] = True


        self.start_epoch = saved_model_state['epoch']
        self.optimize_expression = self.conf.get_bool('train.optimize_expression')
        self.optimize_pose = self.conf.get_bool('train.optimize_camera')
        self.optimize_ldmk = self.conf.get_bool('train.optimize_ldmk', False)
        self.optimize_layer_offset = self.conf.get_bool('test.optimize_layer_offset', False)
        self.use_per_frame_noise = self.conf.get_bool('model.bg_layer_model.args.use_per_frame_noise', False)
        self.optimize_warp_net = self.conf.get_bool('test.optimize_warp_net', False)
        if self.optimize_warp_net:
            assert self.model.gs_img_model is not None

        self.optimize_inputs = self.optimize_expression or self.optimize_pose or self.optimize_ldmk or self.optimize_layer_offset or self.optimize_warp_net

        if kwargs['no_test_opt'] and self.optimize_inputs:
            print('Expresion and poses are optimized during training, but not during testing')
            self.optimize_inputs = False
            self.optimize_expression = False
            self.optimize_pose = False
            self.optimize_ldmk = False
            self.optimize_layer_offset = False
            self.use_per_frame_noise = False
            self.eval_dir = self.eval_dir + "_no_test_opt"

            if self.apply_extra_euc:
                self.eval_dir = self.eval_dir + "_ex_euc"
            if self.mlp_warp_smooth:
                self.eval_dir = self.eval_dir + "_mlp_warp_smooth"
            
        if self.smooth_test_opt_render:
            self.eval_dir = self.eval_dir + "_test_opt_smooth"

        if self.optimize_inputs:
            self.eval_dir = self.eval_dir + f"_{self.conf.get_int('test.opt_iter', 20)}"

        if self.model.gs_img_model is not None and not self.optimize_layer_offset:
            self.model.gs_img_model.apply_test_time_affine = self.conf.get('test.apply_test_time_affine', False)

        if self.simulate_fast_anchor:
            assert self.model.gs_img_model is not None
            self.model.gs_img_model.apply_test_time_affine = True


        self.plot_dataloader = torch.utils.data.DataLoader(self.plot_dataset,
                                                           batch_size=min(int(self.conf.get_int('train.max_points_training') /self.model.pc.points.shape[0]),self.conf.get_int('train.max_batch',default='10')),
                                                           shuffle=False,
                                                           collate_fn=self.plot_dataset.collate_fn
                                                           )
        self.optimize_tracking = False
        if self.optimize_inputs:
            self.input_params_subdir = "TestInputParameters"
            if self.simulate_fast_anchor:
                self.input_params_subdir += "NoMLP"
            test_input_params = []

            lr_input = self.conf.get('test.learning_rate_cam_test', None)
            if lr_input is None:
                lr_input = self.conf.get_float('train.learning_rate_cam')
            if self.optimize_expression:
                init_expression = self.plot_dataset.data["expressions"]

                self.expression = torch.nn.Embedding(len(self.plot_dataset), self.model.deformer_network.num_exp, _weight=init_expression, sparse=True).cuda()
                test_input_params.append({'name': 'test_exp', 'params': list(self.expression.parameters()), 'lr': lr_input})

            if self.optimize_pose:
                self.flame_pose = torch.nn.Embedding(len(self.plot_dataset), 15,
                                                     _weight=self.plot_dataset.data["flame_pose"],
                                                     sparse=True).cuda()
                self.camera_pose = torch.nn.Embedding(len(self.plot_dataset), 3,
                                                      _weight=torch.zeros_like(self.plot_dataset.data["world_mats"][:, :3, 3]),
                                                      sparse=True).cuda()
                test_input_params.append({'name': 'test_pose', 'params': list(self.flame_pose.parameters()) + list(self.camera_pose.parameters()), 'lr': lr_input})

            if self.optimize_ldmk:
                self.ldmks_opted = torch.nn.Embedding(len(self.plot_dataset), self.plot_dataset.data["body_ldmks"].shape[-1],
                                                     _weight=self.plot_dataset.data["body_ldmks"],
                                                     sparse=True).cuda()
                test_input_params.append({'name': 'test_ldmk', 'params': list(self.ldmks_opted.parameters()), 'lr': lr_input})

            if self.optimize_layer_offset:
                if self.simulate_fast_anchor:
                    self.layer_offset_opted = torch.nn.Embedding(len(self.plot_dataset), 8,
                                                        _weight=torch.tensor(torch.tensor([[1,0,0,0,1,0,0,0]]).repeat(self.plot_dataset.data["flame_pose"].shape[0], 1).float().cuda()),
                                                        sparse=True).cuda()
                else:
                    self.layer_offset_opted = torch.nn.Embedding(len(self.plot_dataset), 3, # last digit is rotation in radius
                                                        _weight=torch.zeros_like(self.plot_dataset.data["flame_pose"][:,:3]),
                                                        sparse=True).cuda()
                self.text_offset_params = [
                    {'name': 'test_offset', 'params': list(self.layer_offset_opted.parameters()), 'lr': self.conf.get_float('test.learning_rate_layer_offset')}]
                

            
            if self.use_per_frame_noise:
                noise_dim = self.conf.get_int('model.bg_layer_model.args.frame_noise_dim')
                self.per_frame_noises = torch.nn.Embedding(len(self.plot_dataset), noise_dim,
                                                        _weight=torch.zeros(len(self.plot_dataset), noise_dim),
                                                        sparse=True).cuda()
                
                
                self.use_per_frame_noise = self.conf.get_bool('model.bg_layer_model.args.use_per_frame_noise', False)
                test_input_params.append({'name': 'test_offset', 'params': list(self.layer_offset_opted.parameters()), 'lr': self.conf.get_float('test.learning_rate_layer_offset')})
                
                test_input_params.append({
                    'name': 'test_noise', 
                    'params': list(self.per_frame_noises.parameters()), 
                    'lr': self.conf.get_float('test.learning_rate_noise', self.conf.get_float('train.learning_rate_noise'))})
                
            if self.optimize_warp_net:
                self.warp_net_optimizer = torch.optim.Adam(
                    params=list(self.model.gs_img_model.deform_net.parameters()),
                    lr = self.conf.get_float('test.optimize_warp_net_lr', lr_input)
                )


            self.test_input_params = test_input_params
            self.lr_input = lr_input

            self.optimized_tracking_loaded = False

            
            try:
                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.input_params_subdir, str(kwargs['checkpoint']) + ".pth"))
                if self.optimize_expression:
                    self.expression.load_state_dict(data["expression_state_dict"])
                if self.optimize_pose:
                    self.flame_pose.load_state_dict(data["flame_pose_state_dict"])
                    self.camera_pose.load_state_dict(data["camera_pose_state_dict"])
                if self.optimize_ldmk:
                    self.ldmks_opted.load_state_dict(data["ldmks_state_dict"])
                if self.optimize_layer_offset:
                    self.layer_offset_opted.load_state_dict(data["layer_offset_state_dict"])
                print('Using pre-tracked test expressions')
                self.optimized_tracking_loaded = True
            except Exception as e:
                print(e)
            self.optimize_tracking = True
            from model.loss import Loss
            self.loss = Loss(**self.conf.get('test.loss', {}))
            print('Optimizing test expressions')

        self.img_res = self.plot_dataset.img_res


    def save_test_tracking(self, epoch):

        if not os.path.exists(os.path.join(self.checkpoints_path, self.input_params_subdir)):
            os.mkdir(os.path.join(self.checkpoints_path, self.input_params_subdir))
        if self.optimize_inputs:
            dict_to_save = {}
            dict_to_save["epoch"] = epoch
            if self.optimize_expression:
                dict_to_save["expression_state_dict"] = self.expression.state_dict()
            if self.optimize_pose:
                dict_to_save["flame_pose_state_dict"] = self.flame_pose.state_dict()
                dict_to_save["camera_pose_state_dict"] = self.camera_pose.state_dict()
            if self.optimize_ldmk:
                dict_to_save["ldmks_state_dict"] = self.ldmks_opted.state_dict()
            if self.optimize_layer_offset:
                dict_to_save["layer_offset_state_dict"] = self.layer_offset_opted.state_dict()
            if self.use_per_frame_noise:
                dict_to_save["frame_noise_state_dict"] = self.per_frame_noises.state_dict()
            torch.save(dict_to_save, os.path.join(self.checkpoints_path, self.input_params_subdir, str(epoch) + ".pth"))
            torch.save(dict_to_save, os.path.join(self.checkpoints_path, self.input_params_subdir, "latest.pth"))

    def run(self):
        self.model.eval()
        self.model.training = False

        if self.simulate_fast_anchor:
            distill_inputs = self.train_dataset.get_canonical_frame_inputs()
            for k, v in distill_inputs.items():
                try:
                    distill_inputs[k] = v.cuda()
                except:
                    distill_inputs[k] = v
            self.model(distill_inputs, distill_anchor_model=True)


            ids = torch.randperm(len(self.train_dataset))[:100]

            for i in ids:
                _, filter_inputs, _ = self.train_dataset.collate_fn([self.train_dataset[i]])
                for k, v in filter_inputs.items():
                    try:
                        filter_inputs[k] = v.cuda()
                    except:
                        filter_inputs[k] = v

                self.model(filter_inputs, ransac_filter_anchor=True)




        if self.optimize_tracking:
            print("Optimizing tracking, this is a slow process which is only used for calculating metrics. \n"
                  "for qualitative animation, set optimize_expression and optimize_camera to False in the conf file.")
            all_losses = []

            eval_all = True
            is_first_batch = True
            imgs = []


            if self.smooth_test_opt_render:
                assert self.optimized_tracking_loaded
                # import pdb; pdb.set_trace()

                with torch.no_grad():
                    weights = self.layer_offset_opted.weight.detach().cpu().numpy()
                    smooth_weights = savgol_filter(weights, 61, 2, axis=0)
                    self.layer_offset_opted.weight.data = torch.from_numpy(smooth_weights).type_as(self.layer_offset_opted.weight.data)


            with torch.no_grad():
                if self.mlp_warp_smooth:
                    mlp_delta_uvs = []
                    eval_iterator = iter(self.plot_dataloader)

                    for img_index in tqdm(range(len(self.plot_dataloader)), desc='Smoothing'):
                        indices, model_input, ground_truth = next(eval_iterator)
                        for k, v in model_input.items():
                            try:
                                model_input[k] = v.cuda()
                            except:
                                model_input[k] = v
                        if self.optimize_expression:
                            model_input['expression'] = self.expression(model_input["idx"]).squeeze(1)
                        if self.optimize_pose:
                            model_input['flame_pose'] = self.flame_pose(model_input["idx"]).squeeze(1)
                            model_input['cam_pose_offset'] = self.camera_pose(model_input["idx"]).squeeze(1)
                        if self.optimize_ldmk:
                            model_input['body_ldmk'] = self.ldmks_opted(model_input["idx"]).squeeze(1)
                        if self.optimize_layer_offset:
                            model_input['layer_offset'] = self.layer_offset_opted(model_input["idx"]).squeeze(1)
                        if self.use_per_frame_noise:
                            model_input['frame_noise'] = self.per_frame_noises(model_input["idx"]).squeeze(1)
                        

                        model_outputs = self.model(model_input, clip_texture=self.conf.get('train.train_texture_only_clip', False))

                        if "mlp_warp_delta_uv" in model_outputs:
                            mlp_delta_uvs.append(model_outputs["mlp_warp_delta_uv"].detach().cpu())

                    mlp_delta_uvs = torch.concat(mlp_delta_uvs, axis=0).cpu().numpy()
                    delta_uv_smooth = savgol_filter(mlp_delta_uvs, 41, 2,axis=0)
                    smoothed_mlp_delta_uvs = torch.from_numpy(delta_uv_smooth)



            for data_index, (indices, model_input, ground_truth) in tqdm(enumerate(self.plot_dataloader), desc="Optimizing test pose"):
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

                model_input['ori_expression'] = model_input['expression'].detach().clone()
                model_input['ori_flame_pose'] = model_input['flame_pose'].detach().clone()
                model_input['ori_cam_pose'] = model_input['cam_pose'].detach().clone()
                if 'body_ldmk' in model_input:
                    model_input['ori_body_ldmk'] = model_input['body_ldmk'].detach().clone()
                
                # reset optimizer for every frame
                # TODO: might be better to just use RMSProp etc
                self.optimizer_cam = torch.optim.SparseAdam(self.test_input_params,
                                                            self.lr_input)
                
                if self.optimize_layer_offset:
                    self.optimizer_text_offset = torch.optim.SparseAdam(self.text_offset_params, self.lr_input)

                if self.optimize_warp_net:
                    # need to re-load the ckpt for every frame
                    saved_model_state = torch.load(self.model_ckpt_path)
                    self.model.load_state_dict(saved_model_state["model_state_dict"]) #, strict=False)
                    self.model.gs_img_model.deform_net.train()


                losses = []
                if not self.optimized_tracking_loaded:
                    for i in range(self.conf.get_int('test.opt_iter', 20)):
                        if self.optimize_expression:
                            model_input['expression'] = self.expression(model_input["idx"]).squeeze(1)
                        if self.optimize_pose:
                            model_input['flame_pose'] = self.flame_pose(model_input["idx"]).squeeze(1)
                            # model_input['cam_pose'] = torch.cat([R, self.camera_pose(model_input["idx"]).squeeze(1).unsqueeze(-1)], -1)
                            model_input['cam_pose_offset'] = self.camera_pose(model_input["idx"]).squeeze(1)
                        if self.optimize_ldmk:
                            model_input['body_ldmk'] = self.ldmks_opted(model_input["idx"]).squeeze(1)
                        if self.optimize_layer_offset:
                            model_input['layer_offset'] = self.layer_offset_opted(model_input["idx"]).squeeze(1)
                        if self.use_per_frame_noise:
                            model_input['frame_noise'] = self.per_frame_noises(model_input["idx"]).squeeze(1)

                        model_outputs = self.model(model_input, clip_texture=self.conf.get('train.train_texture_only_clip', False))
                        loss_output = self.loss(model_outputs, ground_truth)
                        loss = loss_output['loss']
                        self.optimizer_cam.zero_grad()
                        if self.optimize_layer_offset:
                            self.optimizer_text_offset.zero_grad()
                        loss.backward(retain_graph=self.optimize_warp_net)
                        self.optimizer_cam.step()
                        if self.optimize_layer_offset:
                            self.optimizer_text_offset.step()


                        if self.optimize_warp_net:
                            # optimize deform net individually with deform loss only
                            self.warp_net_optimizer.zero_grad()
                            deform_loss = model_outputs['anchor_deform_loss']
                            deform_loss.backward()
                            self.warp_net_optimizer.step()

                        losses.append(loss.item())

                    all_losses.append(losses)

                # render again to save img
                with torch.no_grad():
                    self.model.eval()
                    if self.optimize_expression:
                        model_input['expression'] = self.expression(model_input["idx"]).squeeze(1)
                    if self.optimize_pose:
                        model_input['flame_pose'] = self.flame_pose(model_input["idx"]).squeeze(1)
                        model_input['cam_pose_offset'] = self.camera_pose(model_input["idx"]).squeeze(1)
                    if self.optimize_ldmk:
                        model_input['body_ldmk'] = self.ldmks_opted(model_input["idx"]).squeeze(1)
                    if self.optimize_layer_offset:
                        model_input['layer_offset'] = self.layer_offset_opted(model_input["idx"]).squeeze(1)
                    if self.use_per_frame_noise:
                        model_input['frame_noise'] = self.per_frame_noises(model_input["idx"]).squeeze(1)

                    if self.mlp_warp_smooth:
                        model_input["smoothed_mlp_delta_uvs"] = smoothed_mlp_delta_uvs[indices].cuda()

                    
                    model_outputs = self.model(model_input, clip_texture=self.conf.get('train.train_texture_only_clip', False))
                    for k, v in model_outputs.items():
                        try:
                            model_outputs[k] = v.detach()
                        except:
                            model_outputs[k] = v

                    # import pdb; pdb.set_trace()
                    plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], f'epoch_{str(self.start_epoch)}_test') for i in range(len(model_input['sub_dir']))]

                    img_names = model_input['img_name'][:,0].cpu().numpy()
                    # print("Plotting images: {}".format(img_names))
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
                            save_compare=self.quick_eval,
                            )
                    imgs.append(np.array(img))
                    is_first_batch = False
                    del model_outputs, ground_truth

            if self.log_wandb and self.wandb_debug_log and not self.optimized_tracking_loaded:
                # log test opt losses
                all_losses = torch.tensor(all_losses)
                all_losses = all_losses.mean(axis=0)

                for l in all_losses:
                    wandb.log({
                        'test_opt/loss': l
                    })
            if not self.quick_eval:
                self.save_test_tracking(epoch=self.start_epoch)
            
            if self.quick_eval:
                # compress to video
                imgs = np.stack(imgs)
                torchvision.io.write_video(os.path.join(self.eval_dir, 'test.mp4'), imgs, fps=10, video_codec='h264')

            if not self.plot_dataset.only_json and not self.no_metrics:
                from utils.metrics import run as cal_metrics
                # cal_metrics(output_dir=plot_dir[0], gt_dir=self.plot_dataset.gt_dir, pred_file_name='rgb_erode_dilate')
                # cal_metrics(output_dir=plot_dir[0], gt_dir=self.plot_dataset.gt_dir, pred_file_name='rgb_erode_dilate', no_cloth=True)
                metrics = cal_metrics(output_dir=plot_dir[0], gt_dir=self.plot_dataset.gt_dir, pred_file_name='rgb')
                # metrics_no_cloth = cal_metrics(output_dir=plot_dir[0], gt_dir=self.plot_dataset.gt_dir, pred_file_name='rgb', no_cloth=True)
                metrics_no_head = cal_metrics(output_dir=plot_dir[0], gt_dir=self.plot_dataset.gt_dir, pred_file_name='rgb', no_head=True)

                if self.log_wandb:
                    wandb.log({
                        "test/mse": metrics["mse"],
                        "test/psnr": metrics["psnr"],
                        "test/ssim": metrics["ssim"],
                        "test/lpips": metrics["lpips"],
                        # "test/mse_no_cloth": metrics_no_cloth["mse"],
                        # "test/psnr_no_cloth": metrics_no_cloth["psnr"],
                        # "test/ssim_no_cloth": metrics_no_cloth["ssim"],
                        # "test/lpips_no_cloth": metrics_no_cloth["lpips"],
                        "test/mse_no_head": metrics_no_head["mse"],
                        "test/psnr_no_head": metrics_no_head["psnr"],
                        "test/ssim_no_head": metrics_no_head["ssim"],
                        "test/lpips_no_head": metrics_no_head["lpips"],
                    })

            

        else:
            eval_all = True
            
            is_first_batch = True
            imgs = []

            if self.apply_extra_euc:
                assert not self.simulate_fast_anchor

                self.model.gs_img_model.apply_test_time_affine = True
                self.model.gs_img_model.affine_type = 'euclidean'

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
                    delta_uv_smooth = savgol_filter(mlp_delta_uvs, 41, 2,axis=0)
                    smoothed_mlp_delta_uvs = torch.from_numpy(delta_uv_smooth)


            eval_iterator = iter(self.plot_dataloader)

            for img_index in tqdm(range(len(self.plot_dataloader)), desc='Rendering'):
                indices, model_input, ground_truth = next(eval_iterator)
                batch_size = model_input['expression'].shape[0]
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
                
                if self.mlp_warp_smooth:
                    model_input["smoothed_mlp_delta_uvs"] = smoothed_mlp_delta_uvs[indices].cuda()



                model_outputs = self.model(model_input, clip_texture=self.conf.get('train.train_texture_only_clip', False))
                for k, v in model_outputs.items():
                    try:
                        model_outputs[k] = v.detach()
                    except:
                        model_outputs[k] = v


                plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], f'epoch_{str(self.start_epoch)}_test') for i in range(len(model_input['sub_dir']))]

                img_names = model_input['img_name'][:,0].cpu().numpy()
                # print("Plotting images: {}".format(img_names))
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
                        save_compare=self.quick_eval,
                        )
                imgs.append(np.array(img))
                is_first_batch = False
                del model_outputs, ground_truth

            if self.quick_eval:
                # compress to video
                imgs = np.stack(imgs)
                torchvision.io.write_video(os.path.join(self.eval_dir, 'test.mp4'), imgs, fps=10, video_codec='h264')

            if not self.plot_dataset.only_json and not self.no_metrics:
                from utils.metrics import run as cal_metrics
                metrics = cal_metrics(output_dir=plot_dir[0], gt_dir=self.plot_dataset.gt_dir, pred_file_name='rgb')
                metrics_no_head = cal_metrics(output_dir=plot_dir[0], gt_dir=self.plot_dataset.gt_dir, pred_file_name='rgb', no_head=True)

                if self.log_wandb:
                    wandb.log({
                        "test/mse": metrics["mse"],
                        "test/psnr": metrics["psnr"],
                        "test/ssim": metrics["ssim"],
                        "test/lpips": metrics["lpips"],
                        # "test/mse_no_cloth": metrics_no_cloth["mse"],
                        # "test/psnr_no_cloth": metrics_no_cloth["psnr"],
                        # "test/ssim_no_cloth": metrics_no_cloth["ssim"],
                        # "test/lpips_no_cloth": metrics_no_cloth["lpips"],
                        "test/mse_no_head": metrics_no_head["mse"],
                        "test/psnr_no_head": metrics_no_head["psnr"],
                        "test/ssim_no_head": metrics_no_head["ssim"],
                        "test/lpips_no_head": metrics_no_head["lpips"],
                    })



