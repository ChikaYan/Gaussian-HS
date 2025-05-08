import sys
sys.path.append('../code')
sys.path.append('./')
import argparse
from pyhocon import ConfigFactory, ConfigTree
from copy import deepcopy


from scripts.train import TrainRunner
from scripts.test import TestRunner
from scripts.reenact import ReenactRunner

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str)
    parser.add_argument('--conf_reenact', type=str, default=None)
    parser.add_argument('--is_eval', default=False, action="store_true", help='If set, only render images')
    parser.add_argument('--is_reenact', default=False, action="store_true", help='If set, only render reenact images')
    # Training flags
    parser.add_argument('--nepoch', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('--wandb_workspace', type=str, default='pointavatar_gs-code_scripts')
    parser.add_argument('--wandb_tags', type=str, nargs="+", default=[])
    parser.add_argument('--wandb_mode', type=str, default='online')
    parser.add_argument('--is_continue', default=False, action="store_true", help='If set, indicates continuing from a previous run.')
    # Testing flags
    parser.add_argument('--only_json', default=False, action="store_true", help='If set, do not load images during testing. ')
    # Checkpoints
    parser.add_argument('--checkpoint', default='latest', type=str, help='The checkpoint epoch number in case of continuing from a previous run.')
    parser.add_argument('--load_path', type=str, default='', help='If set explicitly, then load from this path, instead of the continue-scripts path')

    parser.add_argument('--quick_eval', default=False, action="store_true", help='Eval with a subset of test samples for quick results')
    parser.add_argument('--subject', type=str, default=None, help='If set explicitly, overwrite the subject')
    parser.add_argument('--no_test_opt', default=False, action="store_true", help='Disable test time optimization')
    parser.add_argument('--run_fast_test', default=False, action="store_true", help='')
    parser.add_argument('--no_metrics', default=False, action="store_true", help='')
    parser.add_argument('--apply_extra_euc', default=False, action="store_true", help='for test no time optimization only')
    parser.add_argument('--mlp_warp_smooth', default=False, action="store_true", help='')
    parser.add_argument('--smooth_test_opt_render', default=False, action="store_true", help='')
    opt = parser.parse_args()

    conf = ConfigFactory.parse_file(opt.conf)

    if opt.conf_reenact is not None:
        conf_reenact = ConfigFactory.parse_file(opt.conf_reenact)
        conf = ConfigTree.merge_configs(conf, conf_reenact)

    if opt.run_fast_test:
        conf.put('model.gs_img_model.args.affine_type','projective')
        conf.put('test.opt_iter', 50)

    if opt.subject == '001':
        # 001 dataset uses a separate folder for test images
        conf['dataset']['subject_name'] = '001'
        conf['dataset']['train']['sub_dir'] = ['train']
        conf['dataset']['train']['frame_interval'] = [0, 99999999]


        
    if opt.subject == '003':
        # required due to failure in mask in 003
        conf.put('model.gs_img_model.args.distill_texture_bbox', [-30, 320, 520, 520])


    if opt.is_reenact:
        runner = ReenactRunner(conf=deepcopy(conf),
                            conf_path=opt.conf,
                            checkpoint='latest',
                            load_path=opt.load_path,
                            only_json=opt.only_json,
                            wandb_workspace=opt.wandb_workspace,
                            quick_eval=opt.quick_eval,
                            subject=opt.subject,
                            run_fast_test=opt.run_fast_test,
                            )

        runner.run()
        exit(0)

    # 001 dataset uses a separate folder for test images
    if opt.subject == '001':
        conf['dataset']['subject_name'] = '001'
        conf['dataset']['json_name'] = 'flame_params.json'
        conf['dataset']['train']['sub_dir'] = ['train']
        conf['dataset']['train']['frame_interval'] = [0, 99999999]
        conf['dataset']['test']['sub_dir'] = ['test']
        conf['dataset']['test']['frame_interval'] = [0, 99999999]


    if not opt.is_eval:
        runner = TrainRunner(conf=deepcopy(conf),
                             conf_path=opt.conf,
                             nepochs=opt.nepoch,
                             checkpoint=opt.checkpoint,
                             is_continue=opt.is_continue,
                             load_path=opt.load_path,
                             wandb_workspace=opt.wandb_workspace,
                             wandb_tags=opt.wandb_tags,
                             subject=opt.subject,
                             wandb_mode=opt.wandb_mode,
                             )
        runner.run()



    runner = TestRunner(conf=deepcopy(conf),
                        conf_path=opt.conf,
                        checkpoint='latest',
                        load_path=opt.load_path,
                        only_json=opt.only_json,
                        wandb_workspace=opt.wandb_workspace,
                        quick_eval=opt.quick_eval,
                        subject=opt.subject,
                        wandb_mode=opt.wandb_mode,
                        no_test_opt=opt.no_test_opt,
                        run_fast_test=opt.run_fast_test,
                        wandb_tags=opt.wandb_tags,
                        no_metrics=opt.no_metrics,
                        apply_extra_euc=opt.apply_extra_euc,
                        mlp_warp_smooth=opt.mlp_warp_smooth,
                        smooth_test_opt_render=opt.smooth_test_opt_render,
                        )
    runner.run()
