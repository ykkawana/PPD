import os
import hashlib
import pickle
import sys
import torch
from torch import optim
import argparse
from torch.utils import data
from data import dataset
from model import movenet
from trainer import imex_trainer
import dotenv
import wandb
from collections import defaultdict
import numpy as np
from utils import config_util
from utils import checkpoint_util
from datetime import datetime
from trains import option_encoder
import random
import math
import yaml
import importlib
from tqdm import tqdm
import time
import retrying
dotenv.load_dotenv()


def get_config(cfg):
    autoencoder_class_path = cfg['model'].get(
        'class', 'model.movenet.movenet.MoveNetAutoEncoder')
    tmp = autoencoder_class_path.split('.')
    module_path = tmp[:-2]
    module_path.append('config')
    module_path = '.'.join(module_path)
    config_ob = getattr(importlib.import_module(module_path), 'Config')(cfg)
    return config_ob


@retrying.retry(stop_max_attempt_number=5, wait_fixed=5000)
def get_artifacts(cfg, iters, epoch, metric_val_best, outdir, run, config_ob):
    artifacts = config_ob.get_artifact(iters, epoch, metric_val_best)
    if 'model' in cfg:
        common_kwargs = dict(only_include_parameters=cfg['model'].get(
            'pretrained', {
                'only_include_parameters': dict()
            }).get('only_include_parameters', []),
                             ignore_artifact_keys=cfg['model'].get(
                                 'pretrained', {
                                     'ignore_artifact_keys': dict()
                                 }).get('ignore_artifact_keys', []))
    else:
        common_kwargs = {}
    if cfg['training']['resume']['type'] == 'local':
        """
        if int(cfg['version'][1:]) <= 2:
            assert os.path.exists(os.path.join(outdir, 'model_latest.pth'))
            assert cfg['training']['resume']['from'] is None
            checkpoint_util.load(artifacts,
                                 filepath=os.path.join(outdir,
                                                       'model_latest.pth'))
        """
        assert os.path.exists(cfg['training']['resume']['path'])

        checkpoint_util.load(artifacts,
                             filepath=cfg['training']['resume']['path'],
                             **common_kwargs)
    elif cfg['training']['resume']['type'] == 'wandb':
        assert os.getenv('WANDB_MODE', '') != 'dryrun'
        wandb_artifact_name = cfg['training']['resume']['from']
        assert len(wandb_artifact_name.split(
            ':')) == 2, 'You need version (e.g. name:version)'
        checkpoint_util.load(artifacts,
                             wandb_artifact=True,
                             wandb_run=run,
                             wandb_artifact_name=wandb_artifact_name,
                             **common_kwargs)
    else:
        raise NotImplementedError

    return artifacts


def eval_step(epoch, iters, metric_val_best, model_selection_metric,
              model_selection_sign, config_ob, outdir, run):
    loss_dict = defaultdict(lambda: [])
    post_loss_dict = defaultdict(lambda: [])
    for step, batch in tqdm(enumerate(config_ob.val_dataloader)):
        losses = config_ob.trainer_ob.eval_step(batch, iters)
        for key, value in losses.items():
            if key.startswith('post_'):
                post_loss_dict[key].append(
                    {k: v.detach().cpu().numpy()
                     for k, v in value.items()})
            else:
                loss_dict[key].append(value.detach().cpu().numpy())
    losses_cpu_eval = {
        'val/' + key: np.nanmean(np.array(value))
        for key, value in loss_dict.items()
    }
    post_losses = config_ob.trainer_ob.post_loss_eval(post_loss_dict)
    post_losses_cpu_eval = {
        'val/' + key: value
        for key, value in post_losses.items()
    }
    losses_cpu_eval.update(post_losses_cpu_eval)
    losses_cpu_eval['epoch'] = epoch
    wandb.log(losses_cpu_eval, step=iters)

    metric_val_dict = {}
    metric_val_dict.update(loss_dict)
    metric_val_dict.update(post_losses)
    metric_val = np.nanmean(np.array(metric_val_dict.get(model_selection_metric, 0)))
    if -model_selection_sign * (metric_val - metric_val_best) > 0:
        metric_val_best = metric_val
        artifacts = config_ob.get_artifact(iters, epoch, metric_val_best)
        checkpoint_util.save(os.path.join(outdir, 'model_best.pth'),
                             artifacts,
                             wandb_artifact=True,
                             wandb_run=run,
                             wandb_artifact_suffix='_best')
    return metric_val_best


def save_checkpoint(epoch, iters, metric_val_best, config_ob, outdir, run):
    artifacts = config_ob.get_artifact(iters, epoch, metric_val_best)
    checkpoint_util.save(os.path.join(outdir, 'model_{}.pth'.format(iters)),
                         artifacts,
                         wandb_artifact=True,
                         wandb_run=run)
    checkpoint_util.save(os.path.join(outdir, 'model_latest.pth'), artifacts)


def visualize_step(config_ob, batch_vis, iters):
    if config_ob.cfg['training']['visualize'].get('use_eval_mode', False):
        config_ob.model.eval()
    ret = config_ob.trainer_ob.visualize(batch_vis)
    if config_ob.cfg['training']['visualize'].get('use_eval_mode', False):
        config_ob.model.train()
    vis = {}
    for r in ret:
        typ = r['type']
        if typ == 'image':
            vis[r['desc']] = [wandb.Image(img) for img in r['data']]
        elif typ == 'array':
            vis['pred_slide'] = r['data']
            print(r['data'])
    wandb.log(vis, step=iters)


@retrying.retry(stop_max_attempt_number=5, wait_fixed=5000)
def init_wandb(wandb_kwargs):
    run = wandb.init(**wandb_kwargs)
    return run


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def init_run(cfg, parse_args, vis_batch_idx=0):
    assert 'version' in cfg and cfg['version'] in ['v2', 'v3']
    if isinstance(cfg.get('seed', None), int):
        print('Overwrite random seeds with {}'.format(cfg['seed']))
        set_random_seed(cfg['seed'])
    non_null_wandb = {}
    for key, item in cfg['wandb'].items():
        if item is not None:
            non_null_wandb[key] = item
    if not parse_args.allow_resume_from_same_run_id:
        if 'id' in non_null_wandb:
            del non_null_wandb['id']
        if 'resume' in non_null_wandb:
            del non_null_wandb['resume']

    #run = wandb.init(**non_null_wandb)
    run = init_wandb(non_null_wandb)
    wandb.config.update(
        cfg, allow_val_change=parse_args.allow_resume_from_same_run_id)

    date_str = datetime.now().strftime(('%Y%m%d_%H%M%S'))
    if cfg['training'].get('resume') == 'local':
        assert cfg['training']['outdir'] == cfg['training']['outdir'].format(
            'aaa')
        outdir = cfg['training']['outdir'].format(date_str)
    else:
        outdir = cfg['training']['outdir'].format(date_str)
    config_ob = get_config(cfg)

    pretrained_weight_path = cfg['model'].get('pretrained_weight', None)
    if pretrained_weight_path is not None:
        checkpoint_util.load_weight(config_ob.model,
                                    torch.load(pretrained_weight_path))
    config_ob.init_dataset_loader(parse_args)
    batch_vis = next(iter(config_ob.vis_dataloader))
    for idx in range(vis_batch_idx):
        print(idx)
        batch_vis = next(iter(config_ob.vis_dataloader))

    wandb.watch(config_ob.model)
    optimizers = {}
    for group, parameters in config_ob.model.parameter_groups.items():
        params = cfg['training']['optimizer_groups'][group]
        optimizers[group] = optim.Adam(parameters,
                                       lr=params['learning_rate'],
                                       **params.get('kwargs', {}))

    epoch = 0
    iters = 1 if parse_args.skip_first_eval_and_vis else 0

    model_selection_mode = cfg['training']['eval'].get('model_selection_mode')
    if not model_selection_mode is None:
        assert model_selection_mode in ['min', 'max']
        model_selection_sign = 1 if model_selection_mode == 'min' else -1
        metric_val_best = model_selection_sign * np.inf
    else:
        model_selection_sign = 1
        metric_val_best = 0

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    model_selection_metric = cfg['training']['eval'].get('model_selection_metric', '')
    if cfg['training'].get('resume') and cfg['training'].get('resume').get(
            'type'):
        artifacts = get_artifacts(cfg, iters, epoch, metric_val_best, outdir,
                                  run, config_ob)
        iters = artifacts['scalar']['iters']
        epoch = artifacts['scalar']['epochs']
        metric_val_best = artifacts['scalar'].get('metric_val_best', 0)

        print('Resume from {} iters, {} epochs. Best {} so far: {}.'.format(
            iters, epoch, model_selection_metric, metric_val_best))

    # Send to GPU
    config_ob.model.to(config_ob.device)
    config_ob.init_trainer()

    if cfg.get('pretrained_model'):
        for model_name, pretrained_cfg in cfg['pretrained_model'][
                'model'].items():
            loaded_pretrained_cfg = get_config_from_run_id(
                pretrained_cfg['training']['resume']['from'])
            pretrained_config_ob = get_config(loaded_pretrained_cfg)
            pretrained_model = get_artifacts(
                pretrained_cfg, -1, -1, -1, None, run,
                pretrained_config_ob)['model']['movenet']
            config_ob.trainer_ob.pretrained_models[
                model_name] = pretrained_model
            print(model_name, 'loaded')

    ret = {}
    ret['iters'] = iters
    ret['epoch'] = epoch
    ret['config_ob'] = config_ob
    ret['outdir'] = outdir
    ret['run'] = run
    ret['batch_vis'] = batch_vis
    ret['metric_val_best'] = metric_val_best
    ret['model_selection_metric'] = model_selection_metric
    ret['model_selection_sign'] = model_selection_sign
    return ret


def gen_mesh(cfg, parse_args):
    mesh_cfg = cfg.get('mesh_generation', {})
    if 'kwargs' in mesh_cfg:
        mesh_cfg['kwargs']['threshold'] = parse_args.mesh_gen_threshold
    else:
        mesh_cfg['kwargs'] = {
            'threshold': parse_args.mesh_gen_threshold
        }
    cfg['mesh_generation'] = mesh_cfg
    ret = init_run(cfg, parse_args)
    config_ob = ret['config_ob']
    config_ob.init_mesh_generator()
    mesh_generator_ob = config_ob.mesh_generator_ob

    method = 'pmd'
    classes = '_'.join(cfg['data']['common']['kwargs']['classes'])
    run_id = parse_args.run_id
    date_str = datetime.now().strftime(('%Y%m%d_%H%M%S'))

    if parse_args.mesh_dirpath is None:
        dirpath = os.path.join(parse_args.mesh_out_dir,
                               '_'.join([method, classes, run_id, date_str]))
    else:
        dirpath = parse_args.mesh_dirpath

    mesh_cfg = {
        'mesh_generation': {
            'base_outdir': parse_args.mesh_out_dir,
            'kwargs': {
                'resolution': 64,
                'threshold': parse_args.mesh_gen_threshold
            },
            'outdir': dirpath
        }
    }
    cfg.update(mesh_cfg)
    test_cfg = {
        'test': {
            'list_path':
            'artifacts/dataset/shape2motion_v3/list/test.lst',
            'data_dir': 'artifacts/dataset/shape2motion_v3/data/sample',
            'filenames': {
                'surface_points_filename':
                'surface_points_whole_shape_sample_wise_normalize.npz',
                'points_values_filename':
                'points_values_whole_shape_sample_wise_normalize_32.npz',
                'source_whole_mesh_filename':
                'whole_mesh.obj',
                'source_part_mesh_filename_template':
                'part_mesh_{}.obj',
                'part_mesh_filename_template':
                'surface_mesh_sample_wise_normalize_{}.obj'
            }
        }
    }
    cfg.update(test_cfg)

    mesh_path = os.path.join(dirpath, 'mesh')
    if not os.path.exists(mesh_path):
        os.makedirs(mesh_path)
    if parse_args.mesh_dirpath is None:
        yaml.dump(cfg, open(os.path.join(mesh_path, 'config.yaml'), 'w'))
    cnt = 0
    for step, batch in tqdm(enumerate(config_ob.test_dataloader)):
        if parse_args.mode == 'gen_mesh':
            rets = mesh_generator_ob.generate_mesh(batch)
        else:
            raise ValueError
        for idx, (ret, class_id, modelname) in enumerate(
                zip(rets, batch['class_id'], batch['modelname'])):
            for key, value in ret.items():
                out_path = os.path.join(mesh_path, class_id,
                                        modelname.replace('/', '_'), key)
                if not os.path.exists(os.path.dirname(out_path)):
                    os.makedirs(os.path.dirname(out_path))

                if out_path.endswith('.obj'):
                    value.export(out_path)
                elif out_path.endswith('.npz'):
                    np.savez(out_path, **value)
                elif out_path.endswith('.pkl'):
                    pickle.dump(value, open(out_path, 'wb'))


def train(cfg, parse_args):
    ret = init_run(cfg, parse_args)
    iters = ret['iters']
    epoch = ret['epoch']
    config_ob = ret['config_ob']
    outdir = ret['outdir']
    run = ret['run']
    batch_vis = ret['batch_vis']
    metric_val_best = ret['metric_val_best']
    model_selection_metric = ret['model_selection_metric']
    model_selection_sign = ret['model_selection_sign']

    terminate_flag = False

    torch.autograd.set_detect_anomaly(True)
    while True:
        data_load_s = time.time()
        for step, batch in enumerate(config_ob.train_dataloader):
            print('data load:', time.time() - data_load_s)
            is_evaled = False
            is_vised = False
            train_step_s = time.time()
            losses = config_ob.trainer_ob.train_step(batch, iters)
            print('train step time:', time.time() - train_step_s)
            losses_cpu = {
                'train/' + key: value.mean().detach().cpu().numpy()
                for key, value in losses.items()
            }

            if iters % cfg['training']['every'] == 0:
                print('Training at {} th epoch, {} th iter'.format(
                    epoch, iters))
                losses_cpu['epoch'] = epoch
                wandb.log(losses_cpu, step=iters)

            if iters % cfg['training']['eval']['every'] == 0:
                print('Evaluate at {} th epoch, {} th iter'.format(
                    epoch, iters))
                is_evaled = True
                metric_val_best = eval_step(epoch, iters, metric_val_best,
                                            model_selection_metric,
                                            model_selection_sign, config_ob,
                                            outdir, run)

            if iters % cfg['training']['checkpoint']['every'] == 0:
                save_checkpoint(epoch, iters, metric_val_best, config_ob,
                                outdir, run)

            if iters % cfg['training']['visualize']['every'] == 0:
                is_vised = True
                print('Visualize at {} th epoch, {} th iter'.format(
                    epoch, iters))
                visualize_step(config_ob, batch_vis, iters)

            iters += 1
            if cfg['training']['terminate_iters'] > 0 and (
                    iters + 1) > cfg['training']['terminate_iters']:
                terminate_flag = True

            if terminate_flag:
                break
            data_load_s = time.time()

        epoch += 1
        if cfg['training']['terminate_epoch'] > 0 and (
                epoch) > cfg['training']['terminate_epoch']:
            terminate_flag = True

        if terminate_flag:
            epoch -= 1
            if not is_evaled:
                metric_val_best = eval_step(epoch, iters, metric_val_best,
                                            model_selection_metric,
                                            model_selection_sign, config_ob,
                                            outdir, run)
            if not is_vised:
                visualize_step(config_ob, batch_vis, iters)
            save_checkpoint(epoch, iters, metric_val_best, config_ob, outdir,
                            run)
            break


def parse(sysargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_seed', type=int, default=0)
    parser.add_argument('-id', '--run_id', type=str, default=None)
    parser.add_argument('-d', '--dryrun', action='store_true')
    parser.add_argument('-nd', '--no_dryrun', action='store_true')
    parser.add_argument('-rf', '--resume_from_run_id', action='store_true')
    parser.add_argument('--from_sweep_config', default=None, type=str)
    parser.add_argument('-cm',
                        '--check_memory',
                        default=None,
                        action='store_true')
    parser.add_argument('--option_combo_config', default=None, type=str)
    parser.add_argument('--skip_first_eval_and_vis', action='store_true')
    parser.add_argument('-cg', '--check_gradient', action='store_true')
    parser.add_argument('-ar',
                        '--allow_resume_from_same_run_id',
                        action='store_true')
    parser.add_argument('-rs',
                        '--resume_from_same_run_id',
                        action='store_true')
    parser.add_argument('--mode',
                        default='gen_mesh',
                        choices=['gen_mesh'])
    parser.add_argument('--advanced_reproducibility',
                        action='store_true')
    parser.add_argument('--checkpoint',
                        type=str,
                        default=None)
    parser.add_argument('--mesh_out_dir', default=None, type=str)
    parser.add_argument('--mesh_dirpath', default=None, type=str)
    parser.add_argument('--mesh_gen_threshold', default=0.5, type=float)
    return parser.parse_known_args(sysargs)


def get_config_from_run_id(run_id):
    assert 'WANDB_USERNAME' in os.environ
    assert 'WANDB_PROJECT' in os.environ
    run_id = run_id.split(':')[0]
    run_id = os.path.join(os.getenv('WANDB_USERNAME'),
                          os.getenv('WANDB_PROJECT'), run_id)
    api = wandb.Api()
    target_run = api.run(run_id.replace('_best', ''))
    cfg = target_run.config
    return cfg

def get_config_from_checkpoint(checkpoint_path):
    cfg = torch.load(checkpoint_path)['config']
    return cfg


def decode_args(parse_args, unknown_args):
    unknown_args_wo_eq = []

    if parse_args.from_sweep_config is not None:
        sweep_cfg = yaml.safe_load(open(parse_args.from_sweep_config))
        parse_args.config = sweep_cfg['base_config']
        for param_name, param in sweep_cfg['parameters'].items():
            unknown_args_wo_eq.append('--' + param_name)
            if 'value' in param:
                unknown_args_wo_eq.append(str(param['value']))
            if 'min' in param:
                value = param['min']
                if 'distribution' in param and 'log_uniform' == param[
                        'distribution']:
                    value = math.exp(value)
                unknown_args_wo_eq.append(str(value))
            if 'values' in param:
                unknown_args_wo_eq.append(str(param['values'][0]))

    for ua in unknown_args:
        unknown_args_wo_eq.extend(ua.split('='))

    if parse_args.resume_from_same_run_id:
        assert parse_args.run_id is not None
        parse_args.resume_from_run_id = True
        parse_args.allow_resume_from_same_run_id = True
        unknown_args_wo_eq.extend(
            ['--wandb.resume', 'allow', '--wandb.id', parse_args.run_id])

    if parse_args.run_id is not None:
        cfg = get_config_from_run_id(parse_args.run_id)
        if 'resume' in cfg['training']:
            del cfg['training']['resume']
        if parse_args.resume_from_run_id:
            if len(parse_args.run_id.split(':')) == 2:
                run_id = parse_args.run_id
            else:
                run_id = '{}:latest'.format(parse_args.run_id)
            unknown_args_wo_eq.extend([
                '--training.resume.type', 'wandb', '--training.resume.from',
                run_id
            ])
    if parse_args.checkpoint is not None:
        assert os.path.exists(parse_args.checkpoint)
        unknown_args_wo_eq.extend([
            '--training.resume.type', 'local', '--training.resume.path', parse_args.checkpoint
        ])
        parse_args.run_id = hashlib.md5(open(parse_args.checkpoint,'rb').read()).hexdigest()[:8]
        cfg = get_config_from_checkpoint(parse_args.checkpoint)

    else:
        cfg = config_util.load_config(parse_args.config)
    if parse_args.check_memory:
        unknown_args_wo_eq.extend(
            ['--training.eval.every', '2', '--training.visualize.every', '2'])

    if parse_args.check_gradient:
        unknown_args_wo_eq.extend([
            '--trainer.kwargs.is_check_gradient_scale', 'true',
            '--training.batch_size', '5'
        ])
        parse_args.skip_first_eval_and_vis = True

    if parse_args.option_combo_config:
        unknown_args_wo_eq = option_encoder.encode(
            unknown_args_wo_eq, parse_args.option_combo_config)
    cfg = config_util.update_dict_with_options(cfg, unknown_args_wo_eq)

    if parse_args.dryrun or parse_args.check_memory:
        os.environ['WANDB_MODE'] = 'dryrun'
    if parse_args.no_dryrun and 'WANDB_MODE' in os.environ:
        del os.environ['WANDB_MODE']

    if parse_args.mode == 'gen_mesh':
        cfg['wandb']['job_type'] = 'gen_mesh'

    return cfg


if __name__ == '__main__':
    parse_args, unknown_args = parse(sys.argv[1:])

    torch.manual_seed(parse_args.seed)
    np.random.seed(parse_args.seed)
    random.seed(parse_args.seed)

    if parse_args.advanced_reproducibility:
        torch.cuda.manual_seed_all(parse_args.seed)
        torch.cuda.manual_seed(parse_args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True)

    cfg = decode_args(parse_args, unknown_args)
    gen_mesh(cfg, parse_args)
