import os
import sys
import torch
from torch import optim
import argparse
from torch.utils import data
from data import dataset
from model import movenet
from trainer import trainer
import dotenv
import wandb
from collections import defaultdict
import numpy as np
from utils import config_util
from utils import checkpoint_util
from datetime import datetime
import random
dotenv.load_dotenv()

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def main(cfg):
    non_null_wandb = {}
    for key, item in cfg['wandb'].items():
        if item is not None:
            non_null_wandb[key] = item
    run = wandb.init(**non_null_wandb)
    wandb.config.update(cfg)

    date_str = datetime.now().strftime(('%Y%m%d_%H%M%S'))
    if cfg['training']['resume'] == 'local':
        assert cfg['training']['outdir'] == cfg['training']['outdir'].format(
            'aaa')
        outdir = cfg['training']['outdir']
    else:
        outdir = cfg['training']['outdir'].format(date_str)

    device = 'cuda'

    moving_dataset = dataset.Simple2DCanonicalMovingDataset(
        cfg['data']['dataset_path'], cfg['data']['filename'],
        cfg['data']['points_n'])
    dataset_len = len(moving_dataset)
    ratio = cfg['data']['train_val_test_ratio']
    sequences = [dataset_len * ratio['train'], dataset_len * ratio['val']]
    sequences = list(map(int, sequences))
    sequences.append(dataset_len - sum(sequences))
    train_dataset, val_dataset, _ = torch.utils.data.dataset.random_split(
        moving_dataset, sequences)
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True)
    val_dataloader = data.DataLoader(
        val_dataset,
        batch_size=cfg['training']['eval']['batch_size'],
        shuffle=False)
    vis_dataloader = data.DataLoader(
        val_dataset,
        batch_size=cfg['training']['visualize']['batch_size'],
        shuffle=False)
    batch_vis = next(iter(vis_dataloader))

    if cfg['trainer']['kwargs']['train_explicit']:
        model = movenet.MoveNetAtlasNetAutoEncoder(
            **cfg['model']['kwargs']).to(device)
    else:
        raise NotImplementedError
    wandb.watch(model)
    optimizers = {}
    for group, parameters in model.parameter_groups.items():
        optimizers[group] = optim.Adam(
            parameters,
            lr=cfg['training']['optimizer_groups'][group]['learning_rate'])

    epoch = 0
    iters = 1

    model_selection_mode = cfg['training']['eval']['model_selection_mode']
    assert model_selection_mode in ['min', 'max']
    model_selection_sign = 1 if model_selection_mode == 'min' else -1
    metric_val_best = model_selection_sign * np.inf

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if cfg['training']['resume'] is not None:
        artifacts = {
            'model': {
                'movenet': model
            },
            'scalar': {
                'iters': iters,
                'epochs': epoch,
                'metric_val_best': metric_val_best
            },
            'config': cfg,
            'optimizer': optimizers
        }
        if cfg['training'].get('resume') and cfg['training'].get('resume').get(
                'type'):
            if cfg['training']['resume']['type'] == 'local':
                assert os.path.exists(os.path.join(outdir, 'model_latest.pth'))
                assert cfg['training']['resume']['from'] is None
                checkpoint_util.load(os.path.join(outdir, 'model_latest.pth'),
                                     artifacts)
            elif cfg['training']['resume']['type'] == 'wandb':
                assert os.getenv('WANDB_MODE', '') != 'dryrun'
                wandb_artifact_name = cfg['training']['resume']['from']
                assert len(wandb_artifact_name.split(
                    ':')) == 2, 'You need version (e.g. name:version)'
                checkpoint_util.load(artifacts,
                                     wandb_artifact=True,
                                     wandb_run=run,
                                     wandb_artifact_name=wandb_artifact_name)
            else:
                raise NotImplementedError
        iters = artifacts['scalar']['iters']
        epoch = artifacts['scalar']['epochs']
        metric_val_best = artifacts['scalar']['metric_val_best']

    print(iters, epoch, metric_val_best)

    # Send to GPU
    model.to(device)
    trainer_ob = trainer.Trainer(model, optimizers, device,
                                 **cfg['trainer']['kwargs'])
    terminate_flag = False

    while True:
        for step, batch in enumerate(train_dataloader):
            losses = trainer_ob.train_step(batch)
            losses_cpu = {
                'train/' + key: value.mean().detach().cpu().numpy()
                for key, value in losses.items()
            }

            if iters % cfg['training']['every'] == 0:
                losses_cpu['epoch'] = epoch
                losses_cpu['step'] = iters
                wandb.log(losses_cpu)

            if iters % cfg['training']['eval']['every'] == 0:
                loss_dict = defaultdict(lambda: [])
                for step, batch in enumerate(val_dataloader):
                    losses = trainer_ob.eval_step(batch)
                    for key, value in losses.items():
                        loss_dict[key].append(value.detach().cpu().numpy())

                losses_cpu_eval = {
                    'val/' + key: np.array(value).mean()
                    for key, value in loss_dict.items()
                }
                losses_cpu_eval['epoch'] = epoch
                losses_cpu_eval['step'] = iters
                wandb.log(losses_cpu_eval)
                metric_val = np.array(loss_dict[
                    cfg['training']['eval']['model_selection_metric']]).mean()
                if -model_selection_sign * (metric_val - metric_val_best) > 0:
                    metric_val_best = metric_val
                    artifacts = {
                        'model': {
                            'movenet': model
                        },
                        'scalar': {
                            'iters': iters,
                            'epochs': epoch,
                            'metric_val_best': metric_val_best
                        },
                        'config': cfg,
                        'optimizer': optimizers
                    }
                    checkpoint_util.save(os.path.join(outdir,
                                                      'model_best.pth'),
                                         artifacts,
                                         wandb_artifact=True,
                                         wandb_run=run,
                                         wandb_artifact_suffix='_best')

            if iters % cfg['training']['checkpoint']['every'] == 0:
                artifacts = {
                    'model': {
                        'movenet': model
                    },
                    'scalar': {
                        'iters': iters,
                        'epochs': epoch,
                        'metric_val_best': metric_val_best
                    },
                    'config': cfg,
                    'optimizer': optimizers
                }
                checkpoint_util.save(os.path.join(
                    outdir, 'model_{}.pth'.format(iters)),
                                     artifacts,
                                     wandb_artifact=True,
                                     wandb_run=run)
                checkpoint_util.save(os.path.join(outdir, 'model_latest.pth'),
                                     artifacts)
            if iters % cfg['training']['visualize']['every'] == 0:
                ret = trainer_ob.visualize(batch_vis)
                vis = {}
                for r in ret:
                    typ = r['type']
                    if typ == 'image':
                        vis[r['desc']] = [
                            wandb.Image(img) for img in r['data']
                        ]
                    elif typ == 'array':
                        vis['pred_slide'] = r['data']
                        print(r['data'])
                wandb.log(vis)

            iters += 1
            if cfg['training']['terminate_iters'] > 0 and iters > cfg[
                    'training']['terminate_iters']:
                terminate_flag = True
                break

        epoch += 1
        if terminate_flag or cfg['training'][
                'terminate_epoch'] > 0 and epoch > cfg['training'][
                    'terminate_epoch']:
            break


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    return parser.parse_known_args()


if __name__ == '__main__':
    parse_args, unknown_args = parse()
    print(unknown_args)
    unknown_args_wo_eq = []
    for ua in unknown_args:
        unknown_args_wo_eq.extend(ua.split('='))
    cfg = config_util.load_config(parse_args.config)
    cfg = config_util.update_dict_with_options(cfg, unknown_args_wo_eq)
    main(cfg)
