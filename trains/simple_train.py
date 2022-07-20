# %%
import os
import sys
import torch
from torch import optim
import argparse
from torch.utils import data
from dataset import dataset
from model import movenet
from trainer import trainer
import dotenv
import wandb
from collections import defaultdict
import numpy as np
from utils import config_util
dotenv.load_dotenv()


# %%
def main(cfg):
    wandb.init()
    wandb.config.update(cfg)

    device = 'cuda'
    train_dataset_ob = dataset.simplest2D_dataset(cfg['dataset_path'])
    train_dataloader = data.DataLoader(train_dataset_ob,
                                       batch_size=cfg['batch_size'],
                                       shuffle=True)
    val_dataset_ob = dataset.simplest2D_dataset(cfg['dataset_path'])
    val_dataloader = data.DataLoader(val_dataset_ob,
                                     batch_size=cfg['batch_size'],
                                     shuffle=False)
    vis_dataset_ob = dataset.simplest2D_dataset(cfg['dataset_path'])
    vis_dataloader = data.DataLoader(vis_dataset_ob,
                                     batch_size=cfg['batch_size'],
                                     shuffle=False)
    batch_vis = next(iter(vis_dataloader))

    if cfg['train_explicit']:
        model = movenet.MoveNetAtlasNetAutoEncoder(
            primitive_num=cfg['max_primitive_num'],
            is_atlasnet_template_sphere=cfg['is_atlasnet_template_sphere'],
            is_atlasnetv2=cfg['is_atlasnetv2']).to(device)
    else:
        model = movenet.MoveNetAutoEncoder(
            primitive_num=cfg['max_primitive_num']).to(device)
    wandb.watch(model)
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])

    trainer_ob = trainer.Trainer(
        model,
        optimizer,
        device,
        train_explicit=cfg['train_explicit'],
        learn_by_moving=cfg['learn_by_moving'],
        learn_by_moving_param_supervised=cfg[
            'learn_by_moving_param_supervised'],
        learn_only_by_chamfer_distance=cfg['learn_only_by_chamfer_distance'])
    epoch = 0
    while True:
        for step, batch in enumerate(train_dataloader):
            losses = trainer_ob.train_step(batch)
            losses_cpu = {
                'train/' + key: value.mean().detach().cpu().numpy()
                for key, value in losses.items()
            }

            if step % cfg['train_log_steps'] == 0:
                wandb.log(losses_cpu)

        loss_dict = defaultdict(lambda: [])
        for step, batch in enumerate(val_dataloader):
            losses = trainer_ob.eval_step(batch)
            for key, value in losses.items():
                loss_dict[key].append(value.detach().cpu().numpy())

            wandb.log({
                'val/' + key: np.array(value).mean()
                for key, value in loss_dict.items()
            })

        ret = trainer_ob.visualize(batch)
        vis = {}
        for r in ret:
            typ = r['type']
            if typ == 'image':
                vis['image'] = [wandb.Image(img) for img in r['data']]
            elif typ == 'array':
                vis['param'] = r['data']
                print(r['data'])
        wandb.log(vis)

        epoch += 1


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    return parser.parse_known_args()


if __name__ == '__main__':
    parse_args, unknown_args = parse()
    cfg = config_util.load_config(parse_args.config)
    cfg = config_util.update_dict_with_options(cfg, unknown_args)
    main(cfg)

# %%
