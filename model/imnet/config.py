from trainer import config
import torch
from torch.utils import data
from data import imnet_shapenet_dataset
import numpy as np
import random
from functools import partial
from model.imnet import mesh_generator

def seed_worker(worker_id, dataset_seed):
    worker_seed = dataset_seed + torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Config(config.Config):
    def __init__(self, cfg, device='cuda'):
        super().__init__(cfg, device=device)

    def init_dataset_loader(self, args):
        if args.advanced_reproducibility:
            g = torch.Generator()
            g.manual_seed(args.data_seed)
            repro_kwargs = dict(
                worker_init_fn=partial(seed_worker, dataset_seed=args.data_seed),
                # generator=g,
            )
        else:
            repro_kwargs = {}

        train_kwargs = self.cfg['data']['train']['kwargs']
        train_kwargs.update(self.cfg['data']['common']['kwargs'])
        train_dataloader_kwargs = self.cfg['data']['train'].get(
            'dataloader_kwargs', {})
        train_dataloader_kwargs.update(self.cfg['data']['common'].get(
            'dataloader_kwargs', {}))
        self.train_dataset = imnet_shapenet_dataset.IMNetShapeNetDataset(
            primitive_num=self.cfg['model']['kwargs']['primitive_num'],
            **train_kwargs)
        val_kwargs = self.cfg['data']['val']['kwargs']
        val_kwargs.update(self.cfg['data']['common']['kwargs'])
        val_dataloader_kwargs = self.cfg['data']['val'].get(
            'dataloader_kwargs', {})
        val_dataloader_kwargs.update(self.cfg['data']['common'].get(
            'dataloader_kwargs', {}))
        self.val_dataset = imnet_shapenet_dataset.IMNetShapeNetDataset(
            primitive_num=self.cfg['model']['kwargs']['primitive_num'],
            **val_kwargs)
        test_kwargs = self.cfg['data']['test']['kwargs']
        test_kwargs.update(self.cfg['data']['common']['kwargs'])
        test_dataloader_kwargs = self.cfg['data']['test'].get(
            'dataloader_kwargs', {})
        test_dataloader_kwargs.update(self.cfg['data']['common'].get(
            'dataloader_kwargs', {}))
        self.test_dataset = imnet_shapenet_dataset.IMNetShapeNetDataset(
            primitive_num=self.cfg['model']['kwargs']['primitive_num'],
            **test_kwargs)
        self.train_dataloader = data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg['training']['batch_size'],
            num_workers=10,
            pin_memory=True,
            shuffle=True,
            **repro_kwargs,
            **train_dataloader_kwargs)
        self.val_dataloader = data.DataLoader(
            self.val_dataset,
            batch_size=self.cfg['training']['eval']['batch_size'],
            shuffle=False,
            **val_dataloader_kwargs)
        self.test_dataloader = data.DataLoader(
            self.test_dataset,
            batch_size=self.cfg['training'].get(
                'test',
                {'batch_size': self.cfg['training']['eval']['batch_size']
                 })['batch_size'],
            shuffle=False,
            **test_dataloader_kwargs)
        self.vis_dataloader = data.DataLoader(
            self.val_dataset,
            batch_size=self.cfg['training']['visualize']['batch_size'],
            shuffle=False,
            **val_dataloader_kwargs)

    def init_mesh_generator(self):
        self.mesh_generator_ob = mesh_generator.MeshGenerator(
            self.model, self.device, self.trainer_ob,
            **self.cfg.get('mesh_generation', {
                'kwargs': {}
            }).get('kwargs', {}))