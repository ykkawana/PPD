from trainer import config
import torch
from torch.utils import data
from data import imnet_shapenet_dataset


class Config(config.Config):
    def __init__(self, cfg, device='cuda'):
        super().__init__(cfg, device=device)

    def init_dataset_loader(self):
        train_kwargs = self.cfg['data']['train']['kwargs']
        train_kwargs.update(self.cfg['data']['common']['kwargs'])
        self.train_dataset = imnet_shapenet_dataset.IMNetShapeNetDataset(
            primitive_num=self.cfg['model']['kwargs']['primitive_num'],
            **train_kwargs)
        val_kwargs = self.cfg['data']['val']['kwargs']
        val_kwargs.update(self.cfg['data']['common']['kwargs'])
        self.val_dataset = imnet_shapenet_dataset.IMNetShapeNetDataset(
            primitive_num=self.cfg['model']['kwargs']['primitive_num'],
            **val_kwargs)
        self.train_dataloader = data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg['training']['batch_size'],
            num_workers=10,
            pin_memory=True,
            shuffle=True)
        self.val_dataloader = data.DataLoader(
            self.val_dataset,
            batch_size=self.cfg['training']['eval']['batch_size'],
            shuffle=False)
        self.vis_dataloader = data.DataLoader(
            self.val_dataset,
            batch_size=self.cfg['training']['visualize']['batch_size'],
            shuffle=False)
