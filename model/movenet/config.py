from trainer import config
import torch
from torch.utils import data
from data import dataset


class Config(config.Config):
    def __init__(self, cfg, device='cuda'):
        cfg['model']['class'] = 'model.movenet.movenet.MoveNetAutoEncoder'
        super().__init__(cfg, device=device)

    def init_dataset_loader(self):
        moving_dataset = dataset.Simple2DImExCanonicalMovingDataset(
            **self.cfg['data']['kwargs'])
        dataset_len = len(moving_dataset)
        ratio = self.cfg['data']['train_val_test_ratio']
        sequences = [dataset_len * ratio['train'], dataset_len * ratio['val']]
        sequences = list(map(int, sequences))
        sequences.append(dataset_len - sum(sequences))
        self.train_dataset, self.val_dataset, _ = torch.utils.data.dataset.random_split(
            moving_dataset, sequences)
        self.train_dataloader = data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg['training']['batch_size'],
            shuffle=True)
        self.val_dataloader = data.DataLoader(
            self.val_dataset,
            batch_size=self.cfg['training']['eval']['batch_size'],
            shuffle=False)
        self.vis_dataloader = data.DataLoader(
            self.val_dataset,
            batch_size=self.cfg['training']['visualize']['batch_size'],
            shuffle=False)
