import importlib
from torch import optim


class Config:
    def __init__(self, cfg, *args, device='cuda', **kwargs):
        self.cfg = cfg
        self.device = device
        autoencoder_class_path = cfg['model']['class']
        tmp = autoencoder_class_path.split('.')
        self.autoencoder_module_path = '.'.join(tmp[:-1])
        self.autoencoder_class_name = tmp[-1]
        self.model = getattr(
            importlib.import_module(self.autoencoder_module_path),
            self.autoencoder_class_name)(**cfg['model']['kwargs']).to(device)
        self.optimizers = {}
        for group, parameters in self.model.parameter_groups.items():
            self.optimizers[group] = optim.Adam(
                parameters,
                lr=cfg['training']['optimizer_groups'][group]['learning_rate'],
                **cfg['training']['optimizer_groups'][group].get('kwargs', {}))
        self.module_path = '.'.join(tmp[:-2])

    def init_trainer(self, *args, **kwargs):
        trainer_module_path = '.'.join([self.module_path, 'trainer'])
        if not importlib.util.find_spec(trainer_module_path):
            trainer_module_path = 'trainer.imex_trainer'
        self.trainer_ob = getattr(importlib.import_module(trainer_module_path),
                                  'Trainer')(self.model, self.optimizers,
                                             self.device,
                                             **self.cfg['trainer']['kwargs'])

    def init_dataset_loader(self, *args, **kwargs):
        raise NotImplementedError

    def init_mesh_generator(self, *args, **kwargs):
        raise NotImplementedError

    def get_artifact(self, iters, epoch, metric_val_best, *args, **kwargs):
        artifacts = {
            'model': {
                'movenet': self.model
            },
            'scalar': {
                'iters': iters,
                'epochs': epoch,
                'metric_val_best': metric_val_best
            },
            'config': self.cfg,
            'optimizer': self.optimizers
        }
        return artifacts
