import torch
import contextlib


@contextlib.contextmanager
def freeze_models(models):
    for model in models:
        for param in model.parameters():
            param.requires_grad = False
    yield
    for model in models:
        for param in model.parameters():
            param.requires_grad = True
