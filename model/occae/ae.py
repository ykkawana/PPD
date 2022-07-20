from torch import nn
from model.imnet import encoder, generator
from model import paramnet, pointnet, decoder
import torch
from torch.nn import functional as F
from utils import geometry
import numpy as np
from collections import OrderedDict
from model.imnet import gumbel_softmax
import importlib


class OccAutoEncoder(nn.Module):
    def __init__(self,
                 primitive_num=3,
                 latent_dim=32,
                 dim=3,
                 decoder_hidden_size=128,
                 decoder_depth=3,
                 decoder_leaky_relu=False,
                 pointnet_hidden_size=128,
                 occnet_decoder_type='model.decoder.DecoderBatchNorm',
                 pointnet_depth=5):
        super().__init__()
        self.primitive_num = primitive_num
        self.decoder_hidden_size = decoder_hidden_size
        self.latent_dim = latent_dim
        self.dim = dim
        self.pointnet_hidden_size = pointnet_hidden_size
        self.pointnet_depth = pointnet_depth
        self.occnet_decoder_type = occnet_decoder_type

        self.encoder = pointnet.ResnetPointnet(
            c_dim=self.latent_dim,
            dim=(self.dim + 1),
            hidden_dim=self.pointnet_hidden_size,
            depth=self.pointnet_depth)
        tmp = self.occnet_decoder_type.split('.')
        module_path = '.'.join(tmp[:-1])
        class_name = tmp[-1]
        self.generator = getattr(importlib.import_module(module_path),
                                 class_name)(out_dim=1,
                                             c_dim=self.latent_dim,
                                             z_dim=0,
                                             dim=dim,
                                             hidden_size=decoder_hidden_size,
                                             leaky=decoder_leaky_relu,
                                             depth=decoder_depth)

        self.model_groups = {}
        self.model_groups['generator'] = [self.generator, self.encoder]
        self.parameter_groups = {}
        self.parameter_groups['generator'] = []
        for model in self.model_groups['generator']:
            self.parameter_groups['generator'].extend(model.parameters())

    def forward(self,
                inputs,
                coord=None,
                mode='generator',
                return_generator=False):
        if mode == 'generator':
            ret = self.encoder(inputs)
            if not return_generator:
                return ret
            z = ret['latent']
            occ = self.generator(coord, None, z)
            # occ[:, :, 1:] = occ[:, :, 1:] * 100
            ret['occupancy'] = occ
            return ret
        else:
            raise NotImplementedError
