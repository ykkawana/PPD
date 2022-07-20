from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class mlpAdj(nn.Module):
    def __init__(self, nlatent=128, dim=2):
        """Atlas decoder"""

        super(mlpAdj, self).__init__()
        self.nlatent = nlatent
        self.conv1 = torch.nn.Conv1d(self.nlatent, self.nlatent, 1)
        self.conv2 = torch.nn.Conv1d(self.nlatent, self.nlatent // 2, 1)
        self.conv3 = torch.nn.Conv1d(self.nlatent // 2, self.nlatent // 4, 1)
        self.conv4 = torch.nn.Conv1d(self.nlatent // 4, dim, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.nlatent)
        self.bn2 = torch.nn.BatchNorm1d(self.nlatent // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.nlatent // 4)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # normalize the point range to -0.5 to 0.5 by / 2
        x = self.th(self.conv4(x)) / 2
        return x


class AtlasNet(nn.Module):
    """Atlas net auto encoder"""
    def __init__(self,
                 c_dim=128,
                 dim=2,
                 sampling_point=1024,
                 primitives_num=2,
                 template_sphere=False):

        super(AtlasNet, self).__init__()

        self.sampling_point = sampling_point
        self.primitive_num = primitives_num
        self.c_dim = c_dim
        self.template_sphere = template_sphere

        if self.template_sphere and dim == 3:
            self.patch_dim = 3
        else:
            self.patch_dim = 2
        #encoder and decoder modules
        #==============================================================================
        self.decoder = nn.ModuleList([
            mlpAdj(nlatent=self.patch_dim + c_dim, dim=dim)
            for i in range(0, self.primitive_num)
        ])
        #==============================================================================

    def forward(self, x):

        outs = []
        patches = []
        for i in range(0, self.primitive_num):

            #random patch
            #==========================================================================
            if self.template_sphere:
                rand_grid = torch.FloatTensor(
                    x.size(0), self.patch_dim,
                    self.sampling_point // self.primitive_num).to(x.device)
                rand_grid.data.normal_(0, 1)
                rand_grid = rand_grid / torch.sqrt(
                    torch.sum(rand_grid**2, dim=1, keepdim=True))
            else:
                rand_grid = torch.FloatTensor(
                    x.size(0), 2,
                    self.sampling_point // self.primitive_num).to(x.device)
                rand_grid.data.uniform_(0, 1)
            #==========================================================================

            #cat with latent vector and decode
            #==========================================================================
            y = x.unsqueeze(2).expand(x.size(0), x.size(1),
                                      rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y).transpose(2, 1).contiguous())
            #==========================================================================

        return outs


class patchDeformationMLP(nn.Module):
    """deformation of a 2D patch into a 3D surface"""
    def __init__(self, patchDim=2, patchDeformDim=3, tanh=True):

        super(patchDeformationMLP, self).__init__()
        layer_size = 128
        self.tanh = tanh
        self.conv1 = torch.nn.Conv1d(patchDim, layer_size, 1)
        self.conv2 = torch.nn.Conv1d(layer_size, layer_size, 1)
        self.conv3 = torch.nn.Conv1d(layer_size, patchDeformDim, 1)
        self.bn1 = torch.nn.BatchNorm1d(layer_size)
        self.bn2 = torch.nn.BatchNorm1d(layer_size)
        self.th = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        if self.tanh:
            x = self.th(self.conv3(x))
        else:
            x = self.conv3(x)
        return x


class AtlasNetV2(nn.Module):
    """Atlas net auto encoder"""
    def __init__(self,
                 c_dim=128,
                 dim=2,
                 sampling_point=1024,
                 primitives_num=2,
                 template_sphere=False):

        super(AtlasNetV2, self).__init__()

        self.sampling_point = sampling_point
        self.primitive_num = primitives_num
        self.c_dim = c_dim
        self.template_sphere = template_sphere
        self.patchDeformDim = 3

        if self.template_sphere and dim == 3:
            self.patch_dim = 3
        else:
            self.patch_dim = 2
        #encoder and decoder modules
        #==============================================================================
        self.decoder = nn.ModuleList([
            mlpAdj(nlatent=self.patchDeformDim + c_dim, dim=dim)
            for i in range(0, self.primitive_num)
        ])
        self.patchDeformation = nn.ModuleList(
            patchDeformationMLP(patchDim=self.patch_dim,
                                patchDeformDim=self.patchDeformDim)
            for i in range(0, self.primitive_num))
        #==============================================================================

    def forward(self, x):

        outs = []
        patches = []
        for i in range(0, self.primitive_num):

            #random patch
            #==========================================================================
            if self.template_sphere:
                rand_grid = torch.FloatTensor(
                    x.size(0), self.patch_dim,
                    self.sampling_point // self.primitive_num).to(x.device)
                rand_grid.data.normal_(0, 1)
                rand_grid = rand_grid / torch.sqrt(
                    torch.sum(rand_grid**2, dim=1, keepdim=True))
            else:
                rand_grid = torch.FloatTensor(
                    x.size(0), 2,
                    self.sampling_point // self.primitive_num).to(x.device)
                rand_grid.data.uniform_(0, 1)

            rand_grid = self.patchDeformation[i](rand_grid)
            #==========================================================================

            #cat with latent vector and decode
            #==========================================================================
            y = x.unsqueeze(2).expand(x.size(0), x.size(1),
                                      rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y).transpose(2, 1).contiguous())
            #==========================================================================

        return outs
