import torch.nn as nn
import torch.nn.functional as F
import torch
from model.layers import (ResnetBlockFC, CResnetBlockConv1d, CBatchNorm1d,
                          CBatchNorm1d_legacy, ResnetBlockConv1d)


class Decoder(nn.Module):
    ''' Decoder class.

    It does not perform any form of normalization.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''
    def __init__(self,
                 dim=3,
                 z_dim=128,
                 c_dim=128,
                 hidden_size=128,
                 leaky=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        # Submodules
        self.fc_p = nn.Linear(dim, hidden_size)

        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        if not c_dim == 0:
            self.fc_c = nn.Linear(c_dim, hidden_size)

        self.block0 = ResnetBlockFC(hidden_size)
        self.block1 = ResnetBlockFC(hidden_size)
        self.block2 = ResnetBlockFC(hidden_size)
        self.block3 = ResnetBlockFC(hidden_size)
        self.block4 = ResnetBlockFC(hidden_size)

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c=None, **kwargs):
        batch_size, T, D = p.size()

        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(1)
            net = net + net_z

        if self.c_dim != 0:
            net_c = self.fc_c(c).unsqueeze(1)
            net = net + net_c

        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)
        net = self.block3(net)
        net = self.block4(net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out


class DecoderCBatchNorm(nn.Module):
    ''' Decoder with conditional batch normalization (CBN) class.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        legacy (bool): whether to use the legacy structure
    '''
    def __init__(self,
                 dim=3,
                 z_dim=128,
                 c_dim=128,
                 hidden_size=256,
                 depth=5,
                 out_dim=3,
                 leaky=False,
                 legacy=False):
        super().__init__()
        self.z_dim = z_dim
        self.depth = depth
        self.out_dim = out_dim
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.blocks = nn.ModuleList([
            CResnetBlockConv1d(c_dim, hidden_size, legacy=False)
            for _ in range(self.depth)
        ])

        if not legacy:
            self.bn = CBatchNorm1d(c_dim, hidden_size)
        else:
            self.bn = CBatchNorm1d_legacy(c_dim, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, self.out_dim, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z

        for block in self.blocks:
            net = block(net, c)

        out = self.fc_out(self.actvn(self.bn(net, c)))
        #out = out.squeeze(1)
        out = out.transpose(1, 2).contiguous()

        return out


class DecoderCBatchNorm2(nn.Module):
    ''' Decoder with CBN class 2.

    It differs from the previous one in that the number of blocks can be
    chosen.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        n_blocks (int): number of ResNet blocks
    '''
    def __init__(self, dim=3, z_dim=0, c_dim=128, hidden_size=256, n_blocks=5):
        super().__init__()
        self.z_dim = z_dim
        if z_dim != 0:
            self.fc_z = nn.Linear(z_dim, c_dim)

        self.conv_p = nn.Conv1d(dim, hidden_size, 1)
        self.blocks = nn.ModuleList(
            [CResnetBlockConv1d(c_dim, hidden_size) for i in range(n_blocks)])

        self.bn = CBatchNorm1d(c_dim, hidden_size)
        self.conv_out = nn.Conv1d(hidden_size, 1, 1)
        self.actvn = nn.ReLU()

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.conv_p(p)

        if self.z_dim != 0:
            c = c + self.fc_z(z)

        for block in self.blocks:
            net = block(net, c)

        out = self.conv_out(self.actvn(self.bn(net, c)))
        out = out.squeeze(1)

        return out


class DecoderCBatchNormNoResnet(nn.Module):
    ''' Decoder CBN with no ResNet blocks class.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''
    def __init__(self,
                 dim=3,
                 z_dim=128,
                 c_dim=128,
                 hidden_size=256,
                 leaky=False):
        super().__init__()
        self.z_dim = z_dim
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.fc_0 = nn.Conv1d(hidden_size, hidden_size, 1)
        self.fc_1 = nn.Conv1d(hidden_size, hidden_size, 1)
        self.fc_2 = nn.Conv1d(hidden_size, hidden_size, 1)
        self.fc_3 = nn.Conv1d(hidden_size, hidden_size, 1)
        self.fc_4 = nn.Conv1d(hidden_size, hidden_size, 1)

        self.bn_0 = CBatchNorm1d(c_dim, hidden_size)
        self.bn_1 = CBatchNorm1d(c_dim, hidden_size)
        self.bn_2 = CBatchNorm1d(c_dim, hidden_size)
        self.bn_3 = CBatchNorm1d(c_dim, hidden_size)
        self.bn_4 = CBatchNorm1d(c_dim, hidden_size)
        self.bn_5 = CBatchNorm1d(c_dim, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z

        net = self.actvn(self.bn_0(net, c))
        net = self.fc_0(net)
        net = self.actvn(self.bn_1(net, c))
        net = self.fc_1(net)
        net = self.actvn(self.bn_2(net, c))
        net = self.fc_2(net)
        net = self.actvn(self.bn_3(net, c))
        net = self.fc_3(net)
        net = self.actvn(self.bn_4(net, c))
        net = self.fc_4(net)
        net = self.actvn(self.bn_5(net, c))
        out = self.fc_out(net)
        out = out.squeeze(1)

        return out


class DecoderBatchNorm(nn.Module):
    ''' Decoder with batch normalization class.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''
    def __init__(self,
                 dim=3,
                 z_dim=128,
                 c_dim=128,
                 hidden_size=256,
                 depth=5,
                 out_dim=3,
                 leaky=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.depth = depth
        self.out_dim = out_dim

        # Submodules
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        if self.c_dim != 0:
            self.fc_c = nn.Linear(c_dim, hidden_size)
        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.blocks = nn.ModuleList(
            [ResnetBlockConv1d(hidden_size) for _ in range(self.depth)])

        self.bn = nn.BatchNorm1d(hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, out_dim, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z

        if self.c_dim != 0:
            net_c = self.fc_c(c).unsqueeze(2)
            net = net + net_c

        for block in self.blocks:
            net = block(net)

        out = self.fc_out(self.actvn(self.bn(net)))

        # batch, points, occupancy=out_dim (primitive_num)
        out = out.transpose(1, 2).contiguous()
        return out


class SDFGenerator(torch.nn.Module):
    def __init__(self,
                 dim=3,
                 c_dim=128,
                 hidden_size=256,
                 depth=6,
                 out_dim=2,
                 last_tanh=False):
        super(SDFGenerator, self).__init__()

        self.layers1 = None
        self.layers2 = None
        self.last_tanh = last_tanh

        assert depth % 2 == 0

        self.latent_channels = c_dim
        self.hidden_channels = hidden_size
        self.num_layers = depth

        in_channels = dim
        out_channels = self.hidden_channels

        self.lins = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(self.num_layers):
            self.lins.append(nn.Linear(in_channels, out_channels))
            self.norms.append(nn.LayerNorm(out_channels))

            if i == (self.num_layers // 2) - 1:
                in_channels = self.hidden_channels + dim
            else:
                in_channels = self.hidden_channels

            if i == self.num_layers - 2:
                out_channels = out_dim

        self.z_lin1 = nn.Linear(self.latent_channels, self.hidden_channels)
        self.z_lin2 = nn.Linear(self.latent_channels, self.hidden_channels)

    def forward(self, p, z, c):
        # pos: [batch_size, num_points, 2 or 3]
        # z: [batch_size, latent_channels]

        pos = p
        assert pos.dim() == 3
        assert pos.size(-1) in [2, 3]

        assert c.dim() == 2
        assert c.size(-1) == self.latent_channels

        assert pos.size(0) == c.size(0)

        x = pos
        for i, (lin, norm) in enumerate(zip(self.lins, self.norms)):
            if i == self.num_layers // 2:
                x = torch.cat([x, pos], dim=-1)

            x = lin(x)

            if i == 0:
                x = self.z_lin1(c).unsqueeze(1) + x

            if i == self.num_layers // 2:
                x = self.z_lin2(c).unsqueeze(1) + x

            if i < self.num_layers - 1:
                x = norm(x) if self.norms else x
                x = F.relu(x)

        if self.last_tanh:
            # as the input space is normalized from -0.5 to 0.5
            x = torch.tanh(x) / 2
        return x
