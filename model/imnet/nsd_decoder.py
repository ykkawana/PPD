import torch
from torch import nn
from torch.nn import functional as F
from model.layers import CBatchNorm1d
from model.imnet import power_spherical
from utils import geometry
import numpy as np
from model import layers


class DecoderNSD(nn.Module):
    def __init__(self,
                 dim=3,
                 z_dim=128,
                 c_dim=128,
                 hidden_size=256,
                 depth=5,
                 out_dim=3,
                 leaky=False,
                 legacy=False,
                 add_primitive_code=False):
        super().__init__()
        self.out_dim = out_dim
        self.c_dim = c_dim
        self.add_primitive_code = add_primitive_code
        self.c64 = int(hidden_size / self.out_dim)
        self.fc_c = nn.Linear(c_dim, self.c64 * self.out_dim)
        self.fc_p = nn.Conv1d(dim, self.c64 * self.out_dim, 1)

        layers = []
        if self.add_primitive_code:
            self.pm_wise_layer_input_dim = self.c64 + self.out_dim,
        else:
            self.pm_wise_layer_input_dim = self.c64

        layers.extend([
            PrimitiveWiseLinear(self.out_dim,
                                self.pm_wise_layer_input_dim,
                                self.c64,
                                act='relu')
        ])
        layers.extend([
            PrimitiveWiseLinear(self.out_dim, self.c64, self.c64, act='relu')
            for _ in range(depth - 1)
        ])
        layers.append(
            PrimitiveWiseLinear(self.out_dim,
                                self.c64,
                                1,
                                act=('leaky' if leaky else 'relu')))
        self.decoder = nn.Sequential(*layers)

    def forward(self, p, _, c):
        B, P, dim = p.shape
        B2, ldim = c.shape
        assert B == B2

        # B, dim, P
        p = p.transpose(1, 2)
        # B, n_pm * c64, P
        net = self.fc_p(p)

        # B, ldim, 1
        net_c = self.fc_c(c).unsqueeze(2)
        # B, n_pm * c64, P
        net = net + net_c

        if self.add_primitive_code:
            identity = torch.eye(self.out_dim,
                                 dtype=net.dtype,
                                 device=net.device).reshape(
                                     1, self.out_dim, self.out_dim,
                                     1).expand(B, -1, -1, P)
            net = torch.cat(
                [net.reshape(B, self.out_dim, self.c64, P), identity], dim=2)
            net = net.reshape(B, self.out_dim * (self.c64 + self.out_dim), P)

        # B, P, n_pm
        net = self.decoder(net).view(B, self.out_dim,
                                     P).transpose(1, 2).contiguous()

        return net


class PrimitiveWiseLinear(nn.Module):
    def __init__(self,
                 n_primitives,
                 input_channels,
                 output_channels,
                 act='leaky',
                 bias=True):
        """
    Calculate n_primitive wise feature encoding.
    Weight is separated by n_primitives.
    n_primitive wise calculation is done by group wise convolution.

    Arguments:
      act (str): One of 'leaky', 'relu', 'none'
    """
        super(PrimitiveWiseLinear, self).__init__()
        self.n_primitives = n_primitives
        self.input_channels = input_channels
        self.output_channels = output_channels
        layers = []
        bn = nn.BatchNorm1d(self.n_primitives * self.input_channels)
        layers.append(bn)
        if act == 'leaky':
            layers.append(nn.LeakyReLU(inplace=True))
        elif act == 'relu':
            layers.append(nn.ReLU(inplace=True))
        else:
            raise NotImplementedError

        linear = nn.Conv1d(self.n_primitives * self.input_channels,
                           self.n_primitives * self.output_channels,
                           kernel_size=1,
                           groups=self.n_primitives,
                           bias=bias)
        layers.append(linear)

        self.main = nn.Sequential(*layers)

    def forward(self, input_data):
        return self.main(input_data)


class Linear(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 act='leaky',
                 bias=True):
        """
    Calculate n_primitive wise feature encoding.
    Weight is separated by n_primitives.
    n_primitive wise calculation is done by group wise convolution.

    Arguments:
      act (str): One of 'leaky', 'relu', 'none'
    """
        super(Linear, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        layers = []
        bn = nn.BatchNorm1d(self.input_channels)
        layers.append(bn)
        if act == 'leaky':
            layers.append(nn.LeakyReLU(inplace=True))
        elif act == 'relu':
            layers.append(nn.ReLU(inplace=True))
        else:
            raise NotImplementedError

        linear = nn.Conv1d(self.input_channels,
                           self.output_channels,
                           kernel_size=1,
                           bias=bias)
        layers.append(linear)

        self.main = nn.Sequential(*layers)

    def forward(self, input_data):
        return self.main(input_data)


class DecoderAtlasNetV2(nn.Module):
    def __init__(self,
                 dim=3,
                 z_dim=128,
                 c_dim=128,
                 hidden_size=256,
                 depth=5,
                 out_dim=3,
                 leaky=False,
                 legacy=False,
                 sequential=False):
        super().__init__()
        self.out_dim = out_dim
        self.c_dim = c_dim
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0
        self.sequential = sequential

        self.fc_c = nn.Linear(c_dim, self.c64)
        self.fc_p = nn.Conv1d(dim, self.c64, 1)

        if self.sequential:
            self.decoders = nn.ModuleList()
            for _ in range(self.out_dim):
                fin = Linear((c_dim + dim), self.c64, act='relu')
                enc_layers = []
                for idx in range(int(depth / 2)):
                    enc_layers.extend([
                        Linear(int(self.c64 / 2**idx),
                               int(self.c64 / 2**(idx + 1)),
                               act='relu')
                    ])
                dec_layers = []
                for idx in range(int(depth / 2), 0, -1):
                    dec_layers.extend([
                        Linear(int(self.c64 / 2**idx),
                               int(self.c64 / 2**(idx - 1)),
                               act='relu')
                    ])

                fout = Linear(self.c64, 1, act=('leaky' if leaky else 'relu'))
                self.decoders.append(
                    nn.Sequential(fin, *enc_layers, *dec_layers, fout))

        else:
            """
            fin = PrimitiveWiseLinear(self.out_dim, (c_dim + dim),
                                      self.c64,
                                      act='relu')
            """
            enc_layers = []
            for idx in range(int(depth / 2)):
                enc_layers.extend([
                    PrimitiveWiseLinear(self.out_dim,
                                        int(self.c64 / 2**idx),
                                        int(self.c64 / 2**(idx + 1)),
                                        act='relu')
                ])
            dec_layers = []
            for idx in range(int(depth / 2), 0, -1):
                dec_layers.extend([
                    PrimitiveWiseLinear(self.out_dim,
                                        int(self.c64 / 2**idx),
                                        int(self.c64 / 2**(idx - 1)),
                                        act='relu')
                ])

            fout = PrimitiveWiseLinear(self.out_dim,
                                       self.c64,
                                       1,
                                       act=('leaky' if leaky else 'relu'))
            self.decoder = nn.Sequential(*enc_layers, *dec_layers, fout)
            #self.decoder = nn.Sequential(fin, *enc_layers, *dec_layers, fout)

    def forward(self, p, _, c):
        B, P, dim = p.shape
        B2, ldim = c.shape
        assert B == B2

        # B, dim, P
        p = p.transpose(1, 2)

        # B, c64, P
        net = self.fc_p(p)

        # B, ldim, 1
        net_c = self.fc_c(c).unsqueeze(2)
        # B, c64, P
        net = net + net_c
        # B, n_pm * c64, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(B, -1, P)
        """
        # B, ldim, 1
        c = c.unsqueeze(-1).expand(-1, -1, P)

        net = torch.cat([p, c], dim=1)
        """

        if self.sequential:
            outs = []
            for decoder in self.decoders:
                outs.append(decoder(net))
            net = torch.cat(outs, dim=1).view(B, self.out_dim,
                                              P).transpose(1, 2).contiguous()
        else:
            """
            net = net.unsqueeze(1).expand(-1, self.out_dim, -1, -1).reshape(
                B, self.out_dim * (dim + ldim), P)
            """

            # B, P, n_pm
            net = self.decoder(net).view(B, self.out_dim,
                                         P).transpose(1, 2).contiguous()

        return net


class GroupWiseResnetBlockConv1d(nn.Module):
    ''' 1D-Convolutional ResNet block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''
    def __init__(self,
                 n_primitives,
                 size_in,
                 size_h=None,
                 size_out=None,
                 batch_norm_momentum=0.1):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in
        self.n_primitives = n_primitives

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.bn_0 = nn.BatchNorm1d(size_in * self.n_primitives,
                                   momentum=batch_norm_momentum)
        self.bn_1 = nn.BatchNorm1d(size_h * self.n_primitives,
                                   momentum=batch_norm_momentum)

        self.fc_0 = nn.Conv1d(size_in * self.n_primitives,
                              size_h * self.n_primitives,
                              kernel_size=1,
                              groups=self.n_primitives)

        self.fc_1 = nn.Conv1d(size_h * self.n_primitives,
                              size_out * self.n_primitives,
                              kernel_size=1,
                              groups=self.n_primitives)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in * self.n_primitives,
                                      size_out * self.n_primitives,
                                      kernel_size=1,
                                      bias=False,
                                      groups=self.n_primitives)

        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(self.bn_0(x)))
        dx = self.fc_1(self.actvn(self.bn_1(net)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class GroupWiseResnetBlockConv1dGBN(nn.Module):
    ''' 1D-Convolutional ResNet block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''
    def __init__(self, n_primitives, size_in, size_h=None, size_out=None):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in
        self.n_primitives = n_primitives

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.bn_0 = nn.GroupNorm(self.n_primitives,
                                 size_in * self.n_primitives)
        self.bn_1 = nn.GroupNorm(self.n_primitives, size_h * self.n_primitives)

        self.fc_0 = nn.Conv1d(size_in * self.n_primitives,
                              size_h * self.n_primitives,
                              kernel_size=1,
                              groups=self.n_primitives)

        self.fc_1 = nn.Conv1d(size_h * self.n_primitives,
                              size_out * self.n_primitives,
                              kernel_size=1,
                              groups=self.n_primitives)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in * self.n_primitives,
                                      size_out * self.n_primitives,
                                      kernel_size=1,
                                      bias=False,
                                      groups=self.n_primitives)

        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(self.bn_0(x)))
        dx = self.fc_1(self.actvn(self.bn_1(net)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class GroupWiseResnetBlockConv1dWoBN(nn.Module):
    ''' 1D-Convolutional ResNet block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''
    def __init__(self, n_primitives, size_in, size_h=None, size_out=None):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in
        self.n_primitives = n_primitives

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules

        self.fc_0 = nn.Conv1d(size_in * self.n_primitives,
                              size_h * self.n_primitives,
                              kernel_size=1,
                              groups=self.n_primitives)

        self.fc_1 = nn.Conv1d(size_h * self.n_primitives,
                              size_out * self.n_primitives,
                              kernel_size=1,
                              groups=self.n_primitives)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in * self.n_primitives,
                                      size_out * self.n_primitives,
                                      kernel_size=1,
                                      bias=False,
                                      groups=self.n_primitives)

        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class DecoderBatchNormAtlasNetV2(nn.Module):
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
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        # Submodules
        if self.c_dim != 0:
            self.fc_c = nn.Linear(c_dim, self.c64)
        self.fc_p = nn.Conv1d(dim, self.c64, 1)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])
        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*enc_layers, *dec_layers)

        self.bn = nn.BatchNorm1d(self.c64 * self.out_dim)

        self.fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                self.out_dim,
                                1,
                                groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.c_dim != 0:
            net_c = self.fc_c(c).unsqueeze(2)
            net = net + net_c
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)
        net = self.decoder(net)

        out = self.fc_out(self.actvn(self.bn(net)))

        # batch, points, occupancy=out_dim (primitive_num)
        out = out.reshape(batch_size, self.out_dim,
                          T).transpose(1, 2).contiguous()
        return out


class DecoderConstInputCBatchNormAtlasNetV2(nn.Module):
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
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        self.const = ConstantInput(self.c64 * self.out_dim)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))
        self.fc_p = nn.Conv1d(dim, self.decoder_input_dim, 1)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseCResnetBlockConv1d(self.c_dim,
                                            self.out_dim,
                                            int(self.c64 / 2**idx),
                                            size_out=int(self.c64 /
                                                         2**(idx - 1)))
            ])

        self.decoders = nn.ModuleList(dec_layers)

        #self.bn = nn.BatchNorm1d(self.c64 * self.out_dim)
        self.bn = CBatchNorm1d(self.c_dim, self.c64 * self.out_dim)

        self.fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                self.out_dim,
                                1,
                                groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c, **kwargs):

        # B, c_dim, 1
        const = self.const(p).unsqueeze(2)
        # B, pm_num * decoder_input_dim, 1
        pm_wise_latent = self.encoder(const)

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        # B, decoder_input_dim, P
        net = self.fc_p(p)
        # B, pm_num * decoder_input_dim, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)

        # B, pm_num * decoder_input_dim, P
        net = net + pm_wise_latent
        for decoder in self.decoders:
            net = decoder(net, c)

        out = self.fc_out(self.actvn(self.bn(net, c)))

        # batch, points, occupancy=out_dim (primitive_num)
        out = out.reshape(batch_size, self.out_dim,
                          T).transpose(1, 2).contiguous()
        return out


class ConstantInput(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.expand(batch, -1)

        return out


class GroupWiseCResnetBlockConv1d(nn.Module):
    ''' Conditional batch normalization-based Resnet block class.

    Args:
        c_dim (int): dimension of latend conditioned code c
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
        norm_method (str): normalization method
        legacy (bool): whether to use legacy blocks 
    '''
    def __init__(self,
                 c_dim,
                 n_primitives,
                 size_in,
                 size_h=None,
                 size_out=None,
                 norm_method='batch_norm',
                 batch_norm_momentum=0.1,
                 legacy=False):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        self.n_primitives = n_primitives

        # Submodules
        if not legacy:
            self.bn_0 = CBatchNorm1d(c_dim,
                                     size_in * self.n_primitives,
                                     norm_method=norm_method.replace(
                                         'layer_norm', 'group_norm'),
                                     batch_norm_momentum=batch_norm_momentum,
                                     groups=self.n_primitives)
            self.bn_1 = CBatchNorm1d(c_dim,
                                     size_h * self.n_primitives,
                                     norm_method=norm_method.replace(
                                         'layer_norm', 'group_norm'),
                                     batch_norm_momentum=batch_norm_momentum,
                                     groups=self.n_primitives)
        else:
            raise NotImplementedError

        self.fc_0 = nn.Conv1d(size_in * self.n_primitives,
                              size_h * self.n_primitives,
                              kernel_size=1,
                              groups=self.n_primitives)
        self.fc_1 = nn.Conv1d(size_h * self.n_primitives,
                              size_out * self.n_primitives,
                              kernel_size=1,
                              groups=self.n_primitives)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in * self.n_primitives,
                                      size_out * self.n_primitives,
                                      kernel_size=1,
                                      bias=False,
                                      groups=self.n_primitives)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c):
        net = self.fc_0(self.actvn(self.bn_0(x, c)))
        dx = self.fc_1(self.actvn(self.bn_1(net, c)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class DecoderBatchNormWithPointWiseMotionDecodingAtlasNetV2(nn.Module):
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
                 param_out_dim=18,
                 leaky=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.depth = depth

        self.out_dim = out_dim
        self.param_out_dim = param_out_dim
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        # Submodules
        if self.c_dim != 0:
            self.fc_c = nn.Linear(c_dim, self.c64)
        self.fc_p = nn.Conv1d(dim, self.c64, 1)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])
        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*enc_layers, *dec_layers)

        self.bn = nn.BatchNorm1d(self.c64 * self.out_dim)

        self.fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                self.out_dim * (1 + self.param_out_dim),
                                1,
                                groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.c_dim != 0:
            net_c = self.fc_c(c).unsqueeze(2)
            net = net + net_c
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)
        net = self.decoder(net)

        out = self.fc_out(self.actvn(self.bn(net)))

        # batch, points, occupancy=out_dim (primitive_num)
        out = out.reshape(batch_size, self.out_dim * (1 + self.param_out_dim),
                          T).transpose(1, 2).contiguous().reshape(
                              batch_size, T, self.out_dim,
                              1 + self.param_out_dim)

        # B, P, pm_num
        occ = out[..., 0]
        # B, P, pm_num, param_out_dim
        param = out[..., 1:]

        return dict(occupancy=occ, param=param)


class DecoderBatchNormWithPointWiseMotionDecodingAtlasNetV2SeparateAdditionalHead(
        nn.Module):
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
                 param_out_dim=18,
                 leaky=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.depth = depth

        self.out_dim = out_dim
        self.param_out_dim = param_out_dim
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        # Submodules
        if self.c_dim != 0:
            self.fc_c = nn.Linear(c_dim, self.c64)
        self.fc_p = nn.Conv1d(dim, self.c64, 1)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])
        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*enc_layers, *dec_layers)

        self.bn = nn.BatchNorm1d(self.c64 * self.out_dim)

        self.fc_occ_out = nn.Conv1d(self.c64 * self.out_dim,
                                    self.out_dim,
                                    1,
                                    groups=self.out_dim)
        self.fc_param_out = nn.Conv1d(self.c64 * self.out_dim,
                                      self.out_dim * self.param_out_dim,
                                      1,
                                      groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.c_dim != 0:
            net_c = self.fc_c(c).unsqueeze(2)
            net = net + net_c
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)
        net = self.decoder(net)

        occ_out = self.fc_occ_out(self.actvn(self.bn(net)))

        occ_out = occ_out.reshape(batch_size, self.out_dim,
                                  T).transpose(1, 2).contiguous()

        param_out = self.fc_param_out(self.actvn(self.bn(net)))

        param_out = param_out.reshape(batch_size,
                                      self.out_dim * self.param_out_dim,
                                      T).transpose(1, 2).contiguous().reshape(
                                          batch_size, T, self.out_dim,
                                          self.param_out_dim)

        return dict(occupancy=occ_out, param=param_out)


class DecoderBatchNormWithPointWiseMotionDecodingAtlasNetV2SShapeAdditionalHead(
        nn.Module):
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
                 param_out_dim=18,
                 leaky=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.depth = depth

        self.out_dim = out_dim
        self.param_out_dim = param_out_dim
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        # Submodules
        if self.c_dim != 0:
            self.fc_c = nn.Linear(c_dim, self.c64)
        self.fc_p = nn.Conv1d(dim, self.c64, 1)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])
        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*enc_layers, *dec_layers)

        self.bn = nn.BatchNorm1d(self.c64 * self.out_dim)

        self.fc_param_out = nn.Conv1d(self.c64 * self.out_dim,
                                      self.out_dim * self.param_out_dim,
                                      1,
                                      groups=self.out_dim)
        self.bn_occ1 = nn.BatchNorm1d(self.param_out_dim * self.out_dim)
        self.fc_occ_out1 = nn.Conv1d(self.param_out_dim * self.out_dim,
                                     self.out_dim * int(self.c64 / 2),
                                     1,
                                     groups=self.out_dim)
        self.bn_occ2 = nn.BatchNorm1d(int(self.c64 / 2) * self.out_dim)
        self.fc_occ_out2 = nn.Conv1d(self.out_dim * int(self.c64 / 2),
                                     self.out_dim,
                                     1,
                                     groups=self.out_dim)
        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.c_dim != 0:
            net_c = self.fc_c(c).unsqueeze(2)
            net = net + net_c
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)
        net = self.decoder(net)

        param_out_raw = self.fc_param_out(self.actvn(self.bn(net)))

        param_out = param_out_raw.reshape(
            batch_size, self.out_dim * self.param_out_dim,
            T).transpose(1,
                         2).contiguous().reshape(batch_size, T, self.out_dim,
                                                 self.param_out_dim)

        occ_out = self.fc_occ_out1(F.relu(self.bn_occ1(param_out_raw)))
        occ_out = self.fc_occ_out2(self.actvn(self.bn_occ2(occ_out)))

        occ_out = occ_out.reshape(batch_size, self.out_dim,
                                  T).transpose(1, 2).contiguous()

        return dict(occupancy=occ_out, param=param_out)


class DecoderSharedConstInputCBatchNormAtlasNetV2SeparateAdditionalHead(
        nn.Module):
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
                 param_out_dim=18,
                 leaky=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.depth = depth
        self.param_out_dim = param_out_dim

        self.out_dim = out_dim
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        self.const = ConstantInput(self.c64)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))
        self.fc_p = nn.Conv1d(dim, self.decoder_input_dim, 1)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseCResnetBlockConv1d(self.c_dim,
                                            self.out_dim,
                                            int(self.c64 / 2**idx),
                                            size_out=int(self.c64 /
                                                         2**(idx - 1)))
            ])

        self.decoders = nn.ModuleList(dec_layers)

        #self.bn = nn.BatchNorm1d(self.c64 * self.out_dim)
        self.bn = CBatchNorm1d(self.c_dim, self.c64 * self.out_dim)

        self.fc_occ_out = nn.Conv1d(self.c64 * self.out_dim,
                                    self.out_dim,
                                    1,
                                    groups=self.out_dim)
        self.fc_param_out = nn.Conv1d(self.c64 * self.out_dim,
                                      self.out_dim * self.param_out_dim,
                                      1,
                                      groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c, **kwargs):

        # B, c_dim, 1
        const = self.const(p).reshape(p.size(0), 1, self.c64, 1).expand(
            -1, self.out_dim, -1, -1).reshape(p.size(0),
                                              self.out_dim * self.c64, 1)
        # B, pm_num * decoder_input_dim, 1
        pm_wise_latent = self.encoder(const)

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        # B, decoder_input_dim, P
        net = self.fc_p(p)
        # B, pm_num * decoder_input_dim, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)

        # B, pm_num * decoder_input_dim, P
        net = net + pm_wise_latent
        for decoder in self.decoders:
            net = decoder(net, c)

        occ_out = self.fc_occ_out(self.actvn(self.bn(net, c)))

        occ_out = occ_out.reshape(batch_size, self.out_dim,
                                  T).transpose(1, 2).contiguous()

        param_out = self.fc_param_out(self.actvn(self.bn(net, c)))

        param_out = param_out.reshape(batch_size,
                                      self.out_dim * self.param_out_dim,
                                      T).transpose(1, 2).contiguous().reshape(
                                          batch_size, T, self.out_dim,
                                          self.param_out_dim)

        return dict(occupancy=occ_out, param=param_out)


class DecoderConstInputAllBatchNormAtlasNetV2(nn.Module):
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
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        self.const = ConstantInput(self.c64 * self.out_dim)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))
        self.fc_p = nn.Conv1d(dim, self.decoder_input_dim, 1)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*dec_layers)

        self.bn = nn.BatchNorm1d(self.c64 * self.out_dim)

        self.fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                self.out_dim,
                                1,
                                groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c, **kwargs):

        # B, c_dim, 1
        const = self.const(p).unsqueeze(2)
        # B, pm_num * decoder_input_dim, 1
        pm_wise_latent = self.encoder(const)

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        # B, decoder_input_dim, P
        net = self.fc_p(p)
        # B, pm_num * decoder_input_dim, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)

        # B, pm_num * decoder_input_dim, P
        net = net + pm_wise_latent
        net = self.decoder(net)

        out = self.fc_out(self.actvn(self.bn(net)))

        # batch, points, occupancy=out_dim (primitive_num)
        out = out.reshape(batch_size, self.out_dim,
                          T).transpose(1, 2).contiguous()
        return out


class DecoderConstInputAllBatchNormAtlasNetV2CBatchDecoderAdd(nn.Module):
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
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        self.const = ConstantInput(self.c64 * self.out_dim)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))
        self.fc_p = nn.Conv1d(dim, self.decoder_input_dim, 1)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*dec_layers)

        self.bn = nn.BatchNorm1d(self.c64 * self.out_dim)

        self.fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                self.out_dim,
                                1,
                                groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.diff_fc_p = nn.Conv1d(dim, self.c64, 1)
        self.diff_decoders = nn.ModuleList([
            GroupWiseCResnetBlockConv1d(self.c_dim, self.out_dim, self.c64)
            for _ in range(self.depth)
        ])

        self.diff_bn = CBatchNorm1d(self.c_dim, self.c64 * self.out_dim)

        self.diff_fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                     self.out_dim,
                                     1,
                                     groups=self.out_dim)

    def forward(self, p, z, c, **kwargs):

        # B, c_dim, 1
        const = self.const(p).unsqueeze(2)
        # B, pm_num * decoder_input_dim, 1
        pm_wise_latent = self.encoder(const)

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        # B, decoder_input_dim, P
        net = self.fc_p(p)
        # B, pm_num * decoder_input_dim, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)

        # B, pm_num * decoder_input_dim, P
        net = net + pm_wise_latent
        net = self.decoder(net)

        out = self.fc_out(self.actvn(self.bn(net)))

        # batch, points, occupancy=out_dim (primitive_num)
        out = out.reshape(batch_size, self.out_dim,
                          T).transpose(1, 2).contiguous()

        # B, c64, P
        diff_net = self.diff_fc_p(p)
        diff_net = diff_net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                                -1).reshape(
                                                    batch_size,
                                                    self.c64 * self.out_dim, T)
        for diff_decoder in self.diff_decoders:
            diff_net = diff_decoder(diff_net, c)
        diff_out = self.diff_fc_out(self.actvn(self.diff_bn(net, c)))
        diff_out = diff_out.reshape(batch_size, self.out_dim,
                                    T).transpose(1, 2).contiguous()
        occ = out + diff_out
        ret = dict(occupancy=occ, canonical_occupancy=out)
        return ret


class DecoderConstInputAllBatchNormAtlasNetV2CBatchAtlasNetV2Add(nn.Module):
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
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        self.const = ConstantInput(self.c64 * self.out_dim)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))
        self.fc_p = nn.Conv1d(dim, self.decoder_input_dim, 1)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*dec_layers)

        self.bn = nn.BatchNorm1d(self.c64 * self.out_dim)

        self.fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                self.out_dim,
                                1,
                                groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        diff_dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            diff_dec_layers.extend([
                GroupWiseCResnetBlockConv1d(self.c_dim,
                                            self.out_dim,
                                            int(self.c64 / 2**idx),
                                            size_out=int(self.c64 /
                                                         2**(idx - 1)))
            ])

        self.diff_decoders = nn.ModuleList(diff_dec_layers)

        self.diff_bn = CBatchNorm1d(self.c_dim, self.c64 * self.out_dim)

        self.diff_fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                     self.out_dim,
                                     1,
                                     groups=self.out_dim)

    def forward(self, p, z, c, **kwargs):

        # B, c_dim, 1
        const = self.const(p).unsqueeze(2)
        # B, pm_num * decoder_input_dim, 1
        pm_wise_latent = self.encoder(const)

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        # B, decoder_input_dim, P
        net = self.fc_p(p)
        # B, pm_num * decoder_input_dim, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)

        # B, pm_num * decoder_input_dim, P
        shared = net + pm_wise_latent
        net = self.decoder(shared)

        out = self.fc_out(self.actvn(self.bn(net)))

        # batch, points, occupancy=out_dim (primitive_num)
        out = out.reshape(batch_size, self.out_dim,
                          T).transpose(1, 2).contiguous()

        diff_net = shared
        for diff_decoder in self.diff_decoders:
            diff_net = diff_decoder(diff_net, c)

        diff_out = self.diff_fc_out(self.actvn(self.diff_bn(diff_net, c)))
        diff_out = diff_out.reshape(batch_size, self.out_dim,
                                    T).transpose(1, 2).contiguous()
        occ = out + diff_out
        ret = dict(occupancy=occ, canonical_occupancy=out)
        return ret


class DecoderConstInputAllBatchNormAtlasNetV2SeparateCBatchAtlasNetV2AddSeparateAdditionalHead(
        nn.Module):
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
                 param_out_dim=18,
                 leaky=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.depth = depth
        self.param_out_dim = param_out_dim

        self.out_dim = out_dim
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        self.const = ConstantInput(self.c64 * self.out_dim)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))
        self.fc_p = nn.Conv1d(dim, self.decoder_input_dim, 1)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*dec_layers)

        self.occ_bn = nn.BatchNorm1d(self.c64 * self.out_dim)
        self.fc_occ_out = nn.Conv1d(self.c64 * self.out_dim,
                                    self.out_dim,
                                    1,
                                    groups=self.out_dim)

        self.param_bn = nn.BatchNorm1d(self.c64 * self.out_dim)
        self.fc_param_out = nn.Conv1d(self.c64 * self.out_dim,
                                      self.out_dim * self.param_out_dim,
                                      1,
                                      groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.diff_fc_c = nn.Linear(c_dim, self.c64)
        self.diff_fc_p = nn.Conv1d(dim, self.c64, 1)
        diff_enc_layers = []
        for idx in range(int(depth / 2)):
            diff_enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])

        diff_dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            diff_dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])

        self.diff_decoder = nn.Sequential(*diff_enc_layers, *diff_dec_layers)

        self.diff_occ_bn = CBatchNorm1d(self.c_dim, self.c64 * self.out_dim)
        self.diff_fc_occ_out = nn.Conv1d(self.c64 * self.out_dim,
                                         self.out_dim,
                                         1,
                                         groups=self.out_dim)

        self.diff_param_bn = CBatchNorm1d(self.c_dim, self.c64 * self.out_dim)
        self.diff_fc_param_out = nn.Conv1d(self.c64 * self.out_dim,
                                           self.out_dim * self.param_out_dim,
                                           1,
                                           groups=self.out_dim)

    def forward(self, p, z, c, **kwargs):

        # B, c_dim, 1
        const = self.const(p).unsqueeze(2)
        # B, pm_num * decoder_input_dim, 1
        pm_wise_latent = self.encoder(const)

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        # B, decoder_input_dim, P
        net = self.fc_p(p)
        # B, pm_num * decoder_input_dim, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)

        # B, pm_num * decoder_input_dim, P
        shared = net + pm_wise_latent
        net = self.decoder(shared)

        occ_out = self.fc_occ_out(self.actvn(self.occ_bn(net)))
        # batch, points, occupancy=out_dim (primitive_num)
        occ_out = occ_out.reshape(batch_size, self.out_dim,
                                  T).transpose(1, 2).contiguous()

        param_out = self.fc_param_out(self.actvn(self.param_bn(net)))
        param_out = param_out.reshape(batch_size,
                                      self.out_dim * self.param_out_dim,
                                      T).transpose(1, 2).contiguous().reshape(
                                          batch_size, T, self.out_dim,
                                          self.param_out_dim)
        diff_net = self.diff_fc_p(p)
        diff_net_c = self.diff_fc_c(c).unsqueeze(2)
        diff_net = diff_net + diff_net_c
        diff_net = diff_net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                                -1).reshape(batch_size, -1, T)
        diff_net = self.diff_decoder(diff_net)

        diff_occ_out = self.diff_fc_occ_out(
            self.actvn(self.diff_occ_bn(diff_net, c)))
        diff_occ_out = diff_occ_out.reshape(batch_size, self.out_dim,
                                            T).transpose(1, 2).contiguous()
        diff_param_out = self.diff_fc_param_out(
            self.actvn(self.diff_param_bn(diff_net, c)))
        diff_param_out = diff_param_out.reshape(
            batch_size, self.out_dim * self.param_out_dim,
            T).transpose(1,
                         2).contiguous().reshape(batch_size, T, self.out_dim,
                                                 self.param_out_dim)
        occ = occ_out + diff_occ_out
        param = param_out + diff_param_out
        ret = dict(occupancy=occ, canonical_occupancy=occ_out, param=param)
        return ret


class DecoderConstInputAllBatchNormAtlasNetV2CBatchAtlasNetV2AddSeparateAdditionalHead(
        nn.Module):
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
                 param_out_dim=18,
                 leaky=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.depth = depth
        self.param_out_dim = param_out_dim

        self.out_dim = out_dim
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        self.const = ConstantInput(self.c64 * self.out_dim)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))
        self.fc_p = nn.Conv1d(dim, self.decoder_input_dim, 1)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*dec_layers)

        self.occ_bn = nn.BatchNorm1d(self.c64 * self.out_dim)
        self.fc_occ_out = nn.Conv1d(self.c64 * self.out_dim,
                                    self.out_dim,
                                    1,
                                    groups=self.out_dim)

        self.param_bn = nn.BatchNorm1d(self.c64 * self.out_dim)
        self.fc_param_out = nn.Conv1d(self.c64 * self.out_dim,
                                      self.out_dim * self.param_out_dim,
                                      1,
                                      groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        diff_dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            diff_dec_layers.extend([
                GroupWiseCResnetBlockConv1d(self.c_dim,
                                            self.out_dim,
                                            int(self.c64 / 2**idx),
                                            size_out=int(self.c64 /
                                                         2**(idx - 1)))
            ])

        self.diff_decoders = nn.ModuleList(diff_dec_layers)

        self.diff_occ_bn = CBatchNorm1d(self.c_dim, self.c64 * self.out_dim)
        self.diff_fc_occ_out = nn.Conv1d(self.c64 * self.out_dim,
                                         self.out_dim,
                                         1,
                                         groups=self.out_dim)

        self.diff_param_bn = CBatchNorm1d(self.c_dim, self.c64 * self.out_dim)
        self.diff_fc_param_out = nn.Conv1d(self.c64 * self.out_dim,
                                           self.out_dim * self.param_out_dim,
                                           1,
                                           groups=self.out_dim)

    def forward(self, p, z, c, **kwargs):

        # B, c_dim, 1
        const = self.const(p).unsqueeze(2)
        # B, pm_num * decoder_input_dim, 1
        pm_wise_latent = self.encoder(const)

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        # B, decoder_input_dim, P
        net = self.fc_p(p)
        # B, pm_num * decoder_input_dim, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)

        # B, pm_num * decoder_input_dim, P
        shared = net + pm_wise_latent
        net = self.decoder(shared)

        occ_out = self.fc_occ_out(self.actvn(self.occ_bn(net)))
        # batch, points, occupancy=out_dim (primitive_num)
        occ_out = occ_out.reshape(batch_size, self.out_dim,
                                  T).transpose(1, 2).contiguous()

        param_out = self.fc_param_out(self.actvn(self.param_bn(net)))
        param_out = param_out.reshape(batch_size,
                                      self.out_dim * self.param_out_dim,
                                      T).transpose(1, 2).contiguous().reshape(
                                          batch_size, T, self.out_dim,
                                          self.param_out_dim)
        diff_net = shared
        for diff_decoder in self.diff_decoders:
            diff_net = diff_decoder(diff_net, c)

        diff_occ_out = self.diff_fc_occ_out(
            self.actvn(self.diff_occ_bn(diff_net, c)))
        diff_occ_out = diff_occ_out.reshape(batch_size, self.out_dim,
                                            T).transpose(1, 2).contiguous()
        diff_param_out = self.diff_fc_param_out(
            self.actvn(self.diff_param_bn(diff_net, c)))
        diff_param_out = diff_param_out.reshape(
            batch_size, self.out_dim * self.param_out_dim,
            T).transpose(1,
                         2).contiguous().reshape(batch_size, T, self.out_dim,
                                                 self.param_out_dim)
        occ = occ_out + diff_occ_out
        param = param_out + diff_param_out
        ret = dict(occupancy=occ, canonical_occupancy=occ_out, param=param)
        return ret


class DecoderConstInputAllBatchNormAtlasNetV2CBatchAtlasNetV2AddCanonicalLocation(
        nn.Module):
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
        self.dim = dim

        self.out_dim = out_dim
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        self.const = ConstantInput(self.c64 * self.out_dim)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))
        self.fc_p = nn.Conv1d(dim, self.decoder_input_dim, 1)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*dec_layers)

        self.bn = nn.BatchNorm1d(self.c64 * self.out_dim)

        self.fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                self.out_dim,
                                1,
                                groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        diff_dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            diff_dec_layers.extend([
                GroupWiseCResnetBlockConv1d(self.c_dim,
                                            self.out_dim,
                                            int(self.c64 / 2**idx),
                                            size_out=int(self.c64 /
                                                         2**(idx - 1)))
            ])

        self.diff_decoders = nn.ModuleList(diff_dec_layers)

        self.diff_bn = CBatchNorm1d(self.c_dim, self.c64 * self.out_dim)

        self.diff_fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                     self.out_dim,
                                     1,
                                     groups=self.out_dim)

        canonical_loc_dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            canonical_loc_dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])
        canonical_loc_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_loc_enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=size_out)
            ])
            idx += 1

        self.canonical_loc_encoder = nn.Sequential(*canonical_loc_enc_layers)

        self.canonical_loc_bn = nn.BatchNorm1d(
            int(self.c64 / 2**idx) * self.out_dim)

        self.canonical_loc_fc_out = nn.Conv1d(int(self.c64 / 2**idx) *
                                              self.out_dim,
                                              self.out_dim * dim,
                                              1,
                                              groups=self.out_dim)

    def forward(self, p, z, c, **kwargs):

        # B, c_dim, 1
        const = self.const(p).unsqueeze(2)
        # B, pm_num * decoder_input_dim, 1
        pm_wise_latent = self.encoder(const)

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        # B, decoder_input_dim, P
        net = self.fc_p(p)
        # B, pm_num * decoder_input_dim, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)

        # B, pm_num * decoder_input_dim, P
        shared = net + pm_wise_latent
        net = self.decoder(shared)

        out = self.fc_out(self.actvn(self.bn(net)))

        # batch, points, occupancy=out_dim (primitive_num)
        out = out.reshape(batch_size, self.out_dim,
                          T).transpose(1, 2).contiguous()

        diff_net = shared
        for diff_decoder in self.diff_decoders:
            diff_net = diff_decoder(diff_net, c)

        diff_out = self.diff_fc_out(self.actvn(self.diff_bn(diff_net, c)))
        diff_out = diff_out.reshape(batch_size, self.out_dim,
                                    T).transpose(1, 2).contiguous()
        occ = out + diff_out

        canonical_loc = self.canonical_loc_encoder(const)
        canonical_loc = self.canonical_loc_fc_out(
            self.actvn(self.canonical_loc_bn(canonical_loc))).reshape(
                batch_size, self.out_dim, self.dim)

        ret = dict(occupancy=occ,
                   canonical_occupancy=out,
                   canonical_location=canonical_loc)
        return ret


class DecoderConstInputAllBatchNormAtlasNetV2CanonicalLocation(nn.Module):
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
        self.dim = dim

        self.out_dim = out_dim
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        self.const = ConstantInput(self.c64 * self.out_dim)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))
        self.fc_p = nn.Conv1d(dim, self.decoder_input_dim, 1)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*dec_layers)

        self.bn = nn.BatchNorm1d(self.c64 * self.out_dim)

        self.fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                self.out_dim,
                                1,
                                groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        canonical_loc_dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            canonical_loc_dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])
        canonical_loc_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_loc_enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=size_out)
            ])
            idx += 1

        self.canonical_loc_encoder = nn.Sequential(*canonical_loc_enc_layers)

        self.canonical_loc_bn = nn.BatchNorm1d(
            int(self.c64 / 2**idx) * self.out_dim)

        self.canonical_loc_fc_out = nn.Conv1d(int(self.c64 / 2**idx) *
                                              self.out_dim,
                                              self.out_dim * dim,
                                              1,
                                              groups=self.out_dim)

    def forward(self, p, z, c, **kwargs):

        # B, c_dim, 1
        const = self.const(p).unsqueeze(2)
        # B, pm_num * decoder_input_dim, 1
        pm_wise_latent = self.encoder(const)

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        # B, decoder_input_dim, P
        net = self.fc_p(p)
        # B, pm_num * decoder_input_dim, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)

        # B, pm_num * decoder_input_dim, P
        shared = net + pm_wise_latent
        net = self.decoder(shared)

        out = self.fc_out(self.actvn(self.bn(net)))

        # batch, points, occupancy=out_dim (primitive_num)
        out = out.reshape(batch_size, self.out_dim,
                          T).transpose(1, 2).contiguous()

        canonical_loc = self.canonical_loc_encoder(const)
        canonical_loc = self.canonical_loc_fc_out(
            self.actvn(self.canonical_loc_bn(canonical_loc))).reshape(
                batch_size, self.out_dim, self.dim)

        ret = dict(occupancy=out,
                   canonical_occupancy=out,
                   canonical_location=canonical_loc)
        return ret


class DecoderConstInputAllBatchNormAtlasNetV2InputDependentCanonicalLocation(
        nn.Module):
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
                 leaky=False,
                 splitted_latent_dim_for_motion=128):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.depth = depth
        self.dim = dim
        self.splitted_latent_dim_for_motion = splitted_latent_dim_for_motion
        assert c_dim > splitted_latent_dim_for_motion

        self.out_dim = out_dim
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        self.const = ConstantInput(self.c64 * self.out_dim)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))
        self.fc_p = nn.Conv1d(dim, self.decoder_input_dim, 1)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*dec_layers)

        self.bn = nn.BatchNorm1d(self.c64 * self.out_dim)

        self.fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                self.out_dim,
                                1,
                                groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        canonical_loc_dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            canonical_loc_dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])
        canonical_loc_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_loc_enc_layers.extend([
                GroupWiseCResnetBlockConv1d(
                    self.splitted_latent_dim_for_motion,
                    self.out_dim,
                    int(self.c64 / 2**idx),
                    size_out=size_out)
            ])
            idx += 1

        self.canonical_loc_encoders = nn.ModuleList(canonical_loc_enc_layers)

        self.canonical_loc_bn = CBatchNorm1d(
            self.splitted_latent_dim_for_motion,
            int(self.c64 / 2**idx) * self.out_dim)

        self.canonical_loc_fc_out = nn.Conv1d(int(self.c64 / 2**idx) *
                                              self.out_dim,
                                              self.out_dim * dim,
                                              1,
                                              groups=self.out_dim)

    def forward(self, p, z, c, **kwargs):

        # B, c_dim, 1
        const = self.const(p).unsqueeze(2)
        # B, pm_num * decoder_input_dim, 1
        pm_wise_latent = self.encoder(const)

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        # B, decoder_input_dim, P
        net = self.fc_p(p)
        # B, pm_num * decoder_input_dim, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)

        # B, pm_num * decoder_input_dim, P
        shared = net + pm_wise_latent
        net = self.decoder(shared)

        out = self.fc_out(self.actvn(self.bn(net)))

        # batch, points, occupancy=out_dim (primitive_num)
        out = out.reshape(batch_size, self.out_dim,
                          T).transpose(1, 2).contiguous()

        motion_z = c[:, :self.splitted_latent_dim_for_motion]
        canonical_loc = const
        for encoder in self.canonical_loc_encoders:
            canonical_loc = encoder(canonical_loc, motion_z)
        canonical_loc = self.canonical_loc_fc_out(
            self.actvn(self.canonical_loc_bn(canonical_loc,
                                             motion_z))).reshape(
                                                 batch_size, self.out_dim,
                                                 self.dim)

        ret = dict(occupancy=out,
                   canonical_occupancy=out,
                   canonical_location=canonical_loc)
        return ret


class DecoderConstInputAllBatchNormAtlasNetV2CanonicalLocationAndDirection(
        nn.Module):
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
                 leaky=False,
                 rot_output_dim=3):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.depth = depth
        self.dim = dim
        self.rot_output_dim = rot_output_dim

        self.out_dim = out_dim
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        self.const = ConstantInput(self.c64 * self.out_dim)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))
        self.fc_p = nn.Conv1d(dim, self.decoder_input_dim, 1)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*dec_layers)

        self.bn = nn.BatchNorm1d(self.c64 * self.out_dim)

        self.fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                self.out_dim,
                                1,
                                groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        canonical_loc_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_loc_enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=size_out)
            ])
            idx += 1

        self.canonical_loc_encoder = nn.Sequential(*canonical_loc_enc_layers)

        self.canonical_loc_bn = nn.BatchNorm1d(
            int(self.c64 / 2**idx) * self.out_dim)

        self.canonical_loc_fc_out = nn.Conv1d(int(self.c64 / 2**idx) *
                                              self.out_dim,
                                              self.out_dim * dim,
                                              1,
                                              groups=self.out_dim)

        canonical_direction_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_direction_enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=size_out)
            ])
            idx += 1

        self.canonical_direction_encoder = nn.Sequential(
            *canonical_direction_enc_layers)

        self.canonical_direction_bn = nn.BatchNorm1d(
            int(self.c64 / 2**idx) * self.out_dim)

        self.canonical_direction_fc_out = nn.Conv1d(
            int(self.c64 / 2**idx) * self.out_dim,
            self.out_dim * self.rot_output_dim,
            1,
            groups=self.out_dim)

    def forward(self, p, z, c, **kwargs):

        # B, c_dim, 1
        const = self.const(p).unsqueeze(2)
        # B, pm_num * decoder_input_dim, 1
        pm_wise_latent = self.encoder(const)

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        # B, decoder_input_dim, P
        net = self.fc_p(p)
        # B, pm_num * decoder_input_dim, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)

        # B, pm_num * decoder_input_dim, P
        shared = net + pm_wise_latent
        net = self.decoder(shared)

        out = self.fc_out(self.actvn(self.bn(net)))

        # batch, points, occupancy=out_dim (primitive_num)
        out = out.reshape(batch_size, self.out_dim,
                          T).transpose(1, 2).contiguous()

        canonical_loc = self.canonical_loc_encoder(const)
        canonical_loc = self.canonical_loc_fc_out(
            self.actvn(self.canonical_loc_bn(canonical_loc))).reshape(
                batch_size, self.out_dim, self.dim)

        canonical_direction = self.canonical_direction_encoder(const)
        canonical_direction = self.canonical_direction_fc_out(
            self.actvn(
                self.canonical_direction_bn(canonical_direction))).reshape(
                    batch_size, self.out_dim, self.rot_output_dim)

        ret = dict(occupancy=out,
                   canonical_occupancy=out,
                   canonical_location=canonical_loc,
                   canonical_direction=canonical_direction)
        return ret


class DecoderConstInputAllBatchNormAtlasNetV2CanonicalLocationAndDirectionDist(
        nn.Module):
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
                 leaky=False,
                 disable_probabilistic_direction_learning=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.depth = depth
        self.dim = dim
        self.disable_probabilistic_direction_learning = disable_probabilistic_direction_learning

        self.out_dim = out_dim
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        self.const = ConstantInput(self.c64 * self.out_dim)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))
        self.fc_p = nn.Conv1d(dim, self.decoder_input_dim, 1)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*dec_layers)

        self.bn = nn.BatchNorm1d(self.c64 * self.out_dim)

        self.fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                self.out_dim,
                                1,
                                groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        canonical_loc_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_loc_enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=size_out)
            ])
            idx += 1

        self.canonical_loc_encoder = nn.Sequential(*canonical_loc_enc_layers)

        self.canonical_loc_bn = nn.BatchNorm1d(
            int(self.c64 / 2**idx) * self.out_dim)

        self.canonical_loc_fc_out_mean = nn.Conv1d(int(self.c64 / 2**idx) *
                                                   self.out_dim,
                                                   self.out_dim * dim,
                                                   1,
                                                   groups=self.out_dim)

        self.canonical_loc_fc_out_std = nn.Conv1d(int(self.c64 / 2**idx) *
                                                  self.out_dim,
                                                  self.out_dim * dim,
                                                  1,
                                                  groups=self.out_dim)
        canonical_direction_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_direction_enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=size_out)
            ])
            idx += 1

        self.canonical_direction_encoder = nn.Sequential(
            *canonical_direction_enc_layers)

        self.canonical_direction_bn = nn.BatchNorm1d(
            int(self.c64 / 2**idx) * self.out_dim)

        self.canonical_direction_fc_out_mean = nn.Conv1d(
            int(self.c64 / 2**idx) * self.out_dim,
            self.out_dim * dim,
            1,
            groups=self.out_dim)
        self.canonical_direction_fc_out_std = nn.Conv1d(
            int(self.c64 / 2**idx) * self.out_dim,
            self.out_dim * 1,
            1,
            groups=self.out_dim)

    def forward(self, p, z, c, **kwargs):

        # B, c_dim, 1
        const = self.const(p).unsqueeze(2)
        # B, pm_num * decoder_input_dim, 1
        pm_wise_latent = self.encoder(const)

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        # B, decoder_input_dim, P
        net = self.fc_p(p)
        # B, pm_num * decoder_input_dim, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)

        # B, pm_num * decoder_input_dim, P
        shared = net + pm_wise_latent
        net = self.decoder(shared)

        out = self.fc_out(self.actvn(self.bn(net)))

        # batch, points, occupancy=out_dim (primitive_num)
        out = out.reshape(batch_size, self.out_dim,
                          T).transpose(1, 2).contiguous()

        canonical_loc = self.canonical_loc_encoder(const)
        canonical_loc_mean = self.canonical_loc_fc_out_mean(
            self.actvn(self.canonical_loc_bn(canonical_loc))).reshape(
                batch_size, self.out_dim, self.dim)
        canonical_loc_std = self.canonical_loc_fc_out_std(
            self.actvn(self.canonical_loc_bn(canonical_loc))).reshape(
                batch_size, self.out_dim, self.dim)
        canonical_loc_std = F.softplus(canonical_loc_std)

        canonical_loc = torch.distributions.Normal(
            canonical_loc_mean, canonical_loc_std).rsample()

        canonical_direction = self.canonical_direction_encoder(const)
        canonical_direction_mean = self.canonical_direction_fc_out_mean(
            self.actvn(
                self.canonical_direction_bn(canonical_direction))).reshape(
                    batch_size, self.out_dim, self.dim)
        canonical_direction_mean = F.normalize(canonical_direction_mean,
                                               dim=-1)
        if self.disable_probabilistic_direction_learning:
            canonical_direction = canonical_direction_mean
        else:
            canonical_direction_std = self.canonical_direction_fc_out_std(
                self.actvn(
                    self.canonical_direction_bn(canonical_direction))).reshape(
                        batch_size, self.out_dim)
            canonical_direction_std = F.softplus(canonical_direction_std)

            canonical_direction = power_spherical.PowerSpherical(
                canonical_direction_mean, canonical_direction_std).rsample()

        ret = dict(occupancy=out,
                   canonical_occupancy=out,
                   canonical_location=canonical_loc,
                   canonical_direction=canonical_direction)
        return ret


class DecoderConstInputAllBatchNormAtlasNetV2CanonicalLocationAndDirectionSkipConn(
        nn.Module):
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
        self.dim = dim

        self.out_dim = out_dim
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        self.const = ConstantInput(self.c64 * self.out_dim)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))
        self.fc_p = nn.Conv1d(dim, self.decoder_input_dim, 1)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*dec_layers)

        self.bn = nn.BatchNorm1d(self.c64 * self.out_dim)

        self.fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                self.out_dim,
                                1,
                                groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        canonical_loc_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_loc_enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=size_out)
            ])
            idx += 1

        self.canonical_loc_encoder = nn.Sequential(*canonical_loc_enc_layers)

        self.canonical_loc_bn = nn.BatchNorm1d(
            int(self.c64 / 2**idx) * self.out_dim)

        self.canonical_loc_fc_out = nn.Conv1d(int(self.c64 / 2**idx) *
                                              self.out_dim,
                                              self.out_dim * dim,
                                              1,
                                              groups=self.out_dim)

        canonical_direction_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_direction_enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=size_out)
            ])
            idx += 1

        self.canonical_direction_encoder = nn.Sequential(
            *canonical_direction_enc_layers)

        self.canonical_direction_bn = nn.BatchNorm1d(
            int(self.c64 / 2**idx) * self.out_dim)

        self.canonical_direction_fc_out = nn.Conv1d(int(self.c64 / 2**idx) *
                                                    self.out_dim,
                                                    self.out_dim * dim,
                                                    1,
                                                    groups=self.out_dim)
        """
        self.skip_connection_decoder = nn.Sequential([
            GroupWiseResnetBlockConv1d(self.out_dim,
                                       dim * 2,
                                       size_out=int(self.c64 // 2)),
            GroupWiseResnetBlockConv1d(self.out_dim,
                                       int(self.c64 // 2),
                                       size_out=self.c64)
        ])
        """
        self.skip_connection_decoder = GroupWiseResnetBlockConv1d(
            self.out_dim, dim * 2, size_out=self.c64)

    def forward(self, p, z, c, **kwargs):

        # B, c_dim, 1
        const = self.const(p).unsqueeze(2)
        # B, pm_num * decoder_input_dim, 1
        pm_wise_latent = self.encoder(const)

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        # B, decoder_input_dim, P
        net = self.fc_p(p)
        # B, pm_num * decoder_input_dim, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)

        # B, pm_num * decoder_input_dim, P
        shared = net + pm_wise_latent
        net = self.decoder(shared)

        canonical_loc = self.canonical_loc_encoder(const)
        canonical_loc = self.canonical_loc_fc_out(
            self.actvn(self.canonical_loc_bn(canonical_loc))).reshape(
                batch_size, self.out_dim, self.dim)

        canonical_direction = self.canonical_direction_encoder(const)
        canonical_direction = self.canonical_direction_fc_out(
            self.actvn(
                self.canonical_direction_bn(canonical_direction))).reshape(
                    batch_size, self.out_dim, self.dim)

        param_decoded = self.skip_connection_decoder(
            torch.cat([canonical_direction, canonical_loc],
                      dim=-1).reshape(batch_size, self.out_dim * self.dim * 2,
                                      1))
        out = self.fc_out(self.actvn(self.bn(net + param_decoded)))

        # batch, points, occupancy=out_dim (primitive_num)
        out = out.reshape(batch_size, self.out_dim,
                          T).transpose(1, 2).contiguous()

        ret = dict(occupancy=out,
                   canonical_occupancy=out,
                   canonical_location=canonical_loc,
                   canonical_direction=canonical_direction)
        return ret


class DecoderConstInputAllBatchNormAtlasNetV2CanonicalLocationAndDirectionDirect(
        nn.Module):
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
        self.dim = dim

        self.out_dim = out_dim
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        self.const = ConstantInput(self.c64 * self.out_dim)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))
        self.fc_p = nn.Conv1d(dim, self.decoder_input_dim, 1)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*dec_layers)

        self.bn = nn.BatchNorm1d(self.c64 * self.out_dim)

        self.fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                self.out_dim,
                                1,
                                groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.location = ConstantInput(dim * self.out_dim)
        self.direction = ConstantInput(dim * self.out_dim)

    def forward(self, p, z, c, **kwargs):

        # B, c_dim, 1
        const = self.const(p).unsqueeze(2)
        # B, pm_num * decoder_input_dim, 1
        pm_wise_latent = self.encoder(const)

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        # B, decoder_input_dim, P
        net = self.fc_p(p)
        # B, pm_num * decoder_input_dim, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)

        # B, pm_num * decoder_input_dim, P
        shared = net + pm_wise_latent
        net = self.decoder(shared)

        out = self.fc_out(self.actvn(self.bn(net)))

        # batch, points, occupancy=out_dim (primitive_num)
        out = out.reshape(batch_size, self.out_dim,
                          T).transpose(1, 2).contiguous()

        canonical_loc = self.location(p).reshape(batch_size, self.out_dim,
                                                 self.dim)
        canonical_direction = self.direction(p).reshape(
            batch_size, self.out_dim, self.dim)

        ret = dict(occupancy=out,
                   canonical_occupancy=out,
                   canonical_location=canonical_loc,
                   canonical_direction=canonical_direction)
        return ret


class DecoderConstInputAllBatchNormAtlasNetV2CanonicalLocationAndDirectionDistDirect(
        nn.Module):
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
                 leaky=False,
                 disable_probabilistic_direction_learning=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.depth = depth
        self.dim = dim
        self.disable_probabilistic_direction_learning = disable_probabilistic_direction_learning

        self.out_dim = out_dim
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        self.const = ConstantInput(self.c64 * self.out_dim)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))
        self.fc_p = nn.Conv1d(dim, self.decoder_input_dim, 1)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*dec_layers)

        self.bn = nn.BatchNorm1d(self.c64 * self.out_dim)

        self.fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                self.out_dim,
                                1,
                                groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.location_mean = ConstantInput(dim * self.out_dim)
        self.location_std = ConstantInput(dim * self.out_dim)
        self.direction_mean = ConstantInput(dim * self.out_dim)
        self.direction_std = ConstantInput(self.out_dim)

    def forward(self, p, z, c, **kwargs):

        # B, c_dim, 1
        const = self.const(p).unsqueeze(2)
        # B, pm_num * decoder_input_dim, 1
        pm_wise_latent = self.encoder(const)

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        # B, decoder_input_dim, P
        net = self.fc_p(p)
        # B, pm_num * decoder_input_dim, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)

        # B, pm_num * decoder_input_dim, P
        shared = net + pm_wise_latent
        net = self.decoder(shared)

        out = self.fc_out(self.actvn(self.bn(net)))

        # batch, points, occupancy=out_dim (primitive_num)
        out = out.reshape(batch_size, self.out_dim,
                          T).transpose(1, 2).contiguous()

        canonical_loc_mean = self.location_mean(p).reshape(
            batch_size, self.out_dim, self.dim)
        canonical_loc_std = self.location_std(p).reshape(
            batch_size, self.out_dim, self.dim)
        canonical_loc_std = F.softplus(canonical_loc_std)

        canonical_loc = torch.distributions.Normal(
            canonical_loc_mean, canonical_loc_std).rsample()

        canonical_direction_mean = self.direction_mean(p).reshape(
            batch_size, self.out_dim, self.dim)
        canonical_direction_mean = F.normalize(canonical_direction_mean,
                                               dim=-1)
        if self.disable_probabilistic_direction_learning:
            canonical_direction = canonical_direction_mean
        else:
            canonical_direction_std = self.direction_std(p).reshape(
                batch_size, self.out_dim)
            canonical_direction_std = F.softplus(canonical_direction_std)

            canonical_direction = power_spherical.PowerSpherical(
                canonical_direction_mean, canonical_direction_std).rsample()

        ret = dict(occupancy=out,
                   canonical_occupancy=out,
                   canonical_location=canonical_loc,
                   canonical_direction=canonical_direction)
        return ret


class DecoderConstInputAtlasNetV2CanonicalLocationAndDirection(nn.Module):
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
                 leaky=False,
                 rot_output_dim=3):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.depth = depth
        self.dim = dim
        self.rot_output_dim = rot_output_dim

        self.out_dim = out_dim
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        self.const = ConstantInput(self.c64 * self.out_dim)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1dWoBN(self.out_dim,
                                               int(self.c64 / 2**idx),
                                               size_out=int(self.c64 /
                                                            2**(idx + 1)))
            ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))
        self.fc_p = nn.Conv1d(dim, self.decoder_input_dim, 1)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1dWoBN(self.out_dim,
                                               int(self.c64 / 2**idx),
                                               size_out=int(self.c64 /
                                                            2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*dec_layers)

        self.fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                self.out_dim,
                                1,
                                groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        canonical_loc_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_loc_enc_layers.extend([
                GroupWiseResnetBlockConv1dWoBN(self.out_dim,
                                               int(self.c64 / 2**idx),
                                               size_out=size_out)
            ])
            idx += 1

        self.canonical_loc_encoder = nn.Sequential(*canonical_loc_enc_layers)

        self.canonical_loc_fc_out = nn.Conv1d(int(self.c64 / 2**idx) *
                                              self.out_dim,
                                              self.out_dim * dim,
                                              1,
                                              groups=self.out_dim)

        canonical_direction_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_direction_enc_layers.extend([
                GroupWiseResnetBlockConv1dWoBN(self.out_dim,
                                               int(self.c64 / 2**idx),
                                               size_out=size_out)
            ])
            idx += 1

        self.canonical_direction_encoder = nn.Sequential(
            *canonical_direction_enc_layers)

        self.canonical_direction_fc_out = nn.Conv1d(
            int(self.c64 / 2**idx) * self.out_dim,
            self.out_dim * self.rot_output_dim,
            1,
            groups=self.out_dim)

    def forward(self, p, z, c, **kwargs):

        # B, c_dim, 1
        const = self.const(p).unsqueeze(2)
        # B, pm_num * decoder_input_dim, 1
        pm_wise_latent = self.encoder(const)

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        # B, decoder_input_dim, P
        net = self.fc_p(p)
        # B, pm_num * decoder_input_dim, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)

        # B, pm_num * decoder_input_dim, P
        shared = net + pm_wise_latent
        net = self.decoder(shared)

        out = self.fc_out(self.actvn(net))

        # batch, points, occupancy=out_dim (primitive_num)
        out = out.reshape(batch_size, self.out_dim,
                          T).transpose(1, 2).contiguous()

        canonical_loc = self.canonical_loc_encoder(const)
        canonical_loc = self.canonical_loc_fc_out(
            self.actvn(canonical_loc)).reshape(batch_size, self.out_dim,
                                               self.dim)

        canonical_direction = self.canonical_direction_encoder(const)
        canonical_direction = self.canonical_direction_fc_out(
            self.actvn(canonical_direction)).reshape(batch_size, self.out_dim,
                                                     self.rot_output_dim)

        ret = dict(occupancy=out,
                   canonical_occupancy=out,
                   canonical_location=canonical_loc,
                   canonical_direction=canonical_direction)
        return ret


class DecoderConstInputAllGroupNormAtlasNetV2CanonicalLocationAndDirection(
        nn.Module):
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
                 leaky=False,
                 rot_output_dim=3):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.depth = depth
        self.dim = dim
        self.rot_output_dim = rot_output_dim

        self.out_dim = out_dim
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        self.const = ConstantInput(self.c64 * self.out_dim)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1dGBN(self.out_dim,
                                              int(self.c64 / 2**idx),
                                              size_out=int(self.c64 /
                                                           2**(idx + 1)))
            ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))
        self.fc_p = nn.Conv1d(dim, self.decoder_input_dim, 1)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1dGBN(self.out_dim,
                                              int(self.c64 / 2**idx),
                                              size_out=int(self.c64 /
                                                           2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*dec_layers)

        self.bn = nn.GroupNorm(self.out_dim, self.c64 * self.out_dim)

        self.fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                self.out_dim,
                                1,
                                groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        canonical_loc_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_loc_enc_layers.extend([
                GroupWiseResnetBlockConv1dGBN(self.out_dim,
                                              int(self.c64 / 2**idx),
                                              size_out=size_out)
            ])
            idx += 1

        self.canonical_loc_encoder = nn.Sequential(*canonical_loc_enc_layers)

        self.canonical_loc_bn = nn.GroupNorm(
            self.out_dim,
            int(self.c64 / 2**idx) * self.out_dim)

        self.canonical_loc_fc_out = nn.Conv1d(int(self.c64 / 2**idx) *
                                              self.out_dim,
                                              self.out_dim * dim,
                                              1,
                                              groups=self.out_dim)

        canonical_direction_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_direction_enc_layers.extend([
                GroupWiseResnetBlockConv1dGBN(self.out_dim,
                                              int(self.c64 / 2**idx),
                                              size_out=size_out)
            ])
            idx += 1

        self.canonical_direction_encoder = nn.Sequential(
            *canonical_direction_enc_layers)

        self.canonical_direction_bn = nn.GroupNorm(
            self.out_dim,
            int(self.c64 / 2**idx) * self.out_dim)

        self.canonical_direction_fc_out = nn.Conv1d(
            int(self.c64 / 2**idx) * self.out_dim,
            self.out_dim * self.rot_output_dim,
            1,
            groups=self.out_dim)

    def forward(self, p, z, c, **kwargs):

        # B, c_dim, 1
        const = self.const(p).unsqueeze(2)
        # B, pm_num * decoder_input_dim, 1
        pm_wise_latent = self.encoder(const)

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        # B, decoder_input_dim, P
        net = self.fc_p(p)
        # B, pm_num * decoder_input_dim, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)

        # B, pm_num * decoder_input_dim, P
        shared = net + pm_wise_latent
        net = self.decoder(shared)

        out = self.fc_out(self.actvn(self.bn(net)))

        # batch, points, occupancy=out_dim (primitive_num)
        out = out.reshape(batch_size, self.out_dim,
                          T).transpose(1, 2).contiguous()

        canonical_loc = self.canonical_loc_encoder(const)
        canonical_loc = self.canonical_loc_fc_out(
            self.actvn(self.canonical_loc_bn(canonical_loc))).reshape(
                batch_size, self.out_dim, self.dim)

        canonical_direction = self.canonical_direction_encoder(const)
        canonical_direction = self.canonical_direction_fc_out(
            self.actvn(
                self.canonical_direction_bn(canonical_direction))).reshape(
                    batch_size, self.out_dim, self.rot_output_dim)

        ret = dict(occupancy=out,
                   canonical_occupancy=out,
                   canonical_location=canonical_loc,
                   canonical_direction=canonical_direction)
        return ret


class Quantize(nn.Module):
    def __init__(self,
                 dim,
                 n_embed,
                 disable_learnable_quantized_latent_vectors=False,
                 is_simple_constant_mode=False):
        super().__init__()
        self.embed = nn.Parameter(torch.rand(dim, n_embed))
        self.dim = dim
        self.n_embed = n_embed
        self.disable_learnable_quantized_latent_vectors = disable_learnable_quantized_latent_vectors
        self.is_simple_constant_mode = is_simple_constant_mode
        assert not (self.is_simple_constant_mode
                    and self.disable_learnable_quantized_latent_vectors)

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (flatten.pow(2).sum(1, keepdim=True) -
                2 * flatten @ self.embed +
                self.embed.pow(2).sum(0, keepdim=True))
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)
        if self.is_simple_constant_mode:
            diff = torch.zeros_like(quantize).mean()
        else:
            if self.disable_learnable_quantized_latent_vectors:
                diff = (quantize.detach() - input).pow(2).mean()
            else:
                diff = (quantize - input).pow(2).mean()
            quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class DecoderMotionQuantizedInputAllBatchNormAtlasNetV2CanonicalLocationAndDirection(
        nn.Module):
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
                 leaky=False,
                 rot_output_dim=3,
                 motion_quantize_num=3):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.depth = depth
        self.dim = dim
        self.rot_output_dim = rot_output_dim

        self.out_dim = out_dim
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        self.const = ConstantInput(self.c64 * self.out_dim)
        self.motion_const = Quantize(self.c64 * self.out_dim,
                                     motion_quantize_num)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))
        self.fc_p = nn.Conv1d(dim, self.decoder_input_dim, 1)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*dec_layers)

        self.bn = nn.BatchNorm1d(self.c64 * self.out_dim)

        self.fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                self.out_dim,
                                1,
                                groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.fc_c = nn.Conv1d(self.c_dim, self.c64 * self.out_dim, 1)
        self.bn_c = nn.BatchNorm1d(self.c_dim)

        canonical_loc_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_loc_enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=size_out)
            ])
            idx += 1

        self.canonical_loc_encoder = nn.Sequential(*canonical_loc_enc_layers)

        self.canonical_loc_bn = nn.BatchNorm1d(
            int(self.c64 / 2**idx) * self.out_dim)

        self.canonical_loc_fc_out = nn.Conv1d(int(self.c64 / 2**idx) *
                                              self.out_dim,
                                              self.out_dim * dim,
                                              1,
                                              groups=self.out_dim)

        canonical_direction_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_direction_enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=size_out)
            ])
            idx += 1

        self.canonical_direction_encoder = nn.Sequential(
            *canonical_direction_enc_layers)

        self.canonical_direction_bn = nn.BatchNorm1d(
            int(self.c64 / 2**idx) * self.out_dim)

        self.canonical_direction_fc_out = nn.Conv1d(
            int(self.c64 / 2**idx) * self.out_dim,
            self.out_dim * self.rot_output_dim,
            1,
            groups=self.out_dim)

    def forward(self, p, z, c, **kwargs):

        # B, c_dim, 1
        const = self.const(p).unsqueeze(2)
        # B, pm_num * decoder_input_dim, 1
        pm_wise_latent = self.encoder(const)

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        # B, decoder_input_dim, P
        net = self.fc_p(p)
        # B, pm_num * decoder_input_dim, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)

        # B, pm_num * decoder_input_dim, P
        shared = net + pm_wise_latent
        net = self.decoder(shared)

        out = self.fc_out(self.actvn(self.bn(net)))

        # batch, points, occupancy=out_dim (primitive_num)
        out = out.reshape(batch_size, self.out_dim,
                          T).transpose(1, 2).contiguous()

        c = self.fc_c(self.actvn(self.bn_c(c.unsqueeze(2)))).squeeze(-1)
        motion_const, quantize_diff, _ = self.motion_const(c)
        canonical_loc = self.canonical_loc_encoder(motion_const.unsqueeze(2))
        canonical_loc = self.canonical_loc_fc_out(
            self.actvn(self.canonical_loc_bn(canonical_loc))).reshape(
                batch_size, self.out_dim, self.dim)

        canonical_direction = self.canonical_direction_encoder(
            motion_const.unsqueeze(2))
        canonical_direction = self.canonical_direction_fc_out(
            self.actvn(
                self.canonical_direction_bn(canonical_direction))).reshape(
                    batch_size, self.out_dim, self.rot_output_dim)

        ret = dict(occupancy=out,
                   canonical_occupancy=out,
                   canonical_location=canonical_loc,
                   canonical_direction=canonical_direction,
                   latent_quantize_diff=quantize_diff)
        return ret


class DecoderFullQuantizedInputAllBatchNormAtlasNetV2CanonicalLocationAndDirection(
        nn.Module):
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
                 leaky=False,
                 rot_output_dim=3,
                 quantize_num=4):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.depth = depth
        self.dim = dim
        self.rot_output_dim = rot_output_dim

        self.out_dim = out_dim
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        self.motion_const = Quantize(self.c64 * self.out_dim, quantize_num)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))
        self.fc_p = nn.Conv1d(dim, self.decoder_input_dim, 1)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*dec_layers)

        self.bn = nn.BatchNorm1d(self.c64 * self.out_dim)

        self.fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                self.out_dim,
                                1,
                                groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.fc_c = nn.Conv1d(self.c_dim, self.c64 * self.out_dim, 1)
        self.bn_c = nn.BatchNorm1d(self.c_dim)

        canonical_loc_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_loc_enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=size_out)
            ])
            idx += 1

        self.canonical_loc_encoder = nn.Sequential(*canonical_loc_enc_layers)

        self.canonical_loc_bn = nn.BatchNorm1d(
            int(self.c64 / 2**idx) * self.out_dim)

        self.canonical_loc_fc_out = nn.Conv1d(int(self.c64 / 2**idx) *
                                              self.out_dim,
                                              self.out_dim * dim,
                                              1,
                                              groups=self.out_dim)

        canonical_direction_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_direction_enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=size_out)
            ])
            idx += 1

        self.canonical_direction_encoder = nn.Sequential(
            *canonical_direction_enc_layers)

        self.canonical_direction_bn = nn.BatchNorm1d(
            int(self.c64 / 2**idx) * self.out_dim)

        self.canonical_direction_fc_out = nn.Conv1d(
            int(self.c64 / 2**idx) * self.out_dim,
            self.out_dim * self.rot_output_dim,
            1,
            groups=self.out_dim)

    def forward(self, p, z, c, **kwargs):

        # B, c_dim, 1
        c = self.fc_c(self.actvn(self.bn_c(c.unsqueeze(2)))).squeeze(-1)
        const, quantize_diff, _ = self.motion_const(c)
        const = const.unsqueeze(2)
        # B, pm_num * decoder_input_dim, 1
        pm_wise_latent = self.encoder(const)

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        # B, decoder_input_dim, P
        net = self.fc_p(p)
        # B, pm_num * decoder_input_dim, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)

        # B, pm_num * decoder_input_dim, P
        shared = net + pm_wise_latent
        net = self.decoder(shared)

        out = self.fc_out(self.actvn(self.bn(net)))

        # batch, points, occupancy=out_dim (primitive_num)
        out = out.reshape(batch_size, self.out_dim,
                          T).transpose(1, 2).contiguous()

        canonical_loc = self.canonical_loc_encoder(const)
        canonical_loc = self.canonical_loc_fc_out(
            self.actvn(self.canonical_loc_bn(canonical_loc))).reshape(
                batch_size, self.out_dim, self.dim)

        canonical_direction = self.canonical_direction_encoder(const)
        canonical_direction = self.canonical_direction_fc_out(
            self.actvn(
                self.canonical_direction_bn(canonical_direction))).reshape(
                    batch_size, self.out_dim, self.rot_output_dim)

        ret = dict(occupancy=out,
                   canonical_occupancy=out,
                   canonical_location=canonical_loc,
                   canonical_direction=canonical_direction,
                   latent_quantize_diff=quantize_diff)
        return ret


class DecoderQuantizedInputAllBatchNormAtlasNetV2CBatchAtlasNetV2Multiply(
        nn.Module):
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
                 leaky=False,
                 rot_output_dim=3,
                 motion_quantize_num=4,
                 shape_quantize_num=4,
                 motion_template_latent_dim=64,
                 shape_template_latent_dim=64,
                 init_canonical_location_out_conv_with_zero=False,
                 init_canonical_direction_out_conv_with_zero=False,
                 sigmoid_to_template=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.depth = depth
        self.rot_output_dim = rot_output_dim
        self.sigmoid_to_template = sigmoid_to_template

        self.dim = dim
        self.out_dim = out_dim
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        self.const = ConstantInput(self.c64 * self.out_dim)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))
        self.fc_p = nn.Conv1d(dim, self.decoder_input_dim, 1)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*dec_layers)

        self.bn = nn.BatchNorm1d(self.c64 * self.out_dim)

        self.fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                self.out_dim,
                                1,
                                groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        diff_dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            diff_dec_layers.extend([
                GroupWiseCResnetBlockConv1d(self.c_dim,
                                            self.out_dim,
                                            int(self.c64 / 2**idx),
                                            size_out=int(self.c64 /
                                                         2**(idx - 1)))
            ])

        self.diff_decoders = nn.ModuleList(diff_dec_layers)

        self.diff_bn = CBatchNorm1d(self.c_dim, self.c64 * self.out_dim)

        self.diff_fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                     self.out_dim,
                                     1,
                                     groups=self.out_dim)

        self.motion_const_quantizer = Quantize(self.c64 * self.out_dim,
                                               motion_quantize_num)
        self.fc_motion_c = nn.Conv1d(motion_template_latent_dim,
                                     self.c64 * self.out_dim, 1)
        self.bn_motion_c = nn.BatchNorm1d(motion_template_latent_dim)

        self.shape_const_quantizer = Quantize(self.c64 * self.out_dim,
                                              shape_quantize_num)
        self.fc_shape_c = nn.Conv1d(shape_template_latent_dim,
                                    self.c64 * self.out_dim, 1)
        self.bn_shape_c = nn.BatchNorm1d(shape_template_latent_dim)

        canonical_loc_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_loc_enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=size_out)
            ])
            idx += 1

        self.canonical_loc_encoder = nn.Sequential(*canonical_loc_enc_layers)

        self.canonical_loc_bn = nn.BatchNorm1d(
            int(self.c64 / 2**idx) * self.out_dim)

        self.canonical_loc_fc_out = nn.Conv1d(int(self.c64 / 2**idx) *
                                              self.out_dim,
                                              self.out_dim * dim,
                                              1,
                                              groups=self.out_dim)

        if init_canonical_location_out_conv_with_zero:
            self.canonical_loc_fc_out.weight.data.fill_(0.0)
            self.canonical_loc_fc_out.bias.data.fill_(0.0)
        canonical_direction_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_direction_enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=size_out)
            ])
            idx += 1

        self.canonical_direction_encoder = nn.Sequential(
            *canonical_direction_enc_layers)

        self.canonical_direction_bn = nn.BatchNorm1d(
            int(self.c64 / 2**idx) * self.out_dim)

        self.canonical_direction_fc_out = nn.Conv1d(
            int(self.c64 / 2**idx) * self.out_dim,
            self.out_dim * self.rot_output_dim,
            1,
            groups=self.out_dim)

        if init_canonical_direction_out_conv_with_zero:
            self.canonical_direction_fc_out.weight.data.fill_(0.0)
            self.canonical_direction_fc_out.bias.data.fill_(0.0)

    def forward(self,
                p,
                z,
                c,
                shape_template_latent=None,
                motion_template_latent=None):
        assert shape_template_latent is not None
        assert motion_template_latent is not None

        # B, c_dim, 1
        #const = self.const(p).unsqueeze(2)

        shape_c = self.fc_shape_c(
            self.actvn(self.bn_shape_c(
                shape_template_latent.unsqueeze(2)))).squeeze(-1)
        shape_const, shape_quantize_diff, _ = self.shape_const_quantizer(
            shape_c)

        # B, pm_num * decoder_input_dim, 1
        shape_const = shape_const.unsqueeze(2)
        pm_wise_latent = self.encoder(shape_const)

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        # B, decoder_input_dim, P
        net = self.fc_p(p)
        # B, pm_num * decoder_input_dim, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)

        # B, pm_num * decoder_input_dim, P
        shared = net + pm_wise_latent
        net = self.decoder(shared)

        out = self.fc_out(self.actvn(self.bn(net)))

        # batch, points, occupancy=out_dim (primitive_num)
        out = out.reshape(batch_size, self.out_dim,
                          T).transpose(1, 2).contiguous()

        diff_net = shared
        for diff_decoder in self.diff_decoders:
            diff_net = diff_decoder(diff_net, c)

        diff_out = self.diff_fc_out(self.actvn(self.diff_bn(diff_net, c)))
        diff_out = diff_out.reshape(batch_size, self.out_dim,
                                    T).transpose(1, 2).contiguous()
        if self.sigmoid_to_template:
            occ = torch.sigmoid(out) * diff_out
        else:
            occ = out * torch.sigmoid(diff_out)

        motion_c = self.fc_motion_c(
            self.actvn(self.bn_motion_c(
                motion_template_latent.unsqueeze(2)))).squeeze(-1)
        motion_const, motion_quantize_diff, _ = self.motion_const_quantizer(
            motion_c)

        motion_const = motion_const.unsqueeze(2)
        canonical_loc = self.canonical_loc_encoder(motion_const)
        canonical_loc = self.canonical_loc_fc_out(
            self.actvn(self.canonical_loc_bn(canonical_loc))).reshape(
                batch_size, self.out_dim, self.dim)

        canonical_direction = self.canonical_direction_encoder(motion_const)
        canonical_direction = self.canonical_direction_fc_out(
            self.actvn(
                self.canonical_direction_bn(canonical_direction))).reshape(
                    batch_size, self.out_dim, self.rot_output_dim)

        ret = dict(occupancy=occ,
                   canonical_occupancy=out,
                   canonical_location=canonical_loc,
                   canonical_direction=canonical_direction,
                   latent_quantize_diff=(shape_quantize_diff +
                                         motion_quantize_diff))
        return ret


class AttentionConstMemory(nn.Module):
    def __init__(self, input_dim, output_dim, memory_latent_dim, depth):
        super().__init__()
        self.output_dim = output_dim
        self.memory_latent_dim = memory_latent_dim
        self.depth = depth
        self.input_dim = input_dim

        self.const_memory = ConstantInput(self.depth * self.memory_latent_dim)
        self.fc_query = nn.Conv1d(self.input_dim,
                                  self.depth * self.output_dim,
                                  1,
                                  bias=False)
        self.fc_key = nn.Conv1d(self.depth, self.depth, 1, bias=False)
        self.fc_value = nn.Conv1d(self.depth, self.depth, 1, bias=False)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        query = self.fc_query(inputs.unsqueeze(-1)).view(
            batch_size, self.depth, self.output_dim)  # b, depth, query
        memory = self.const_memory(inputs).view(batch_size, self.depth,
                                                self.memory_latent_dim)
        key = self.fc_key(memory)  # b, depth, mem
        value = self.fc_key(memory)  # b, depth, mem
        attention = torch.matmul(query.transpose(1, 2) / self.depth**0.5,
                                 key)  # b, query, mem
        attention_normalized = F.softmax(attention, dim=-1)  # b, query, mem
        output = torch.matmul(attention_normalized,
                              value.transpose(1, 2))  # b, query ,depth

        return output


class DecoderAttentionInputAllBatchNormAtlasNetV2CBatchAtlasNetV2Multiply(
        nn.Module):
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
                 leaky=False,
                 rot_output_dim=3,
                 motion_quantize_num=4,
                 shape_quantize_num=4,
                 motion_template_latent_dim=64,
                 shape_template_latent_dim=64,
                 sigmoid_to_template=False,
                 attention_type='pm_wise_attention',
                 init_canonical_direction_out_conv_with_zero=False,
                 init_canonical_location_out_conv_with_zero=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.depth = depth
        self.rot_output_dim = rot_output_dim
        self.sigmoid_to_template = sigmoid_to_template
        self.shape_quantize_num = shape_quantize_num
        self.motion_quantize_num = motion_quantize_num
        self.motion_template_latent_dim = motion_template_latent_dim
        self.shape_template_latent_dim = shape_template_latent_dim

        self.dim = dim
        self.out_dim = out_dim
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))
        self.fc_p = nn.Conv1d(dim, self.decoder_input_dim, 1)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*dec_layers)

        self.bn = nn.BatchNorm1d(self.c64 * self.out_dim)

        self.fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                self.out_dim,
                                1,
                                groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        diff_dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            diff_dec_layers.extend([
                GroupWiseCResnetBlockConv1d(self.c_dim,
                                            self.out_dim,
                                            int(self.c64 / 2**idx),
                                            size_out=int(self.c64 /
                                                         2**(idx - 1)))
            ])

        self.diff_decoders = nn.ModuleList(diff_dec_layers)

        self.diff_bn = CBatchNorm1d(self.c_dim, self.c64 * self.out_dim)

        self.diff_fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                     self.out_dim,
                                     1,
                                     groups=self.out_dim)

        self.motion_attention_module = AttentionConstMemory(
            self.motion_template_latent_dim, self.out_dim,
            self.motion_quantize_num, self.c64)
        self.shape_attention_module = AttentionConstMemory(
            self.shape_template_latent_dim, self.out_dim,
            self.shape_quantize_num, self.c64)
        canonical_loc_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_loc_enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=size_out)
            ])
            idx += 1

        self.canonical_loc_encoder = nn.Sequential(*canonical_loc_enc_layers)

        self.canonical_loc_bn = nn.BatchNorm1d(
            int(self.c64 / 2**idx) * self.out_dim)

        self.canonical_loc_fc_out = nn.Conv1d(int(self.c64 / 2**idx) *
                                              self.out_dim,
                                              self.out_dim * dim,
                                              1,
                                              groups=self.out_dim)

        if init_canonical_location_out_conv_with_zero:
            self.canonical_loc_fc_out.weight.data.fill_(0.0)
            self.canonical_loc_fc_out.bias.data.fill_(0.0)

        canonical_direction_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_direction_enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=size_out)
            ])
            idx += 1

        self.canonical_direction_encoder = nn.Sequential(
            *canonical_direction_enc_layers)

        self.canonical_direction_bn = nn.BatchNorm1d(
            int(self.c64 / 2**idx) * self.out_dim)

        self.canonical_direction_fc_out = nn.Conv1d(
            int(self.c64 / 2**idx) * self.out_dim,
            self.out_dim * self.rot_output_dim,
            1,
            groups=self.out_dim)

        if init_canonical_direction_out_conv_with_zero:
            self.canonical_direction_fc_out.weight.data.fill_(0.0)
            self.canonical_direction_fc_out.bias.data.fill_(0.0)

    def forward(self,
                p,
                z,
                c,
                shape_template_latent=None,
                motion_template_latent=None):
        assert shape_template_latent is not None
        assert motion_template_latent is not None

        # B, c_dim, 1
        #const = self.const(p).unsqueeze(2)

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()

        shape_const = self.shape_attention_module(shape_template_latent).view(
            batch_size, self.out_dim * self.c64, 1)

        # B, pm_num * decoder_input_dim, 1
        pm_wise_latent = self.encoder(shape_const)

        # B, decoder_input_dim, P
        net = self.fc_p(p)
        # B, pm_num * decoder_input_dim, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)

        # B, pm_num * decoder_input_dim, P
        shared = net + pm_wise_latent
        net = self.decoder(shared)

        out = self.fc_out(self.actvn(self.bn(net)))

        # batch, points, occupancy=out_dim (primitive_num)
        out = out.reshape(batch_size, self.out_dim,
                          T).transpose(1, 2).contiguous()

        diff_net = shared
        for diff_decoder in self.diff_decoders:
            diff_net = diff_decoder(diff_net, c)

        diff_out = self.diff_fc_out(self.actvn(self.diff_bn(diff_net, c)))
        diff_out = diff_out.reshape(batch_size, self.out_dim,
                                    T).transpose(1, 2).contiguous()
        if self.sigmoid_to_template:
            occ = torch.sigmoid(out) * diff_out
        else:
            occ = out * torch.sigmoid(diff_out)

        motion_const = self.motion_attention_module(
            motion_template_latent).view(batch_size, self.out_dim * self.c64,
                                         1)
        canonical_loc = self.canonical_loc_encoder(motion_const)
        canonical_loc = self.canonical_loc_fc_out(
            self.actvn(self.canonical_loc_bn(canonical_loc))).reshape(
                batch_size, self.out_dim, self.dim)

        canonical_direction = self.canonical_direction_encoder(motion_const)
        canonical_direction = self.canonical_direction_fc_out(
            self.actvn(
                self.canonical_direction_bn(canonical_direction))).reshape(
                    batch_size, self.out_dim, self.rot_output_dim)

        ret = dict(occupancy=occ,
                   canonical_occupancy=out,
                   canonical_location=canonical_loc,
                   canonical_direction=canonical_direction)
        return ret


class DecoderQuantizedInputOutMultiplyCanonicalShape(nn.Module):
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
                 leaky=False,
                 shape_quantize_num=4,
                 shape_template_latent_dim=64):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.depth = depth

        self.dim = dim
        self.out_dim = out_dim
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        self.const = ConstantInput(self.c64 * self.out_dim)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))
        self.fc_p = nn.Conv1d(dim, self.decoder_input_dim, 1)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*dec_layers)

        self.bn = nn.BatchNorm1d(self.c64 * self.out_dim)

        self.fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                self.out_dim,
                                1,
                                groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        diff_dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            diff_dec_layers.extend([
                GroupWiseCResnetBlockConv1d(self.c_dim,
                                            self.out_dim,
                                            int(self.c64 / 2**idx),
                                            size_out=int(self.c64 /
                                                         2**(idx - 1)))
            ])

        self.diff_decoders = nn.ModuleList(diff_dec_layers)

        self.diff_bn = CBatchNorm1d(self.c_dim, self.c64 * self.out_dim)

        self.diff_fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                     self.out_dim,
                                     1,
                                     groups=self.out_dim)

        self.shape_const_quantizer = Quantize(self.c64 * self.out_dim,
                                              shape_quantize_num)
        self.fc_shape_c = nn.Conv1d(shape_template_latent_dim,
                                    self.c64 * self.out_dim, 1)
        self.bn_shape_c = nn.BatchNorm1d(shape_template_latent_dim)

    def forward(self, p, z, c, shape_template_latent=None):
        assert shape_template_latent is not None

        # B, c_dim, 1
        #const = self.const(p).unsqueeze(2)

        shape_c = self.fc_shape_c(
            self.actvn(self.bn_shape_c(
                shape_template_latent.unsqueeze(2)))).squeeze(-1)
        shape_const, shape_quantize_diff, _ = self.shape_const_quantizer(
            shape_c)

        # B, pm_num * decoder_input_dim, 1
        shape_const = shape_const.unsqueeze(2)
        pm_wise_latent = self.encoder(shape_const)

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        # B, decoder_input_dim, P
        net = self.fc_p(p)
        # B, pm_num * decoder_input_dim, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)

        # B, pm_num * decoder_input_dim, P
        shared = net + pm_wise_latent
        net = self.decoder(shared)

        out = self.fc_out(self.actvn(self.bn(net)))

        # batch, points, occupancy=out_dim (primitive_num)
        out = out.reshape(batch_size, self.out_dim,
                          T).transpose(1, 2).contiguous()

        diff_net = shared
        for diff_decoder in self.diff_decoders:
            diff_net = diff_decoder(diff_net, c)

        diff_out = self.diff_fc_out(self.actvn(self.diff_bn(diff_net, c)))
        diff_out = diff_out.reshape(batch_size, self.out_dim,
                                    T).transpose(1, 2).contiguous()
        occ = torch.sigmoid(out) * diff_out
        """
        print(
            'diff_weight',
            self.diff_fc_out.weight.mean(),
            self.diff_fc_out.weight[0, 0],
        )
        if self.diff_fc_out.weight.grad is not None:
            print('grad', self.diff_fc_out.weight.grad.mean())
        """
        ret = dict(occupancy=occ,
                   canonical_occupancy=out,
                   latent_quantize_diff=shape_quantize_diff)
        return ret


class Ellipsoid(nn.Module):
    def __init__(self, surface_sample_point_num):
        super().__init__()
        self.surface_sample_point_num = surface_sample_point_num

    def forward(self, x, shape_c, center, abc):
        """
        center: (B, primitive_num, 3)
        abc: (B, primitive_num, 3)
        """

        batch_size = x.size(0)
        # implicit
        #breakpoint()
        center = center.unsqueeze(-3)
        x_centered = x.unsqueeze(-2) - center
        abc = abc.unsqueeze(-3)
        abcxyz = ((x_centered / abc)**2).sum(-1)
        occ = 1 - abcxyz

        # explicit
        sampled_angle = torch.rand(batch_size,
                                   self.surface_sample_point_num,
                                   2,
                                   device=x.device,
                                   dtype=x.dtype)
        sampled_angle[..., 0] = sampled_angle[..., 0] * 2 * np.pi - np.pi
        sampled_angle[..., 1] = sampled_angle[..., 1] * np.pi - np.pi / 2
        sampled_coord_on_ellipsoid_surface = geometry.sphere_polar2cartesian(
            1,
            sampled_angle.unsqueeze(-2),
            a=abc[..., 0],
            b=abc[..., 1],
            c=abc[..., 2]) + center
        return dict(occ=occ, surface_points=sampled_coord_on_ellipsoid_surface)


class DecoderQuantizedInputAllBatchNormAtlasNetV2CBatchAtlasNetV2MultiplyEllipsoid(
        nn.Module):
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
                 leaky=False,
                 rot_output_dim=3,
                 motion_quantize_num=4,
                 shape_quantize_num=4,
                 motion_template_latent_dim=64,
                 shape_template_latent_dim=64,
                 surface_sample_point_num=1024):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.depth = depth
        self.rot_output_dim = rot_output_dim

        self.dim = dim
        self.out_dim = out_dim
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        self.const = ConstantInput(self.c64 * self.out_dim)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))
        self.fc_p = nn.Conv1d(dim, self.decoder_input_dim, 1)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*dec_layers)

        self.ellipsoid = Ellipsoid(surface_sample_point_num)
        self.abc_bn = nn.BatchNorm1d(self.c64 * self.out_dim)

        self.abc_fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                    self.out_dim * 3,
                                    1,
                                    groups=self.out_dim)
        self.center_bn = nn.BatchNorm1d(self.c64 * self.out_dim)

        self.center_fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                       self.out_dim * 3,
                                       1,
                                       groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        diff_dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            diff_dec_layers.extend([
                GroupWiseCResnetBlockConv1d(self.c_dim,
                                            self.out_dim,
                                            int(self.c64 / 2**idx),
                                            size_out=int(self.c64 /
                                                         2**(idx - 1)))
            ])

        self.diff_decoders = nn.ModuleList(diff_dec_layers)

        self.diff_bn = CBatchNorm1d(self.c_dim, self.c64 * self.out_dim)

        self.diff_fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                     self.out_dim,
                                     1,
                                     groups=self.out_dim)

        self.motion_const_quantizer = Quantize(self.c64 * self.out_dim,
                                               motion_quantize_num)
        self.fc_motion_c = nn.Conv1d(motion_template_latent_dim,
                                     self.c64 * self.out_dim, 1)
        self.bn_motion_c = nn.BatchNorm1d(motion_template_latent_dim)

        self.shape_const_quantizer = Quantize(self.c64 * self.out_dim,
                                              shape_quantize_num)
        self.fc_shape_c = nn.Conv1d(shape_template_latent_dim,
                                    self.c64 * self.out_dim, 1)
        self.bn_shape_c = nn.BatchNorm1d(shape_template_latent_dim)

        canonical_loc_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_loc_enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=size_out)
            ])
            idx += 1

        self.canonical_loc_encoder = nn.Sequential(*canonical_loc_enc_layers)

        self.canonical_loc_bn = nn.BatchNorm1d(
            int(self.c64 / 2**idx) * self.out_dim)

        self.canonical_loc_fc_out = nn.Conv1d(int(self.c64 / 2**idx) *
                                              self.out_dim,
                                              self.out_dim * dim,
                                              1,
                                              groups=self.out_dim)

        canonical_direction_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_direction_enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=size_out)
            ])
            idx += 1

        self.canonical_direction_encoder = nn.Sequential(
            *canonical_direction_enc_layers)

        self.canonical_direction_bn = nn.BatchNorm1d(
            int(self.c64 / 2**idx) * self.out_dim)

        self.canonical_direction_fc_out = nn.Conv1d(
            int(self.c64 / 2**idx) * self.out_dim,
            self.out_dim * self.rot_output_dim,
            1,
            groups=self.out_dim)

    def forward(self,
                raw_p,
                z,
                c,
                shape_template_latent=None,
                motion_template_latent=None):
        assert shape_template_latent is not None
        assert motion_template_latent is not None

        # B, c_dim, 1
        #const = self.const(p).unsqueeze(2)

        shape_c = self.fc_shape_c(
            self.actvn(self.bn_shape_c(
                shape_template_latent.unsqueeze(2)))).squeeze(-1)
        shape_const, shape_quantize_diff, _ = self.shape_const_quantizer(
            shape_c)

        # B, pm_num * decoder_input_dim, 1
        shape_const = shape_const.unsqueeze(2)
        pm_wise_latent = self.encoder(shape_const)

        p = raw_p.transpose(1, 2)
        batch_size, D, T = p.size()
        # B, decoder_input_dim, P
        net = self.fc_p(p)
        # B, pm_num * decoder_input_dim, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)

        # B, pm_num * decoder_input_dim, P
        ellipsis_latent = self.decoder(pm_wise_latent)
        abc = F.softplus(
            self.abc_fc_out(self.actvn(self.abc_bn(ellipsis_latent))).view(
                batch_size, self.out_dim, 3) * 10) / 10
        """
        abc = torch.relu(
            self.abc_fc_out(self.actvn(self.abc_bn(ellipsis_latent))).view(
                batch_size, self.out_dim, 3)) + 0.01
        """
        center = torch.tanh(
            self.center_fc_out(
                self.actvn(self.center_bn(ellipsis_latent))).view(
                    batch_size, self.out_dim, 3)) * 0.5

        ellipsoid_ret = self.ellipsoid(raw_p, shape_template_latent, center,
                                       abc)

        out = ellipsoid_ret['occ']
        surface_points = ellipsoid_ret['surface_points']

        shared = net + pm_wise_latent
        diff_net = shared
        for diff_decoder in self.diff_decoders:
            diff_net = diff_decoder(diff_net, c)

        diff_out = self.diff_fc_out(self.actvn(self.diff_bn(diff_net, c)))
        diff_out = diff_out.reshape(batch_size, self.out_dim,
                                    T).transpose(1, 2).contiguous()
        occ = torch.sigmoid(out) * diff_out

        motion_c = self.fc_motion_c(
            self.actvn(self.bn_motion_c(
                motion_template_latent.unsqueeze(2)))).squeeze(-1)
        motion_const, motion_quantize_diff, _ = self.motion_const_quantizer(
            motion_c)

        motion_const = motion_const.unsqueeze(2)
        canonical_loc = self.canonical_loc_encoder(motion_const)
        canonical_loc = self.canonical_loc_fc_out(
            self.actvn(self.canonical_loc_bn(canonical_loc))).reshape(
                batch_size, self.out_dim, self.dim)

        canonical_direction = self.canonical_direction_encoder(motion_const)
        canonical_direction = self.canonical_direction_fc_out(
            self.actvn(
                self.canonical_direction_bn(canonical_direction))).reshape(
                    batch_size, self.out_dim, self.rot_output_dim)

        ret = dict(occupancy=occ,
                   canonical_occupancy=out,
                   surface_points=surface_points,
                   canonical_location=canonical_loc,
                   canonical_direction=canonical_direction,
                   latent_quantize_diff=(shape_quantize_diff +
                                         motion_quantize_diff))
        return ret


class DecoderAttentionInputAllBatchNormAtlasNetV2CBatchAtlasNetV2MultiplyEllipsoid(
        nn.Module):
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
                 leaky=False,
                 rot_output_dim=3,
                 motion_quantize_num=4,
                 shape_quantize_num=4,
                 motion_template_latent_dim=64,
                 shape_template_latent_dim=64,
                 surface_sample_point_num=1024):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.depth = depth
        self.rot_output_dim = rot_output_dim
        self.motion_quantize_num = motion_quantize_num
        self.shape_quantize_num = shape_quantize_num
        self.motion_template_latent_dim = motion_template_latent_dim
        self.shape_template_latent_dim = shape_template_latent_dim

        self.dim = dim
        self.out_dim = out_dim
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        self.const = ConstantInput(self.c64 * self.out_dim)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))
        self.fc_p = nn.Conv1d(dim, self.decoder_input_dim, 1)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*dec_layers)

        self.ellipsoid = Ellipsoid(surface_sample_point_num)
        self.abc_bn = nn.BatchNorm1d(self.c64 * self.out_dim)

        self.abc_fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                    self.out_dim * 3,
                                    1,
                                    groups=self.out_dim)
        self.center_bn = nn.BatchNorm1d(self.c64 * self.out_dim)

        self.center_fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                       self.out_dim * 3,
                                       1,
                                       groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        diff_dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            diff_dec_layers.extend([
                GroupWiseCResnetBlockConv1d(self.c_dim,
                                            self.out_dim,
                                            int(self.c64 / 2**idx),
                                            size_out=int(self.c64 /
                                                         2**(idx - 1)))
            ])

        self.diff_decoders = nn.ModuleList(diff_dec_layers)

        self.diff_bn = CBatchNorm1d(self.c_dim, self.c64 * self.out_dim)

        self.diff_fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                     self.out_dim,
                                     1,
                                     groups=self.out_dim)

        self.motion_attention_module = AttentionConstMemory(
            self.motion_template_latent_dim, self.out_dim,
            self.motion_quantize_num, self.c64)
        self.shape_attention_module = AttentionConstMemory(
            self.shape_template_latent_dim, self.out_dim,
            self.shape_quantize_num, self.c64)

        canonical_loc_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_loc_enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=size_out)
            ])
            idx += 1

        self.canonical_loc_encoder = nn.Sequential(*canonical_loc_enc_layers)

        self.canonical_loc_bn = nn.BatchNorm1d(
            int(self.c64 / 2**idx) * self.out_dim)

        self.canonical_loc_fc_out = nn.Conv1d(int(self.c64 / 2**idx) *
                                              self.out_dim,
                                              self.out_dim * dim,
                                              1,
                                              groups=self.out_dim)

        canonical_direction_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_direction_enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=size_out)
            ])
            idx += 1

        self.canonical_direction_encoder = nn.Sequential(
            *canonical_direction_enc_layers)

        self.canonical_direction_bn = nn.BatchNorm1d(
            int(self.c64 / 2**idx) * self.out_dim)

        self.canonical_direction_fc_out = nn.Conv1d(
            int(self.c64 / 2**idx) * self.out_dim,
            self.out_dim * self.rot_output_dim,
            1,
            groups=self.out_dim)

    def forward(self,
                raw_p,
                z,
                c,
                shape_template_latent=None,
                motion_template_latent=None):
        assert shape_template_latent is not None
        assert motion_template_latent is not None

        p = raw_p.transpose(1, 2)
        batch_size, D, T = p.size()

        shape_const = self.shape_attention_module(shape_template_latent).view(
            batch_size, self.out_dim * self.c64, 1)
        pm_wise_latent = self.encoder(shape_const)

        # B, decoder_input_dim, P
        net = self.fc_p(p)
        # B, pm_num * decoder_input_dim, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)

        # B, pm_num * decoder_input_dim, P
        ellipsis_latent = self.decoder(pm_wise_latent)

        abc = F.softplus(
            self.abc_fc_out(self.actvn(self.abc_bn(ellipsis_latent))).view(
                batch_size, self.out_dim, 3)) * 0.3
        center = torch.tanh(
            self.center_fc_out(
                self.actvn(self.center_bn(ellipsis_latent))).view(
                    batch_size, self.out_dim, 3)) * 0.5

        ellipsoid_ret = self.ellipsoid(raw_p, shape_template_latent, center,
                                       abc)

        out = ellipsoid_ret['occ']
        surface_points = ellipsoid_ret['surface_points']

        shared = net + pm_wise_latent
        diff_net = shared
        for diff_decoder in self.diff_decoders:
            diff_net = diff_decoder(diff_net, c)

        diff_out = self.diff_fc_out(self.actvn(self.diff_bn(diff_net, c)))
        diff_out = diff_out.reshape(batch_size, self.out_dim,
                                    T).transpose(1, 2).contiguous()
        occ = torch.sigmoid(out) * diff_out

        motion_const = self.motion_attention_module(
            motion_template_latent).view(batch_size, self.out_dim * self.c64,
                                         1)
        canonical_loc = self.canonical_loc_encoder(motion_const)
        canonical_loc = self.canonical_loc_fc_out(
            self.actvn(self.canonical_loc_bn(canonical_loc))).reshape(
                batch_size, self.out_dim, self.dim)

        canonical_direction = self.canonical_direction_encoder(motion_const)
        canonical_direction = self.canonical_direction_fc_out(
            self.actvn(
                self.canonical_direction_bn(canonical_direction))).reshape(
                    batch_size, self.out_dim, self.rot_output_dim)

        ret = dict(occupancy=occ,
                   canonical_occupancy=out,
                   surface_points=surface_points,
                   canonical_location=canonical_loc,
                   canonical_direction=canonical_direction)
        return ret


class DecoderAttentionInputAllBatchNormAtlasNetV2OnlyCanonicalEllipsoid(
        nn.Module):
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
                 leaky=False,
                 rot_output_dim=3,
                 motion_quantize_num=4,
                 shape_quantize_num=4,
                 motion_template_latent_dim=64,
                 shape_template_latent_dim=64,
                 surface_sample_point_num=1024):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.depth = depth
        self.rot_output_dim = rot_output_dim
        self.motion_quantize_num = motion_quantize_num
        self.shape_quantize_num = shape_quantize_num
        self.motion_template_latent_dim = motion_template_latent_dim
        self.shape_template_latent_dim = shape_template_latent_dim

        self.dim = dim
        self.out_dim = out_dim
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        self.const = ConstantInput(self.c64 * self.out_dim)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))
        self.fc_p = nn.Conv1d(dim, self.decoder_input_dim, 1)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*dec_layers)

        self.ellipsoid = Ellipsoid(surface_sample_point_num)
        self.abc_bn = nn.BatchNorm1d(self.c64 * self.out_dim)

        self.abc_fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                    self.out_dim * 3,
                                    1,
                                    groups=self.out_dim)
        self.center_bn = nn.BatchNorm1d(self.c64 * self.out_dim)

        self.center_fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                       self.out_dim * 3,
                                       1,
                                       groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.motion_attention_module = AttentionConstMemory(
            self.motion_template_latent_dim, self.out_dim,
            self.motion_quantize_num, self.c64)
        self.shape_attention_module = AttentionConstMemory(
            self.shape_template_latent_dim, self.out_dim,
            self.shape_quantize_num, self.c64)

        canonical_loc_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_loc_enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=size_out)
            ])
            idx += 1

        self.canonical_loc_encoder = nn.Sequential(*canonical_loc_enc_layers)

        self.canonical_loc_bn = nn.BatchNorm1d(
            int(self.c64 / 2**idx) * self.out_dim)

        self.canonical_loc_fc_out = nn.Conv1d(int(self.c64 / 2**idx) *
                                              self.out_dim,
                                              self.out_dim * dim,
                                              1,
                                              groups=self.out_dim)

        canonical_direction_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_direction_enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=size_out)
            ])
            idx += 1

        self.canonical_direction_encoder = nn.Sequential(
            *canonical_direction_enc_layers)

        self.canonical_direction_bn = nn.BatchNorm1d(
            int(self.c64 / 2**idx) * self.out_dim)

        self.canonical_direction_fc_out = nn.Conv1d(
            int(self.c64 / 2**idx) * self.out_dim,
            self.out_dim * self.rot_output_dim,
            1,
            groups=self.out_dim)

    def forward(self,
                raw_p,
                z,
                c,
                shape_template_latent=None,
                motion_template_latent=None):
        assert shape_template_latent is not None
        assert motion_template_latent is not None

        p = raw_p.transpose(1, 2)
        batch_size, D, T = p.size()

        shape_const = self.shape_attention_module(shape_template_latent).view(
            batch_size, self.out_dim * self.c64, 1)
        pm_wise_latent = self.encoder(shape_const)

        # B, decoder_input_dim, P
        net = self.fc_p(p)
        # B, pm_num * decoder_input_dim, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)

        # B, pm_num * decoder_input_dim, P
        ellipsis_latent = self.decoder(pm_wise_latent)

        abc = F.softplus(
            self.abc_fc_out(self.actvn(self.abc_bn(ellipsis_latent))).view(
                batch_size, self.out_dim, 3)) * 0.3
        center = torch.tanh(
            self.center_fc_out(
                self.actvn(self.center_bn(ellipsis_latent))).view(
                    batch_size, self.out_dim, 3)) * 0.5

        ellipsoid_ret = self.ellipsoid(raw_p, shape_template_latent, center,
                                       abc)

        out = ellipsoid_ret['occ']
        surface_points = ellipsoid_ret['surface_points']

        shared = net + pm_wise_latent
        motion_const = self.motion_attention_module(
            motion_template_latent).view(batch_size, self.out_dim * self.c64,
                                         1)
        canonical_loc = self.canonical_loc_encoder(motion_const)
        canonical_loc = self.canonical_loc_fc_out(
            self.actvn(self.canonical_loc_bn(canonical_loc))).reshape(
                batch_size, self.out_dim, self.dim)

        canonical_direction = self.canonical_direction_encoder(motion_const)
        canonical_direction = self.canonical_direction_fc_out(
            self.actvn(
                self.canonical_direction_bn(canonical_direction))).reshape(
                    batch_size, self.out_dim, self.rot_output_dim)

        ret = dict(occupancy=out,
                   canonical_occupancy=out,
                   surface_points=surface_points,
                   canonical_location=canonical_loc,
                   canonical_direction=canonical_direction)
        return ret


class DecoderConstInputAllBatchNormAtlasNetV2CBatchAtlasNetV2Multiply(
        nn.Module):
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
                 leaky=False,
                 rot_output_dim=3,
                 attention_type='pm_wise_attention',
                 init_canonical_direction_out_conv_with_zero=False,
                 init_canonical_location_out_conv_with_zero=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.depth = depth
        self.rot_output_dim = rot_output_dim

        self.dim = dim
        self.out_dim = out_dim
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))
        self.fc_p = nn.Conv1d(dim, self.decoder_input_dim, 1)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*dec_layers)

        self.bn = nn.BatchNorm1d(self.c64 * self.out_dim)

        self.fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                self.out_dim,
                                1,
                                groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        diff_dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            diff_dec_layers.extend([
                GroupWiseCResnetBlockConv1d(self.c_dim,
                                            self.out_dim,
                                            int(self.c64 / 2**idx),
                                            size_out=int(self.c64 /
                                                         2**(idx - 1)))
            ])

        self.diff_decoders = nn.ModuleList(diff_dec_layers)

        self.diff_bn = CBatchNorm1d(self.c_dim, self.c64 * self.out_dim)

        self.diff_fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                     self.out_dim,
                                     1,
                                     groups=self.out_dim)

        self.motion_const_module = ConstantInput(self.out_dim * self.c64)
        self.shape_const_module = ConstantInput(self.out_dim * self.c64)

        canonical_loc_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_loc_enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=size_out)
            ])
            idx += 1

        self.canonical_loc_encoder = nn.Sequential(*canonical_loc_enc_layers)

        self.canonical_loc_bn = nn.BatchNorm1d(
            int(self.c64 / 2**idx) * self.out_dim)

        self.canonical_loc_fc_out = nn.Conv1d(int(self.c64 / 2**idx) *
                                              self.out_dim,
                                              self.out_dim * dim,
                                              1,
                                              groups=self.out_dim)

        if init_canonical_location_out_conv_with_zero:
            self.canonical_loc_fc_out.weight.data.fill_(0.0)
            self.canonical_loc_fc_out.bias.data.fill_(0.0)

        canonical_direction_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_direction_enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=size_out)
            ])
            idx += 1

        self.canonical_direction_encoder = nn.Sequential(
            *canonical_direction_enc_layers)

        self.canonical_direction_bn = nn.BatchNorm1d(
            int(self.c64 / 2**idx) * self.out_dim)

        self.canonical_direction_fc_out = nn.Conv1d(
            int(self.c64 / 2**idx) * self.out_dim,
            self.out_dim * self.rot_output_dim,
            1,
            groups=self.out_dim)

        if init_canonical_direction_out_conv_with_zero:
            self.canonical_direction_fc_out.weight.data.fill_(0.0)
            self.canonical_direction_fc_out.bias.data.fill_(0.0)

    def forward(self, p, z, c):

        # B, c_dim, 1
        #const = self.const(p).unsqueeze(2)

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()

        shape_const = self.shape_const_module(p).view(batch_size,
                                                      self.out_dim * self.c64,
                                                      1)

        # B, pm_num * decoder_input_dim, 1
        pm_wise_latent = self.encoder(shape_const)

        # B, decoder_input_dim, P
        net = self.fc_p(p)
        # B, pm_num * decoder_input_dim, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)

        # B, pm_num * decoder_input_dim, P
        shared = net + pm_wise_latent
        net = self.decoder(shared)

        out = self.fc_out(self.actvn(self.bn(net)))

        # batch, points, occupancy=out_dim (primitive_num)
        out = out.reshape(batch_size, self.out_dim,
                          T).transpose(1, 2).contiguous()

        diff_net = shared
        for diff_decoder in self.diff_decoders:
            diff_net = diff_decoder(diff_net, c)

        diff_out = self.diff_fc_out(self.actvn(self.diff_bn(diff_net, c)))
        diff_out = diff_out.reshape(batch_size, self.out_dim,
                                    T).transpose(1, 2).contiguous()
        occ = torch.sigmoid(out) * diff_out

        motion_const = self.motion_const_module(p).view(
            batch_size, self.out_dim * self.c64, 1)
        canonical_loc = self.canonical_loc_encoder(motion_const)
        canonical_loc = self.canonical_loc_fc_out(
            self.actvn(self.canonical_loc_bn(canonical_loc))).reshape(
                batch_size, self.out_dim, self.dim)

        canonical_direction = self.canonical_direction_encoder(motion_const)
        canonical_direction = self.canonical_direction_fc_out(
            self.actvn(
                self.canonical_direction_bn(canonical_direction))).reshape(
                    batch_size, self.out_dim, self.rot_output_dim)

        ret = dict(occupancy=occ,
                   canonical_occupancy=out,
                   canonical_location=canonical_loc,
                   canonical_direction=canonical_direction)
        return ret


class DecoderAttentionInputAllBatchNormAtlasNetV2CBatchAtlasNetV2Explicit(
        nn.Module):
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
                 leaky=False,
                 rot_output_dim=3,
                 motion_quantize_num=4,
                 shape_quantize_num=4,
                 motion_template_latent_dim=64,
                 shape_template_latent_dim=64,
                 init_canonical_direction_out_conv_with_zero=False,
                 init_canonical_location_out_conv_with_zero=False,
                 surface_sample_point_num=1024):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.depth = depth
        self.rot_output_dim = rot_output_dim
        self.motion_quantize_num = motion_quantize_num
        self.shape_quantize_num = shape_quantize_num
        self.motion_template_latent_dim = motion_template_latent_dim
        self.shape_template_latent_dim = shape_template_latent_dim
        self.surface_sample_point_num = surface_sample_point_num
        self.dim = dim
        self.out_dim = out_dim
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        self.const = ConstantInput(self.c64 * self.out_dim)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))
        self.fc_p = nn.Conv1d(2, self.decoder_input_dim, 1)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*dec_layers)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        diff_dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            diff_dec_layers.extend([
                GroupWiseCResnetBlockConv1d(self.c_dim,
                                            self.out_dim,
                                            int(self.c64 / 2**idx),
                                            size_out=int(self.c64 /
                                                         2**(idx - 1)))
            ])

        self.diff_decoders = nn.ModuleList(diff_dec_layers)

        self.diff_bn = CBatchNorm1d(self.c_dim, self.c64 * self.out_dim)

        self.diff_fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                     self.out_dim * 3,
                                     1,
                                     groups=self.out_dim)

        self.motion_attention_module = AttentionConstMemory(
            self.motion_template_latent_dim, self.out_dim,
            self.motion_quantize_num, self.c64)
        self.shape_attention_module = AttentionConstMemory(
            self.shape_template_latent_dim, self.out_dim,
            self.shape_quantize_num, self.c64)

        canonical_loc_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_loc_enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=size_out)
            ])
            idx += 1

        self.canonical_loc_encoder = nn.Sequential(*canonical_loc_enc_layers)

        self.canonical_loc_bn = nn.BatchNorm1d(
            int(self.c64 / 2**idx) * self.out_dim)

        self.canonical_loc_fc_out = nn.Conv1d(int(self.c64 / 2**idx) *
                                              self.out_dim,
                                              self.out_dim * dim,
                                              1,
                                              groups=self.out_dim)

        canonical_direction_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_direction_enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=size_out)
            ])
            idx += 1

        self.canonical_direction_encoder = nn.Sequential(
            *canonical_direction_enc_layers)

        self.canonical_direction_bn = nn.BatchNorm1d(
            int(self.c64 / 2**idx) * self.out_dim)

        self.canonical_direction_fc_out = nn.Conv1d(
            int(self.c64 / 2**idx) * self.out_dim,
            self.out_dim * self.rot_output_dim,
            1,
            groups=self.out_dim)

        if init_canonical_direction_out_conv_with_zero:
            self.canonical_direction_fc_out.weight.data.fill_(0.0)
            self.canonical_direction_fc_out.bias.data.fill_(0.0)

        if init_canonical_location_out_conv_with_zero:
            self.canonical_loc_fc_out.weight.data.fill_(0.0)
            self.canonical_loc_fc_out.bias.data.fill_(0.0)

    def forward(self,
                raw_p,
                z,
                c,
                shape_template_latent=None,
                motion_template_latent=None):
        assert shape_template_latent is not None
        assert motion_template_latent is not None

        p = raw_p.transpose(1, 2)
        batch_size, D, T = p.size()

        shape_const = self.shape_attention_module(shape_template_latent).view(
            batch_size, self.out_dim * self.c64, 1)
        pm_wise_latent = self.encoder(shape_const)

        planar_points = torch.rand(batch_size,
                                   2,
                                   self.surface_sample_point_num,
                                   device=p.device,
                                   dtype=p.dtype)
        # B, decoder_input_dim, P
        net = self.fc_p(planar_points)
        # B, pm_num * decoder_input_dim, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1, -1).reshape(
            batch_size, -1, self.surface_sample_point_num)

        shared = net + pm_wise_latent
        diff_net = shared
        for diff_decoder in self.diff_decoders:
            diff_net = diff_decoder(diff_net, c)

        diff_out = self.diff_fc_out(self.actvn(self.diff_bn(diff_net, c)))
        surface_points = torch.tanh(
            diff_out.reshape(batch_size, self.out_dim * 3,
                             self.surface_sample_point_num).transpose(
                                 1, 2).contiguous().view(
                                     batch_size, self.surface_sample_point_num,
                                     self.out_dim, 3))

        motion_const = self.motion_attention_module(
            motion_template_latent).view(batch_size, self.out_dim * self.c64,
                                         1)
        canonical_loc = self.canonical_loc_encoder(motion_const)
        canonical_loc = self.canonical_loc_fc_out(
            self.actvn(self.canonical_loc_bn(canonical_loc))).reshape(
                batch_size, self.out_dim, self.dim)

        canonical_direction = self.canonical_direction_encoder(motion_const)
        canonical_direction = self.canonical_direction_fc_out(
            self.actvn(
                self.canonical_direction_bn(canonical_direction))).reshape(
                    batch_size, self.out_dim, self.rot_output_dim)

        occ = torch.zeros(batch_size,
                          T,
                          self.out_dim,
                          device=p.device,
                          dtype=p.dtype)
        ret = dict(occupancy=occ,
                   canonical_occupancy=occ,
                   surface_points=surface_points,
                   canonical_location=canonical_loc,
                   canonical_direction=canonical_direction)
        return ret


class DecoderAttentionInputAllBatchNormAtlasNetV2CBatchAtlasNetV2ExplicitDiff(
        nn.Module):
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
                 leaky=False,
                 rot_output_dim=3,
                 motion_quantize_num=4,
                 shape_quantize_num=4,
                 motion_template_latent_dim=64,
                 shape_template_latent_dim=64,
                 init_canonical_direction_out_conv_with_zero=False,
                 init_canonical_location_out_conv_with_zero=False,
                 surface_sample_point_num=1024,
                 only_const=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.depth = depth
        self.rot_output_dim = rot_output_dim
        self.motion_quantize_num = motion_quantize_num
        self.shape_quantize_num = shape_quantize_num
        self.motion_template_latent_dim = motion_template_latent_dim
        self.shape_template_latent_dim = shape_template_latent_dim
        self.surface_sample_point_num = surface_sample_point_num
        self.dim = dim
        self.out_dim = out_dim
        self.c64 = int(hidden_size / self.out_dim)
        self.only_const = only_const
        assert depth % 2 == 0

        self.const = ConstantInput(self.c64 * self.out_dim)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))
        self.fc_p = nn.Conv1d(2, self.decoder_input_dim, 1)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*dec_layers)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        diff_dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            diff_dec_layers.extend([
                GroupWiseCResnetBlockConv1d(self.c_dim,
                                            self.out_dim,
                                            int(self.c64 / 2**idx),
                                            size_out=int(self.c64 /
                                                         2**(idx - 1)))
            ])

        self.diff_decoders = nn.ModuleList(diff_dec_layers)

        self.diff_bn = CBatchNorm1d(self.c_dim, self.c64 * self.out_dim)

        self.diff_fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                     self.out_dim * 3,
                                     1,
                                     groups=self.out_dim)

        self.motion_attention_module = AttentionConstMemory(
            self.motion_template_latent_dim, self.out_dim,
            self.motion_quantize_num, self.c64)
        self.shape_attention_module = AttentionConstMemory(
            self.shape_template_latent_dim, self.out_dim,
            self.shape_quantize_num, self.c64)

        canonical_loc_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_loc_enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=size_out)
            ])
            idx += 1

        self.canonical_loc_encoder = nn.Sequential(*canonical_loc_enc_layers)

        self.canonical_loc_bn = nn.BatchNorm1d(
            int(self.c64 / 2**idx) * self.out_dim)

        self.canonical_loc_fc_out = nn.Conv1d(int(self.c64 / 2**idx) *
                                              self.out_dim,
                                              self.out_dim * dim,
                                              1,
                                              groups=self.out_dim)

        canonical_direction_enc_layers = []
        idx = 0
        while True:
            size_out = int(self.c64 / 2**(idx + 1))
            if size_out < dim * 2:
                break
            canonical_direction_enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=size_out)
            ])
            idx += 1

        self.canonical_direction_encoder = nn.Sequential(
            *canonical_direction_enc_layers)

        self.canonical_direction_bn = nn.BatchNorm1d(
            int(self.c64 / 2**idx) * self.out_dim)

        self.canonical_direction_fc_out = nn.Conv1d(
            int(self.c64 / 2**idx) * self.out_dim,
            self.out_dim * self.rot_output_dim,
            1,
            groups=self.out_dim)

        if init_canonical_direction_out_conv_with_zero:
            self.canonical_direction_fc_out.weight.data.fill_(0.0)
            self.canonical_direction_fc_out.bias.data.fill_(0.0)

        if init_canonical_location_out_conv_with_zero:
            self.canonical_loc_fc_out.weight.data.fill_(0.0)
            self.canonical_loc_fc_out.bias.data.fill_(0.0)

        self.bn = nn.BatchNorm1d(self.c64 * self.out_dim)

        self.fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                self.out_dim * 3,
                                1,
                                groups=self.out_dim)

    def forward(self,
                raw_p,
                z,
                c,
                shape_template_latent=None,
                motion_template_latent=None):
        assert shape_template_latent is not None
        assert motion_template_latent is not None

        p = raw_p.transpose(1, 2)
        batch_size, D, T = p.size()

        shape_const = self.shape_attention_module(shape_template_latent).view(
            batch_size, self.out_dim * self.c64, 1)
        pm_wise_latent = self.encoder(shape_const)

        planar_points = torch.rand(batch_size,
                                   2,
                                   self.surface_sample_point_num,
                                   device=p.device,
                                   dtype=p.dtype)
        # B, decoder_input_dim, P
        net = self.fc_p(planar_points)
        # B, pm_num * decoder_input_dim, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1, -1).reshape(
            batch_size, -1, self.surface_sample_point_num)

        shared = net + pm_wise_latent

        out = self.fc_out(self.actvn(self.bn(self.decoder(shared))))
        surface_points = out.reshape(
            batch_size, self.out_dim * 3,
            self.surface_sample_point_num).transpose(1, 2).contiguous().view(
                batch_size, self.surface_sample_point_num, self.out_dim, 3)

        diff_net = shared
        for diff_decoder in self.diff_decoders:
            diff_net = diff_decoder(diff_net, c)

        diff_out = self.diff_fc_out(self.actvn(self.diff_bn(diff_net, c)))
        surface_points_diff = diff_out.reshape(
            batch_size, self.out_dim * 3,
            self.surface_sample_point_num).transpose(1, 2).contiguous().view(
                batch_size, self.surface_sample_point_num, self.out_dim, 3)

        if self.only_const:
            surface_points = torch.tanh(surface_points)
        else:
            surface_points = torch.tanh(surface_points_diff + surface_points)

        motion_const = self.motion_attention_module(
            motion_template_latent).view(batch_size, self.out_dim * self.c64,
                                         1)
        canonical_loc = self.canonical_loc_encoder(motion_const)
        canonical_loc = self.canonical_loc_fc_out(
            self.actvn(self.canonical_loc_bn(canonical_loc))).reshape(
                batch_size, self.out_dim, self.dim)

        canonical_direction = self.canonical_direction_encoder(motion_const)
        canonical_direction = self.canonical_direction_fc_out(
            self.actvn(
                self.canonical_direction_bn(canonical_direction))).reshape(
                    batch_size, self.out_dim, self.rot_output_dim)

        occ = torch.zeros(batch_size,
                          T,
                          self.out_dim,
                          device=p.device,
                          dtype=p.dtype)
        ret = dict(occupancy=occ,
                   canonical_occupancy=occ,
                   surface_points=surface_points,
                   canonical_location=canonical_loc,
                   canonical_direction=canonical_direction)
        return ret


class DecoderConstInputAllBatchNormMultiply(nn.Module):
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

        self.dim = dim
        self.out_dim = out_dim
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))
        self.fc_p = nn.Conv1d(dim, self.decoder_input_dim, 1)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*dec_layers)

        self.bn = nn.BatchNorm1d(self.c64 * self.out_dim)

        self.fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                self.out_dim,
                                1,
                                groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        diff_dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            diff_dec_layers.extend([
                GroupWiseCResnetBlockConv1d(self.c_dim,
                                            self.out_dim,
                                            int(self.c64 / 2**idx),
                                            size_out=int(self.c64 /
                                                         2**(idx - 1)))
            ])

        self.diff_decoders = nn.ModuleList(diff_dec_layers)

        self.diff_bn = CBatchNorm1d(self.c_dim, self.c64 * self.out_dim)

        self.diff_fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                     self.out_dim,
                                     1,
                                     groups=self.out_dim)

        self.shape_const_module = ConstantInput(self.out_dim * self.c64)

    def forward(self, p, z, c):

        # B, c_dim, 1
        #const = self.const(p).unsqueeze(2)

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()

        shape_const = self.shape_const_module(p).view(batch_size,
                                                      self.out_dim * self.c64,
                                                      1)

        # B, pm_num * decoder_input_dim, 1
        pm_wise_latent = self.encoder(shape_const)

        # B, decoder_input_dim, P
        net = self.fc_p(p)
        # B, pm_num * decoder_input_dim, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)

        # B, pm_num * decoder_input_dim, P
        shared = net + pm_wise_latent
        net = self.decoder(shared)

        out = self.fc_out(self.actvn(self.bn(net)))

        # batch, points, occupancy=out_dim (primitive_num)
        out = out.reshape(batch_size, self.out_dim,
                          T).transpose(1, 2).contiguous()

        diff_net = shared
        for diff_decoder in self.diff_decoders:
            diff_net = diff_decoder(diff_net, c)

        diff_out = self.diff_fc_out(self.actvn(self.diff_bn(diff_net, c)))
        diff_out = diff_out.reshape(batch_size, self.out_dim,
                                    T).transpose(1, 2).contiguous()
        occ = torch.sigmoid(out) * diff_out

        ret = dict(occupancy=occ, canonical_occupancy=out)
        return ret


class DecoderQuantizedInputAllBatchNormMultiply(nn.Module):
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
                 leaky=False,
                 shape_quantize_num=4,
                 shape_template_latent_dim=64,
                 disable_diff=False,
                 batch_norm_momentum=0.1,
                 batch_norm_type='batch_norm',
                 is_extra_non_moving_capacity=False,
                 extra_non_moving_capacity_type='multiple_group_wise_channel',
                 coord_scale=1,
                 extra_non_moving_capacity_extra_prim_num=3,
                 static_primitive_hidden_size=64,
                 diff_hidden_size=None,
                 diff_decoder_type='default',
                 disable_canonical=False,
                 disable_learnable_quantized_latent_vectors=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.depth = depth
        self.coord_scale = coord_scale
        self.disable_diff = disable_diff
        self.disable_canonical = disable_canonical
        assert not (self.disable_diff and self.disable_canonical)
        self.is_extra_non_moving_capacity = is_extra_non_moving_capacity
        self.extra_non_moving_capacity_type = extra_non_moving_capacity_type
        assert self.extra_non_moving_capacity_type in [
            'multiple_group_wise_channel', 'separate_static_primitive'
        ]
        self.extra_non_moving_capacity_extra_prim_num = extra_non_moving_capacity_extra_prim_num
        self.disable_learnable_quantized_latent_vectors = disable_learnable_quantized_latent_vectors
        self.diff_decoder_type = diff_decoder_type

        self.dim = dim
        if self.is_extra_non_moving_capacity:
            if self.extra_non_moving_capacity_type == 'multiple_group_wise_channel':
                self.out_dim = out_dim + self.extra_non_moving_capacity_extra_prim_num
            elif self.extra_non_moving_capacity_type == 'separate_static_primitive':
                self.out_dim = out_dim - 1
        else:
            self.out_dim = out_dim
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0
        self.static_primitive_hidden_size = static_primitive_hidden_size

        self.const = ConstantInput(self.c64 * self.out_dim)

        enc_layers = []
        for idx in range(int(depth / 2)):
            if batch_norm_type == 'batch_norm':
                enc_layers.extend([
                    GroupWiseResnetBlockConv1d(
                        self.out_dim,
                        int(self.c64 / 2**idx),
                        size_out=int(self.c64 / 2**(idx + 1)),
                        batch_norm_momentum=batch_norm_momentum)
                ])
            elif batch_norm_type == 'layer_norm':
                enc_layers.extend([
                    layers.GroupWiseResnetBlockConv1dGBN(
                        self.out_dim,
                        int(self.c64 / 2**idx),
                        size_out=int(self.c64 / 2**(idx + 1)))
                ])
            elif batch_norm_type == 'no_norm':
                enc_layers.extend([
                    GroupWiseResnetBlockConv1dWoBN(self.out_dim,
                                                   int(self.c64 / 2**idx),
                                                   size_out=int(self.c64 /
                                                                2**(idx + 1)))
                ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))
        self.fc_p = nn.Conv1d(dim, self.decoder_input_dim, 1)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            if batch_norm_type == 'batch_norm':
                dec_layers.extend([
                    GroupWiseResnetBlockConv1d(
                        self.out_dim,
                        int(self.c64 / 2**idx),
                        size_out=int(self.c64 / 2**(idx - 1)),
                        batch_norm_momentum=batch_norm_momentum)
                ])
            elif batch_norm_type == 'layer_norm':
                dec_layers.extend([
                    layers.GroupWiseResnetBlockConv1dGBN(
                        self.out_dim,
                        int(self.c64 / 2**idx),
                        size_out=int(self.c64 / 2**(idx - 1)))
                ])
            elif batch_norm_type == 'no_norm':
                dec_layers.extend([
                    GroupWiseResnetBlockConv1dWoBN(self.out_dim,
                                                   int(self.c64 / 2**idx),
                                                   size_out=int(self.c64 /
                                                                2**(idx - 1)))
                ])
        self.decoder = nn.Sequential(*dec_layers)

        if batch_norm_type == 'batch_norm':
            self.bn = nn.BatchNorm1d(self.c64 * self.out_dim,
                                     momentum=batch_norm_momentum)
        elif batch_norm_type == 'layer_norm':
            self.bn = nn.GroupNorm(self.out_dim, self.c64 * self.out_dim)
        elif batch_norm_type == 'no_norm':
            self.bn = nn.Identity()

        self.fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                self.out_dim,
                                1,
                                groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        if diff_hidden_size is None:
            diff_c64 = self.c64
        else:
            diff_c64 = int(diff_hidden_size / self.out_dim)
        diff_dec_layers = []
        if diff_decoder_type == 'default':
            for iidx, idx in enumerate(range(int(depth / 2), 0, -1)):
                if iidx == 0:
                    in_c64 = self.c64
                else:
                    in_c64 = diff_c64

                diff_dec_layers.extend([
                    GroupWiseCResnetBlockConv1d(
                        self.c_dim,
                        self.out_dim,
                        int(in_c64 / 2**idx),
                        size_out=int(diff_c64 / 2**(idx - 1)),
                        norm_method=batch_norm_type,
                        batch_norm_momentum=batch_norm_momentum)
                ])
            self.diff_decoders = nn.ModuleList(diff_dec_layers)

            self.diff_bn = CBatchNorm1d(
                self.c_dim,
                diff_c64 * self.out_dim,
                norm_method=batch_norm_type.replace('layer_norm',
                                                    'group_norm'),
                groups=self.out_dim,
                batch_norm_momentum=batch_norm_momentum)

            self.diff_fc_out = nn.Conv1d(diff_c64 * self.out_dim,
                                         self.out_dim,
                                         1,
                                         groups=self.out_dim)
        elif diff_decoder_type == 'independent_onet':
            self.diff_fc_p = nn.Conv1d(dim, diff_c64 * self.out_dim, 1)
            for iidx, idx in enumerate(range(depth)):
                diff_dec_layers.extend([
                    GroupWiseCResnetBlockConv1d(
                        self.c_dim,
                        self.out_dim,
                        diff_c64,
                        norm_method=batch_norm_type,
                        batch_norm_momentum=batch_norm_momentum)
                ])

            self.diff_decoders = nn.ModuleList(diff_dec_layers)

            self.diff_bn = CBatchNorm1d(
                self.c_dim,
                diff_c64 * self.out_dim,
                norm_method=batch_norm_type.replace('layer_norm',
                                                    'group_norm'),
                groups=self.out_dim,
                batch_norm_momentum=batch_norm_momentum)

            self.diff_fc_out = nn.Conv1d(diff_c64 * self.out_dim,
                                         self.out_dim,
                                         1,
                                         groups=self.out_dim)
        else:
            raise NotImplementedError

        self.shape_const_quantizer = Quantize(
            self.c64 * self.out_dim,
            shape_quantize_num,
            disable_learnable_quantized_latent_vectors=
            disable_learnable_quantized_latent_vectors)
        self.fc_shape_c = nn.Conv1d(shape_template_latent_dim,
                                    self.c64 * self.out_dim, 1)

        if batch_norm_type == 'batch_norm':
            self.bn_shape_c = nn.BatchNorm1d(shape_template_latent_dim,
                                             momentum=batch_norm_momentum)
        elif batch_norm_type == 'layer_norm':
            self.bn_shape_c = nn.LayerNorm([shape_template_latent_dim, 1])
        elif batch_norm_type == 'no_norm':
            self.bn_shape_c = nn.Identity()

        if self.is_extra_non_moving_capacity:
            if self.extra_non_moving_capacity_type == 'separate_static_primitive':
                self.static_pm_model = DecoderQuantizedInputAllBatchNormMultiply(
                    dim=dim,
                    z_dim=z_dim,
                    c_dim=c_dim,
                    hidden_size=self.static_primitive_hidden_size,
                    depth=depth,
                    out_dim=1,
                    leaky=leaky,
                    shape_quantize_num=shape_quantize_num,
                    shape_template_latent_dim=shape_template_latent_dim,
                    disable_canonical=disable_canonical,
                    batch_norm_momentum=batch_norm_momentum,
                    batch_norm_type=batch_norm_type,
                    coord_scale=coord_scale,
                    diff_hidden_size=diff_hidden_size,
                    is_extra_non_moving_capacity=False,
                    diff_decoder_type=diff_decoder_type,
                    disable_learnable_quantized_latent_vectors=
                    disable_learnable_quantized_latent_vectors)

    def forward(self, p, z, c, shape_template_latent=None):
        assert shape_template_latent is not None
        if self.is_extra_non_moving_capacity:
            if self.extra_non_moving_capacity_type == 'separate_static_primitive':
                static_pm_ret = self.static_pm_model(p, z, c,
                                                     shape_template_latent)
        p = p * self.coord_scale
        # B, c_dim, 1
        #const = self.const(p).unsqueeze(2)

        shape_c = self.fc_shape_c(
            self.actvn(self.bn_shape_c(
                shape_template_latent.unsqueeze(2)))).squeeze(-1)
        shape_const, shape_quantize_diff, shape_quantize_indices = self.shape_const_quantizer(
            shape_c)

        # B, pm_num * decoder_input_dim, 1
        shape_const = shape_const.unsqueeze(2)
        pm_wise_latent = self.encoder(shape_const)

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        # B, decoder_input_dim, P
        net = self.fc_p(p)
        # B, pm_num * decoder_input_dim, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)

        # B, pm_num * decoder_input_dim, P
        shared = net + pm_wise_latent
        net = self.decoder(shared)

        out = self.fc_out(self.actvn(self.bn(net)))

        # batch, points, occupancy=out_dim (primitive_num)
        out = out.reshape(batch_size, self.out_dim,
                          T).transpose(1, 2).contiguous()

        if self.diff_decoder_type == 'default':
            diff_net = shared
            for diff_decoder in self.diff_decoders:
                diff_net = diff_decoder(diff_net, c)

            diff_out = self.diff_fc_out(self.actvn(self.diff_bn(diff_net, c)))
            diff_out = diff_out.reshape(batch_size, self.out_dim,
                                        T).transpose(1, 2).contiguous()
        elif self.diff_decoder_type == 'independent_onet':
            diff_net = self.diff_fc_p(p)
            for diff_decoder in self.diff_decoders:
                diff_net = diff_decoder(diff_net, c)

            diff_out = self.diff_fc_out(self.actvn(self.diff_bn(diff_net, c)))
            diff_out = diff_out.reshape(batch_size, self.out_dim,
                                        T).transpose(1, 2).contiguous()
        else:
            raise NotImplementedError
        if self.disable_diff:
            occ = out
        elif self.disable_canonical:
            occ = diff_out
        else:
            occ = torch.sigmoid(out) * diff_out

        if self.is_extra_non_moving_capacity:
            if self.extra_non_moving_capacity_type == 'multiple_group_wise_channel':
                occ_static = occ[:, :, :(
                    1 + self.extra_non_moving_capacity_extra_prim_num)].max(
                        -1, keepdim=True)[0]
                occ_moving = occ[:, :, (
                    1 + self.extra_non_moving_capacity_extra_prim_num):]
                occ = torch.cat([occ_static, occ_moving], dim=-1)

                out_static = out[:, :, :(
                    1 + self.extra_non_moving_capacity_extra_prim_num)].max(
                        -1, keepdim=True)[0]
                out_moving = out[:, :, (
                    1 + self.extra_non_moving_capacity_extra_prim_num):]
                out = torch.cat([out_static, out_moving], dim=-1)
            elif self.extra_non_moving_capacity_type == 'separate_static_primitive':
                occ_static = static_pm_ret['occupancy']
                out_static = static_pm_ret['canonical_occupancy']
                diff_static = static_pm_ret['latent_quantize_diff']
                occ = torch.cat([occ_static, occ], dim=-1)
                out = torch.cat([out_static, out], dim=-1)
                shape_quantize_diff = shape_quantize_diff + diff_static

        ret = dict(occupancy=occ,
                   canonical_occupancy=out,
                   latent_quantize_diff=shape_quantize_diff,
                   quantize_indices=shape_quantize_indices)
        return ret


class DecoderAttentionInputAllBatchNormMultiply(nn.Module):
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
                 leaky=False,
                 shape_quantize_num=4,
                 shape_template_latent_dim=64,
                 disable_diff=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.depth = depth
        self.disable_diff = disable_diff
        self.shape_quantize_num = shape_quantize_num
        self.shape_template_latent_dim = shape_template_latent_dim

        self.dim = dim
        self.out_dim = out_dim
        self.c64 = int(hidden_size / self.out_dim)
        assert depth % 2 == 0

        self.const = ConstantInput(self.c64 * self.out_dim)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx + 1)))
            ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))
        self.fc_p = nn.Conv1d(dim, self.decoder_input_dim, 1)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                GroupWiseResnetBlockConv1d(self.out_dim,
                                           int(self.c64 / 2**idx),
                                           size_out=int(self.c64 /
                                                        2**(idx - 1)))
            ])

        self.decoder = nn.Sequential(*dec_layers)

        self.bn = nn.BatchNorm1d(self.c64 * self.out_dim)

        self.fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                self.out_dim,
                                1,
                                groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        diff_dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            diff_dec_layers.extend([
                GroupWiseCResnetBlockConv1d(self.c_dim,
                                            self.out_dim,
                                            int(self.c64 / 2**idx),
                                            size_out=int(self.c64 /
                                                         2**(idx - 1)))
            ])

        self.diff_decoders = nn.ModuleList(diff_dec_layers)

        self.diff_bn = CBatchNorm1d(self.c_dim, self.c64 * self.out_dim)

        self.diff_fc_out = nn.Conv1d(self.c64 * self.out_dim,
                                     self.out_dim,
                                     1,
                                     groups=self.out_dim)

        self.shape_attention_module = AttentionConstMemory(
            self.shape_template_latent_dim, self.out_dim,
            self.shape_quantize_num, self.c64)

    def forward(self, p, z, c, shape_template_latent=None):
        assert shape_template_latent is not None

        # B, c_dim, 1
        #const = self.const(p).unsqueeze(2)

        batch_size = p.size(0)
        shape_const = self.shape_attention_module(shape_template_latent).view(
            batch_size, self.out_dim * self.c64, 1)
        pm_wise_latent = self.encoder(shape_const)

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        # B, decoder_input_dim, P
        net = self.fc_p(p)
        # B, pm_num * decoder_input_dim, P
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, T)

        # B, pm_num * decoder_input_dim, P
        shared = net + pm_wise_latent
        net = self.decoder(shared)

        out = self.fc_out(self.actvn(self.bn(net)))

        # batch, points, occupancy=out_dim (primitive_num)
        out = out.reshape(batch_size, self.out_dim,
                          T).transpose(1, 2).contiguous()

        diff_net = shared
        for diff_decoder in self.diff_decoders:
            diff_net = diff_decoder(diff_net, c)

        diff_out = self.diff_fc_out(self.actvn(self.diff_bn(diff_net, c)))
        diff_out = diff_out.reshape(batch_size, self.out_dim,
                                    T).transpose(1, 2).contiguous()
        if self.disable_diff:
            occ = out
        else:
            occ = torch.sigmoid(out) * diff_out

        ret = dict(occupancy=occ, canonical_occupancy=out)
        return ret


class DecoderQuantizedInputAllGenericNormMultiply(nn.Module):
    ''' Decoder with batch normalization class.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''
    def __init__(
        self,
        dim=3,
        z_dim=128,
        c_dim=128,
        hidden_size=256,
        depth=5,
        out_dim=3,
        leaky=False,
        shape_quantize_num=4,
        shape_template_latent_dim=64,
        disable_diff=False,
        batch_norm_momentum=0.1,
        batch_norm_type='batch_norm',
        batch_norm_type_const=None,
        is_extra_non_moving_capacity=False,
        extra_non_moving_capacity_type='multiple_group_wise_channel',
        coord_scale=1,
        extra_non_moving_capacity_extra_prim_num=3,
        static_primitive_hidden_size=64,
        diff_hidden_size=None,
        diff_decoder_type='default',
        disable_learnable_quantized_latent_vectors=False,
        group_norm_groups_per_channel=1,
        disable_canonical=False,
        is_simple_constant_mode=False,
        is_input_point_with_constant_latent_vector=False,
        last_norm_type=None,
        init_methods={},
        add_noise_to_out=False,
        noise_to_out_scale=0.01,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.depth = depth
        self.coord_scale = coord_scale
        self.disable_diff = disable_diff
        self.disable_canonical = disable_canonical
        assert not (disable_canonical and disable_diff)
        self.is_extra_non_moving_capacity = is_extra_non_moving_capacity
        self.extra_non_moving_capacity_type = extra_non_moving_capacity_type
        assert self.extra_non_moving_capacity_type in [
            'multiple_group_wise_channel', 'separate_static_primitive'
        ]
        self.extra_non_moving_capacity_extra_prim_num = extra_non_moving_capacity_extra_prim_num
        self.disable_learnable_quantized_latent_vectors = disable_learnable_quantized_latent_vectors
        self.diff_decoder_type = diff_decoder_type
        self.group_norm_groups_per_channel = group_norm_groups_per_channel

        self.add_noise_to_out = add_noise_to_out
        self.noise_to_out_scale = noise_to_out_scale
        self.dim = dim
        if self.is_extra_non_moving_capacity:
            if self.extra_non_moving_capacity_type == 'multiple_group_wise_channel':
                self.out_dim = out_dim + self.extra_non_moving_capacity_extra_prim_num
            elif self.extra_non_moving_capacity_type == 'separate_static_primitive':
                self.out_dim = out_dim - 1
        else:
            self.out_dim = out_dim
        self.c64 = int(hidden_size / self.out_dim /
                       group_norm_groups_per_channel)
        assert depth % 2 == 0
        self.static_primitive_hidden_size = static_primitive_hidden_size
        if batch_norm_type_const is None:
            batch_norm_type_const = batch_norm_type
        self.is_input_point_with_constant_latent_vector = is_input_point_with_constant_latent_vector

        self.const = ConstantInput(self.c64 * self.out_dim)

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.extend([
                layers.GroupWiseResnetBlockConv1dGenericNorm(
                    self.out_dim,
                    int(self.c64 / 2**idx),
                    size_out=int(self.c64 / 2**(idx + 1)),
                    batch_norm_type=batch_norm_type_const,
                    group_norm_groups_per_channel=group_norm_groups_per_channel,
                    batch_norm_momentum=batch_norm_momentum)
            ])

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder_input_dim = int(self.c64 / 2**(idx + 1))

        if self.is_input_point_with_constant_latent_vector:
            self.fc_p = layers.NormDependentConv1d(
                dim,
                self.c64 * self.out_dim * group_norm_groups_per_channel,
                1,
                norm_type=batch_norm_type_const)
        else:
            self.fc_p = layers.NormDependentConv1d(
                dim,
                self.decoder_input_dim,
                1,
                norm_type=batch_norm_type_const)

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.extend([
                layers.GroupWiseResnetBlockConv1dGenericNorm(
                    self.out_dim,
                    int(self.c64 / 2**idx),
                    size_out=int(self.c64 / 2**(idx - 1)),
                    batch_norm_type=batch_norm_type_const,
                    group_norm_groups_per_channel=group_norm_groups_per_channel,
                    batch_norm_momentum=batch_norm_momentum)
            ])
        self.decoder = nn.Sequential(*dec_layers)

        self.bn = layers.GenericNorm1d(
            self.c64 * self.out_dim * group_norm_groups_per_channel,
            norm_type=(batch_norm_type_const
                       if last_norm_type is None else last_norm_type),
            groups=self.out_dim * group_norm_groups_per_channel,
            momentum=batch_norm_momentum)

        self.fc_out = nn.Conv1d(self.c64 * self.out_dim *
                                group_norm_groups_per_channel,
                                self.out_dim,
                                1,
                                groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        if diff_hidden_size is None:
            diff_c64 = self.c64
        else:
            diff_c64 = int(diff_hidden_size / self.out_dim /
                           group_norm_groups_per_channel)
        diff_dec_layers = []
        if diff_decoder_type == 'default':
            for iidx, idx in enumerate(range(int(depth / 2), 0, -1)):
                if iidx == 0:
                    in_c64 = self.c64
                else:
                    in_c64 = diff_c64

                diff_dec_layers.extend([
                    layers.GroupWiseResnetBlockConv1dGenericNorm(
                        self.out_dim,
                        int(in_c64 / 2**idx),
                        size_out=int(diff_c64 / 2**(idx - 1)),
                        batch_norm_type=batch_norm_type,
                        size_c=self.c_dim,
                        conditional=True,
                        group_norm_groups_per_channel=
                        group_norm_groups_per_channel,
                        batch_norm_momentum=batch_norm_momentum)
                ])
                """
                diff_dec_layers.extend([
                    GroupWiseCResnetBlockConv1d(
                        self.c_dim,
                        self.out_dim,
                        int(in_c64 / 2**idx),
                        size_out=int(diff_c64 / 2**(idx - 1)),
                        norm_method=batch_norm_type,
                        batch_norm_momentum=batch_norm_momentum)
                ])
                """
            self.diff_decoders = nn.ModuleList(diff_dec_layers)
            """
            self.diff_bn = CBatchNorm1d(
                self.c_dim,
                diff_c64 * self.out_dim,
                norm_method=batch_norm_type.replace('layer_norm',
                                                    'group_norm'),
                groups=self.out_dim,
                batch_norm_momentum=batch_norm_momentum)
            """
            self.diff_bn = layers.GenericNorm1d(
                diff_c64 * self.out_dim * group_norm_groups_per_channel,
                norm_type=(batch_norm_type
                           if last_norm_type is None else last_norm_type),
                groups=self.out_dim * group_norm_groups_per_channel,
                c_channel=self.c_dim,
                conditional=True,
                momentum=batch_norm_momentum)

            self.diff_fc_out = nn.Conv1d(diff_c64 * self.out_dim *
                                         group_norm_groups_per_channel,
                                         self.out_dim,
                                         1,
                                         groups=self.out_dim)
        elif diff_decoder_type == 'independent_onet':
            self.diff_fc_p = layers.NormDependentConv1d(
                dim,
                diff_c64 * self.out_dim * group_norm_groups_per_channel,
                1,
                norm_type=batch_norm_type)
            #self.diff_fc_p = nn.Conv1d(dim, diff_c64 * self.out_dim, 1)
            for iidx, idx in enumerate(range(depth)):
                """
                diff_dec_layers.extend([
                    GroupWiseCResnetBlockConv1d(
                        self.c_dim,
                        self.out_dim,
                        diff_c64,
                        norm_method=batch_norm_type,
                        batch_norm_momentum=batch_norm_momentum)
                ])
                """

                diff_dec_layers.extend([
                    layers.GroupWiseResnetBlockConv1dGenericNorm(
                        self.out_dim,
                        diff_c64,
                        batch_norm_type=batch_norm_type,
                        size_c=self.c_dim,
                        conditional=True,
                        group_norm_groups_per_channel=
                        group_norm_groups_per_channel,
                        momentum=batch_norm_momentum)
                ])
            self.diff_decoders = nn.ModuleList(diff_dec_layers)
            """
            self.diff_bn = CBatchNorm1d(
                self.c_dim,
                diff_c64 * self.out_dim,
                norm_method=batch_norm_type.replace('layer_norm',
                                                    'group_norm'),
                groups=self.out_dim,
                batch_norm_momentum=batch_norm_momentum)
            """
            self.diff_bn = layers.GenericNorm1d(
                diff_c64 * self.out_dim * group_norm_groups_per_channel,
                norm_type=batch_norm_type,
                groups=self.out_dim * group_norm_groups_per_channel,
                c_channel=self.c_dim,
                conditional=True,
                momentum=batch_norm_momentum)

            self.diff_fc_out = nn.Conv1d(diff_c64 * self.out_dim *
                                         group_norm_groups_per_channel,
                                         self.out_dim,
                                         1,
                                         groups=self.out_dim)
        else:
            raise NotImplementedError

        self.shape_const_quantizer = Quantize(
            self.c64 * self.out_dim * group_norm_groups_per_channel,
            shape_quantize_num,
            is_simple_constant_mode=is_simple_constant_mode,
            disable_learnable_quantized_latent_vectors=
            disable_learnable_quantized_latent_vectors)
        self.fc_shape_c = layers.NormDependentConv1d(
            shape_template_latent_dim,
            self.c64 * self.out_dim * group_norm_groups_per_channel,
            1,
            norm_type=batch_norm_type_const)
        """
        if batch_norm_type == 'batch_norm':
            self.bn_shape_c = nn.BatchNorm1d(shape_template_latent_dim,
                                             momentum=batch_norm_momentum)
        elif batch_norm_type == 'layer_norm':
            self.bn_shape_c = nn.LayerNorm([shape_template_latent_dim, 1])
        elif batch_norm_type == 'no_norm':
            self.bn_shape_c = nn.Identity()
        """
        self.bn_shape_c = layers.GenericNorm1d(shape_template_latent_dim,
                                               momentum=batch_norm_momentum,
                                               norm_type=batch_norm_type_const,
                                               groups=1)

        if self.is_extra_non_moving_capacity:
            if self.extra_non_moving_capacity_type == 'separate_static_primitive':
                self.static_pm_model = DecoderQuantizedInputAllGenericNormMultiply(
                    dim=dim,
                    z_dim=z_dim,
                    c_dim=c_dim,
                    hidden_size=self.static_primitive_hidden_size,
                    depth=depth,
                    out_dim=1,
                    leaky=leaky,
                    shape_quantize_num=shape_quantize_num,
                    shape_template_latent_dim=shape_template_latent_dim,
                    disable_diff=disable_diff,
                    batch_norm_momentum=batch_norm_momentum,
                    is_input_point_with_constant_latent_vector=
                    is_input_point_with_constant_latent_vector,
                    batch_norm_type=batch_norm_type,
                    batch_norm_type_const=batch_norm_type_const,
                    coord_scale=coord_scale,
                    is_simple_constant_mode=is_simple_constant_mode,
                    disable_canonical=disable_canonical,
                    diff_hidden_size=diff_hidden_size,
                    is_extra_non_moving_capacity=False,
                    diff_decoder_type=diff_decoder_type,
                    disable_learnable_quantized_latent_vectors=
                    disable_learnable_quantized_latent_vectors,
                    group_norm_groups_per_channel=1,
                    last_norm_type=last_norm_type,
                    init_methods={},
                    add_noise_to_out=False,
                    noise_to_out_scale=1,
                )

        if init_methods.get('canonical_out', 'none') == 'zero':
            self.fc_out.weight.data.fill_(0.0)
            self.fc_out.bias.data.fill_(0.0)
        elif init_methods.get('canonical_out', 'none') == 'uniform_01':
            self.fc_out.weight.data.uniform_(-0.1, 0.1)
            self.fc_out.bias.data.uniform_(-0.1, 0.1)

        if init_methods.get('diff_out', 'none') == 'zero':
            self.diff_fc_out.weight.data.fill_(0.0)
            self.diff_fc_out.bias.data.fill_(0.0)
        elif init_methods.get('diff_out', 'none') == 'uniform_01':
            self.diff_fc_out.weight.data.uniform_(-0.1, 0.1)
            self.diff_fc_out.bias.data.uniform_(-0.1, 0.1)

    def forward(self, p, z, c, shape_template_latent=None):
        assert shape_template_latent is not None
        if self.is_extra_non_moving_capacity:
            if self.extra_non_moving_capacity_type == 'separate_static_primitive':
                static_pm_ret = self.static_pm_model(p, z, c,
                                                     shape_template_latent)
        p = p * self.coord_scale
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        # B, c_dim, 1
        #const = self.const(p).unsqueeze(2)

        shape_c = self.fc_shape_c(
            self.actvn(self.bn_shape_c(
                shape_template_latent.unsqueeze(2)))).squeeze(-1)
        shape_const, shape_quantize_diff, shape_quantize_indices = self.shape_const_quantizer(
            shape_c)

        # B, pm_num * decoder_input_dim, 1
        shape_const = shape_const.unsqueeze(2)

        if self.is_input_point_with_constant_latent_vector:
            shape_const_p = shape_const + self.fc_p(p)
            shared = self.encoder(shape_const_p)
        else:
            pm_wise_latent = self.encoder(shape_const)
            # B, decoder_input_dim, P
            net = self.fc_p(p)
            # B, pm_num * decoder_input_dim, P
            net = net.unsqueeze(1).expand(
                -1, self.out_dim * self.group_norm_groups_per_channel, -1,
                -1).reshape(batch_size, -1, T)

            # B, pm_num * decoder_input_dim, P
            shared = net + pm_wise_latent
        net = self.decoder(shared)

        out = self.fc_out(self.actvn(self.bn(net)))

        # batch, points, occupancy=out_dim (primitive_num)
        out = out.reshape(batch_size, self.out_dim,
                          T).transpose(1, 2).contiguous()

        if self.diff_decoder_type == 'default':
            diff_net = shared
            for diff_decoder in self.diff_decoders:
                diff_net = diff_decoder(diff_net, c)

            diff_out = self.diff_fc_out(self.actvn(self.diff_bn(diff_net, c)))
            diff_out = diff_out.reshape(batch_size, self.out_dim,
                                        T).transpose(1, 2).contiguous()
        elif self.diff_decoder_type == 'independent_onet':
            diff_net = self.diff_fc_p(p)
            for diff_decoder in self.diff_decoders:
                diff_net = diff_decoder(diff_net, c)

            diff_out = self.diff_fc_out(self.actvn(self.diff_bn(diff_net, c)))
            diff_out = diff_out.reshape(batch_size, self.out_dim,
                                        T).transpose(1, 2).contiguous()
        else:
            raise NotImplementedError
        if self.disable_diff:
            occ = out
        elif self.disable_canonical:
            occ = diff_out
        else:
            occ = torch.sigmoid(out) * diff_out

        if self.is_extra_non_moving_capacity:
            if self.extra_non_moving_capacity_type == 'multiple_group_wise_channel':
                occ_static = occ[:, :, :(
                    1 + self.extra_non_moving_capacity_extra_prim_num)].max(
                        -1, keepdim=True)[0]
                occ_moving = occ[:, :, (
                    1 + self.extra_non_moving_capacity_extra_prim_num):]
                occ = torch.cat([occ_static, occ_moving], dim=-1)

                out_static = out[:, :, :(
                    1 + self.extra_non_moving_capacity_extra_prim_num)].max(
                        -1, keepdim=True)[0]
                out_moving = out[:, :, (
                    1 + self.extra_non_moving_capacity_extra_prim_num):]
                out = torch.cat([out_static, out_moving], dim=-1)
            elif self.extra_non_moving_capacity_type == 'separate_static_primitive':
                occ_static = static_pm_ret['occupancy']
                out_static = static_pm_ret['canonical_occupancy']
                diff_static = static_pm_ret['latent_quantize_diff']
                occ = torch.cat([occ_static, occ], dim=-1)
                out = torch.cat([out_static, out], dim=-1)
                shape_quantize_diff = shape_quantize_diff + diff_static

        if self.add_noise_to_out:
            occ = occ + torch.randn_like(occ) * self.noise_to_out_scale

        ret = dict(occupancy=occ,
                   canonical_occupancy=out,
                   latent_quantize_diff=shape_quantize_diff,
                   quantize_indices=shape_quantize_indices)
        return ret


class DecoderQuantizedInputAllSirenMultiply(nn.Module):
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
                 leaky=False,
                 shape_quantize_num=4,
                 shape_template_latent_dim=64,
                 is_simple_constant_mode=False,
                 disable_diff=False,
                 disable_learnable_quantized_latent_vectors=False,
                 diff_hidden_size=None,
                 diff_depth=None,
                 init_methods={},
                 add_noise_to_out=False,
                 noise_to_out_scale=1,
                 disable_canonical=False,
                 diff_canonical_add=False,
                 diff_canonical_add_weight=1.0,
                 **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.depth = depth
        self.dim = dim
        self.out_dim = out_dim
        self.disable_diff = disable_diff
        self.c64 = int(hidden_size / self.out_dim)
        self.disable_canonical = disable_canonical
        self.diff_canonical_add = diff_canonical_add
        self.diff_canonical_add_weight = diff_canonical_add_weight
        assert depth % 2 == 0

        self.add_noise_to_out = add_noise_to_out
        self.noise_to_out_scale = noise_to_out_scale
        self.const_latent_layers = nn.ModuleList([
            nn.LeakyReLU(0.2),
            nn.Conv1d(shape_template_latent_dim, self.c64 * self.out_dim, 1),
            Quantize(self.c64 * self.out_dim,
                     shape_quantize_num,
                     is_simple_constant_mode=is_simple_constant_mode,
                     disable_learnable_quantized_latent_vectors=
                     disable_learnable_quantized_latent_vectors),
            layers.SirenConv1d(self.c64 * self.out_dim,
                               self.c64 * self.out_dim,
                               groups=self.out_dim,
                               input_kind='latent'),
            layers.SirenConv1d(self.c64 * self.out_dim,
                               self.c64 * self.out_dim,
                               groups=self.out_dim)
        ])

        self.fc_p = layers.SirenConv1d(dim,
                                       self.c64 * self.out_dim,
                                       input_kind='coord')

        enc_layers = []
        for idx in range(int(depth / 2)):
            enc_layers.append(
                layers.SirenConv1d(int(self.c64 / 2**idx) * self.out_dim,
                                   int(self.c64 / 2**(idx + 1)) * self.out_dim,
                                   groups=self.out_dim))

        self.encoder = nn.Sequential(*enc_layers)
        encoder_output_dim = int(self.c64 / 2**(idx + 1)) * self.out_dim

        dec_layers = []
        for idx in range(int(depth / 2), 0, -1):
            dec_layers.append(
                layers.SirenConv1d(int(self.c64 / 2**idx) * self.out_dim,
                                   int(self.c64 / 2**(idx - 1)) * self.out_dim,
                                   groups=self.out_dim))
        self.decoder = nn.Sequential(*dec_layers)

        self.fc_out = layers.SirenConv1d(self.c64 * self.out_dim,
                                         self.out_dim,
                                         groups=self.out_dim,
                                         is_outermost=True)
        if init_methods.get('canonical_out', 'none') == 'zero':
            self.fc_out.conv.weight.data.fill_(0.0)
            self.fc_out.conv.bias.data.fill_(0.0)
        elif init_methods.get('canonical_out', 'none') == 'uniform_01':
            self.fc_out.conv.weight.data.uniform_(-0.1, 0.1)
            self.fc_out.conv.bias.data.uniform_(-0.1, 0.1)

        self.latent = nn.Sequential(
            layers.SirenConv1d(self.c_dim,
                               encoder_output_dim,
                               input_kind='latent'),
            layers.SirenConv1d(encoder_output_dim, encoder_output_dim))

        if diff_hidden_size is None:
            diff_c64 = self.c64
        else:
            diff_c64 = int(diff_hidden_size / self.out_dim)

        diff_depth = depth if diff_depth is None else diff_depth

        diff_dec_layers = []
        for iidx, idx in enumerate(range(int(diff_depth / 2), 0, -1)):
            if iidx == 0:
                in_dim = int(self.c64 / 2**(int(depth / 2))) * self.out_dim
            else:
                in_dim = int(diff_c64 / 2**idx) * self.out_dim

            diff_dec_layers.append(
                layers.SirenConv1d(in_dim,
                                   int(diff_c64 / 2**(idx - 1)) * self.out_dim,
                                   groups=self.out_dim))

        self.diff_decoder = nn.Sequential(*diff_dec_layers)

        self.diff_fc_out = layers.SirenConv1d(diff_c64 * self.out_dim,
                                              self.out_dim,
                                              groups=self.out_dim,
                                              is_outermost=True)

        if init_methods.get('diff_out', 'none') == 'zero':
            self.diff_fc_out.conv.weight.data.fill_(0.0)
            self.diff_fc_out.conv.bias.data.fill_(0.0)
        elif init_methods.get('diff_out', 'none') == 'uniform_01':
            self.diff_fc_out.conv.weight.data.uniform_(-0.1, 0.1)
            self.diff_fc_out.conv.bias.data.uniform_(-0.1, 0.1)

    def forward(self, p, z, c, shape_template_latent=None):
        assert shape_template_latent is not None
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        # B, c_dim, 1
        #const = self.const(p).unsqueeze(2)

        x = shape_template_latent
        for idx, layer in enumerate(self.const_latent_layers):
            if idx == 1:
                x = layer(x.unsqueeze(-1))
            elif idx == 2:
                x, shape_quantize_diff, shape_quantize_indices = layer(
                    x.squeeze(-1))
                x = x.unsqueeze(-1)
            elif idx == len(self.const_latent_layers) - 1:
                shape_const = layer(x)
            else:
                x = layer(x)

        shape_const_p = shape_const + self.fc_p(p)
        shared = self.encoder(shape_const_p)
        net = self.decoder(shared)

        out = self.fc_out(net)

        # batch, points, occupancy=out_dim (primitive_num)
        out = out.reshape(batch_size, self.out_dim,
                          T).transpose(1, 2).contiguous()

        latent_vector = self.latent(c.unsqueeze(-1))
        diff_net = self.diff_decoder(shared + latent_vector)

        diff_out = self.diff_fc_out(diff_net)
        diff_out = diff_out.reshape(batch_size, self.out_dim,
                                    T).transpose(1, 2).contiguous()

        if self.disable_diff:
            occ = out
        elif self.disable_canonical:
            occ = diff_out
        elif self.diff_canonical_add:
            occ = out * self.diff_canonical_add_weight + diff_out
        else:
            occ = torch.sigmoid(out) * diff_out

        if self.add_noise_to_out:
            occ = occ + torch.randn_like(occ) * self.noise_to_out_scale

        ret = dict(occupancy=occ,
                   canonical_occupancy=out,
                   latent_quantize_diff=shape_quantize_diff,
                   quantize_indices=shape_quantize_indices)
        return ret
