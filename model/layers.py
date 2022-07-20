import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''
    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
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


class CResnetBlockConv1d(nn.Module):
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
                 size_in,
                 size_h=None,
                 size_out=None,
                 norm_method='batch_norm',
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
        # Submodules
        if not legacy:
            self.bn_0 = CBatchNorm1d(c_dim, size_in, norm_method=norm_method)
            self.bn_1 = CBatchNorm1d(c_dim, size_h, norm_method=norm_method)
        else:
            self.bn_0 = CBatchNorm1d_legacy(c_dim,
                                            size_in,
                                            norm_method=norm_method)
            self.bn_1 = CBatchNorm1d_legacy(c_dim,
                                            size_h,
                                            norm_method=norm_method)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)
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


class ResnetBlockConv1d(nn.Module):
    ''' 1D-Convolutional ResNet block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''
    def __init__(self, size_in, size_h=None, size_out=None):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.bn_0 = nn.BatchNorm1d(size_in)
        self.bn_1 = nn.BatchNorm1d(size_h)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)

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


class ResnetBlockConv2d(nn.Module):
    ''' 1D-Convolutional ResNet block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''
    def __init__(self, size_in, size_h=None, size_out=None):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.bn_0 = nn.BatchNorm2d(size_in)
        self.bn_1 = nn.BatchNorm2d(size_h)

        self.fc_0 = nn.Conv2d(size_in, size_h, 1)
        self.fc_1 = nn.Conv2d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv2d(size_in, size_out, 1, bias=False)

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


# Utility modules
class AffineLayer(nn.Module):
    ''' Affine layer class.

    Args:
        c_dim (tensor): dimension of latent conditioned code c
        dim (int): input dimension
    '''
    def __init__(self, c_dim, dim=3):
        super().__init__()
        self.c_dim = c_dim
        self.dim = dim
        # Submodules
        self.fc_A = nn.Linear(c_dim, dim * dim)
        self.fc_b = nn.Linear(c_dim, dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.fc_A.weight)
        nn.init.zeros_(self.fc_b.weight)
        with torch.no_grad():
            self.fc_A.bias.copy_(torch.eye(3).view(-1))
            self.fc_b.bias.copy_(torch.tensor([0., 0., 2.]))

    def forward(self, x, p):
        assert (x.size(0) == p.size(0))
        assert (p.size(2) == self.dim)
        batch_size = x.size(0)
        A = self.fc_A(x).view(batch_size, 3, 3)
        b = self.fc_b(x).view(batch_size, 1, 3)
        out = p @ A + b
        return out


class CBatchNorm1d(nn.Module):
    ''' Conditional batch normalization layer class.

    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    '''
    def __init__(self,
                 c_dim,
                 f_dim,
                 norm_method='batch_norm',
                 groups=None,
                 batch_norm_momentum=0.1):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        # Submodules
        self.conv_gamma = nn.Conv1d(c_dim, f_dim, 1)
        self.conv_beta = nn.Conv1d(c_dim, f_dim, 1)
        if norm_method == 'batch_norm':
            self.bn = nn.BatchNorm1d(f_dim,
                                     affine=False,
                                     momentum=batch_norm_momentum)
        elif norm_method == 'instance_norm':
            self.bn = nn.InstanceNorm1d(f_dim, affine=False)
        elif norm_method == 'group_norm':
            self.bn = nn.GroupNorm(groups, f_dim, affine=False)
        elif norm_method == 'no_norm':
            self.bn = nn.Identity()
        else:
            raise ValueError('Invalid normalization method!')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x, c):
        assert (x.size(0) == c.size(0))
        assert (c.size(1) == self.c_dim)

        # c is assumed to be of size batch_size x c_dim x T
        if len(c.size()) == 2:
            c = c.unsqueeze(2)

        # Affine mapping
        gamma = self.conv_gamma(c)
        beta = self.conv_beta(c)

        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out


class CBatchNorm1d_legacy(nn.Module):
    ''' Conditional batch normalization legacy layer class.

    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    '''
    def __init__(self, c_dim, f_dim, norm_method='batch_norm'):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        # Submodules
        self.fc_gamma = nn.Linear(c_dim, f_dim)
        self.fc_beta = nn.Linear(c_dim, f_dim)
        if norm_method == 'batch_norm':
            self.bn = nn.BatchNorm1d(f_dim, affine=False)
        elif norm_method == 'instance_norm':
            self.bn = nn.InstanceNorm1d(f_dim, affine=False)
        elif norm_method == 'group_norm':
            self.bn = nn.GroupNorm1d(f_dim, affine=False)
        else:
            raise ValueError('Invalid normalization method!')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.fc_gamma.weight)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.ones_(self.fc_gamma.bias)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, x, c):
        batch_size = x.size(0)
        # Affine mapping
        gamma = self.fc_gamma(c)
        beta = self.fc_beta(c)
        gamma = gamma.view(batch_size, self.f_dim, 1)
        beta = beta.view(batch_size, self.f_dim, 1)
        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out


class GroupWiseResnetBlockConv1d(nn.Module):
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
        self.bn_0 = nn.BatchNorm1d(size_in * self.n_primitives)
        self.bn_1 = nn.BatchNorm1d(size_h * self.n_primitives)

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


class ResnetBlockConv1dWoBN(nn.Module):
    ''' 1D-Convolutional ResNet block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''
    def __init__(self, size_in, size_h=None, size_out=None):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)

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


class ConstantInput(nn.Module):
    def __init__(self, channel, init_method='none'):
        super().__init__()

        if init_method == 'zero':
            self.input = nn.Parameter(torch.zeros(1, channel))
        elif init_method == 'uniform_01':
            self.input = nn.Parameter((torch.rand(1, channel) * 2 - 1) / 10)
        elif init_method == 'normal_01':
            self.input = nn.Parameter(torch.randn(1, channel) * 0.1)

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.expand(batch, -1).to(input.device)

        return out


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


class ResnetBlockConv1dLayerNorm(nn.Module):
    ''' 1D-Convolutional ResNet block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''
    def __init__(self, size_in, size_h=None, size_out=None):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.bn_0 = nn.LayerNorm([size_in, 1])
        self.bn_1 = nn.LayerNorm([size_h, 1])

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)

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


class GroupWiseResnetBlockConv1dGenericNorm(nn.Module):
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
                 size_c=None,
                 batch_norm_momentum=0.1,
                 batch_norm_type='batch_norm',
                 group_norm_groups_per_channel=1,
                 conditional=False):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in
        self.n_primitives = n_primitives
        self.size_c = size_c

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        self.batch_norm_type = batch_norm_type
        self.conditional = conditional
        if self.size_c is not None:
            assert self.conditional
        # Submodules

        self.bn_0 = GenericNorm1d(
            size_in * self.n_primitives * group_norm_groups_per_channel,
            groups=self.n_primitives * group_norm_groups_per_channel,
            norm_type=self.batch_norm_type,
            momentum=batch_norm_momentum,
            c_channel=size_c,
            conditional=self.conditional)
        self.bn_1 = GenericNorm1d(
            size_h * self.n_primitives * group_norm_groups_per_channel,
            groups=self.n_primitives * group_norm_groups_per_channel,
            norm_type=self.batch_norm_type,
            momentum=batch_norm_momentum,
            c_channel=size_c,
            conditional=self.conditional)

        self.fc_0 = NormDependentConv1d(
            size_in * self.n_primitives * group_norm_groups_per_channel,
            size_h * self.n_primitives * group_norm_groups_per_channel,
            kernel_size=1,
            groups=self.n_primitives,
            norm_type=self.batch_norm_type)

        self.fc_1 = NormDependentConv1d(
            size_h * self.n_primitives * group_norm_groups_per_channel,
            size_out * self.n_primitives * group_norm_groups_per_channel,
            kernel_size=1,
            groups=self.n_primitives,
            norm_type=self.batch_norm_type)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(
                size_in * self.n_primitives * group_norm_groups_per_channel,
                size_out * self.n_primitives * group_norm_groups_per_channel,
                kernel_size=1,
                bias=False,
                groups=self.n_primitives)

        # Initialization
        nn.init.zeros_(self.fc_1.conv.weight)

    def forward(self, x, c=None):
        if self.conditional:
            net = self.fc_0(self.actvn(self.bn_0(x, c)))
            dx = self.fc_1(self.actvn(self.bn_1(net, c)))
        else:
            net = self.fc_0(self.actvn(self.bn_0(x)))
            dx = self.fc_1(self.actvn(self.bn_1(net)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class Conv1dWS(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(Conv1dWS, self).__init__(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                            keepdim=True)
        weight = weight - weight_mean
        std = torch.sqrt(
            torch.var(weight.view(weight.size(0), -1), dim=-1) + 1e-12).view(
                -1, 1, 1) + 1e-5
        #std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv1d(x, weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class GenericNorm1d(nn.Module):
    def __init__(self,
                 in_channel,
                 norm_type='batch_norm',
                 groups=1,
                 momentum=0.1,
                 conditional=False,
                 c_channel=None):
        super().__init__()
        self.conditional = conditional
        self.c_channel = c_channel
        if self.c_channel is not None:
            assert self.conditional
        self.norm_type = norm_type
        if self.norm_type == 'batch_norm':
            self.bn = nn.BatchNorm1d(in_channel,
                                     momentum=momentum,
                                     affine=(not self.conditional))
        elif self.norm_type == 'batch_renorm':
            self.bn = BatchRenorm1d(in_channel,
                                    momentum=momentum,
                                    affine=(not self.conditional))
        elif self.norm_type in ['layer_norm', 'layer_norm_ws']:
            self.bn = nn.GroupNorm(groups,
                                   in_channel,
                                   affine=(not self.conditional))
        elif self.norm_type in ['instance_norm', 'instance_norm_no_mean']:
            self.bn = nn.InstanceNorm1d(in_channel, momentum=momentum)
        elif self.norm_type in ['no_norm', 'weight_norm', 'spectral_norm']:
            self.bn = nn.Identity()
        else:
            raise ValueError('Invalid norm type {}'.format(norm_type))

        if self.conditional:
            self.conv_gamma = nn.Conv1d(self.c_channel, in_channel, 1)
            self.conv_beta = nn.Conv1d(self.c_channel, in_channel, 1)
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x, c=None):
        if self.conditional:
            assert (x.size(0) == c.size(0))
            assert (c.size(1) == self.c_channel)

            # c is assumed to be of size batch_size x c_dim x T
            if len(c.size()) == 2:
                c = c.unsqueeze(2)

            # Affine mapping
            gamma = self.conv_gamma(c)
            beta = self.conv_beta(c)

            # Batchnorm
            if self.norm_type == 'instance_norm_no_mean':
                mean = torch.mean(x, dim=-1, keepdim=True)
            net = self.bn(x)
            if self.norm_type == 'instance_norm_no_mean':
                net = net + mean
            out = gamma * net + beta
            return out

        else:
            if self.norm_type == 'instance_norm_no_mean':
                mean = torch.mean(x, dim=-1, keepdim=True)
            net = self.bn(x)
            if self.norm_type == 'instance_norm_no_mean':
                net = net + mean
            return net


class NormDependentConv1d(nn.Module):
    def __init__(self, *args, norm_type='batch_norm', **kwargs):
        super().__init__()
        self.norm_type = norm_type
        if self.norm_type == 'layer_norm_ws':
            conv_cls = Conv1dWS
        else:
            conv_cls = nn.Conv1d
        self.conv = conv_cls(*args, **kwargs)

        if self.norm_type == 'weight_norm':
            self.conv = nn.utils.weight_norm(self.conv)
        elif self.norm_type == 'spectral_norm':
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, x):
        return self.conv(x)


class InputDependentConvGenericNormRelu(nn.Module):
    def __init__(self,
                 *args,
                 norm_type='batch_norm',
                 momentum=0.1,
                 groups=1,
                 conditional=False,
                 c_channel=None,
                 **kwargs):
        super().__init__()
        self.conditional = conditional
        self.bn = GenericNorm1d(args[0],
                                norm_type=norm_type,
                                momentum=momentum,
                                groups=groups,
                                conditional=conditional,
                                c_channel=c_channel)
        self.conv = NormDependentConv1d(*args, norm_type=norm_type, **kwargs)
        self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, x, c=None):
        if self.conditional:
            return self.actvn(self.bn(self.conv(x), c))
        else:
            return self.actvn(self.bn(self.conv(x)))


class RBFLayer(nn.Module):
    '''Transforms incoming data using a given radial basis function.
        - Input: (1, N, in_features) where N is an arbitrary batch size
        - Output: (1, N, out_features) where N is an arbitrary batch size'''
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigmas = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

        self.freq = nn.Parameter(np.pi * torch.ones((1, self.out_features)))

    def reset_parameters(self):
        nn.init.uniform_(self.centres, -1, 1)
        nn.init.constant_(self.sigmas, 10)

    def forward(self, input):
        input = input[0, ...]
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1) * self.sigmas.unsqueeze(0)
        return self.gaussian(distances).unsqueeze(0)

    def gaussian(self, alpha):
        phi = torch.exp(-1 * alpha.pow(2))
        return phi


class MetaModule(nn.Module):
    """
    Base class for PyTorch meta-learning modules. These modules accept an
    additional argument `params` in their `forward` method.
    Notes
    -----
    Objects inherited from `MetaModule` are fully compatible with PyTorch
    modules from `torch.nn.Module`. The argument `params` is a dictionary of
    tensors, with full support of the computation graph (for differentiation).
    """
    def meta_named_parameters(self, prefix='', recurse=True):
        gen = self._named_members(lambda module: module._parameters.items()
                                  if isinstance(module, MetaModule) else [],
                                  prefix=prefix,
                                  recurse=recurse)
        for elem in gen:
            yield elem

    def meta_parameters(self, recurse=True):
        for name, param in self.meta_named_parameters(recurse=recurse):
            yield param


class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(
            weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output


class MetaSequential(nn.Sequential, MetaModule):
    __doc__ = nn.Sequential.__doc__

    def forward(self, input, params=None):
        for name, module in self._modules.items():
            if isinstance(module, MetaModule):
                input = module(input, params=get_subdict(params, name))
            elif isinstance(module, nn.Module):
                input = module(input)
            else:
                raise TypeError(
                    'The module must be either a torch module '
                    '(inheriting from `nn.Module`), or a `MetaModule`. '
                    'Got type: `{0}`'.format(type(module)))
        return input


def get_subdict(dictionary, key=None):
    if dictionary is None:
        return None
    if (key is None) or (key == ''):
        return dictionary
    key_re = re.compile(r'^{0}\.(.+)'.format(re.escape(key)))
    return OrderedDict((key_re.sub(r'\1', k), value)
                       for (k, value) in dictionary.items()
                       if key_re.match(k) is not None)


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30,
                              np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class MetaModule(nn.Module):
    """
    Base class for PyTorch meta-learning modules. These modules accept an
    additional argument `params` in their `forward` method.
    Notes
    -----
    Objects inherited from `MetaModule` are fully compatible with PyTorch
    modules from `torch.nn.Module`. The argument `params` is a dictionary of
    tensors, with full support of the computation graph (for differentiation).
    """
    def meta_named_parameters(self, prefix='', recurse=True):
        gen = self._named_members(lambda module: module._parameters.items()
                                  if isinstance(module, MetaModule) else [],
                                  prefix=prefix,
                                  recurse=recurse)
        for elem in gen:
            yield elem

    def meta_parameters(self, recurse=True):
        for name, param in self.meta_named_parameters(recurse=recurse):
            yield param


def hyper_weight_init(m, in_features_main_net):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight,
                                a=0.0,
                                nonlinearity='relu',
                                mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias'):
        with torch.no_grad():
            m.bias.uniform_(-1 / in_features_main_net,
                            1 / in_features_main_net)


def hyper_bias_init(m):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight,
                                a=0.0,
                                nonlinearity='relu',
                                mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias'):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        with torch.no_grad():
            m.bias.uniform_(-1 / fan_in, 1 / fan_in)


class SirenConv1d(nn.Module):
    def __init__(self,
                 inc,
                 outc,
                 groups=1,
                 is_outermost=False,
                 input_kind='siren'):
        super().__init__()
        assert input_kind in ['siren', 'latent', 'coord']

        self.conv = nn.Conv1d(inc, outc, 1, groups=groups)

        if input_kind == 'siren':
            self.conv.apply(sine_init)
            self.act = Sine()
        elif input_kind == 'coord':
            self.conv.apply(first_layer_sine_init)
            self.act = Sine()
        elif input_kind == 'latent':
            self.conv.apply(
                lambda m: hyper_weight_init(m,
                                            m.weight.data.size()[1]))
            self.act = nn.Identity()

        if is_outermost:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))


class SingleBVPNet(MetaModule):
    '''A canonical representation network for a BVP.'''
    def __init__(self,
                 out_features=1,
                 type='sine',
                 in_features=2,
                 mode='mlp',
                 hidden_features=256,
                 num_hidden_layers=3,
                 **kwargs):
        super().__init__()
        self.mode = mode

        assert self.mode == 'mlp'
        assert self.type == 'sin'

        self.net = FCBlock(in_features=in_features,
                           out_features=out_features,
                           num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features,
                           outermost_linear=True,
                           nonlinearity=type)

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input['coords'].clone().detach().requires_grad_(
            True)
        coords = coords_org

        output = self.net(coords, get_subdict(params, 'net'))
        return {'model_in': coords_org, 'model_out': output}

    def forward_with_activations(self, model_input):
        '''Returns not only model output, but also intermediate activations.'''
        coords = model_input['coords'].clone().detach().requires_grad_(True)
        activations = self.net.forward_with_activations(coords)
        return {
            'model_in': coords,
            'model_out': activations.popitem(),
            'activations': activations
        }


class FCBlock(MetaModule):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''
    def __init__(self,
                 in_features,
                 out_features,
                 num_hidden_layers,
                 hidden_features,
                 outermost_linear=False,
                 nonlinearity='relu',
                 weight_init=None):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {
            'sine': (Sine(), sine_init, first_layer_sine_init),
        }

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(
            MetaSequential(BatchLinear(in_features, hidden_features), nl))

        for i in range(num_hidden_layers):
            self.net.append(
                MetaSequential(BatchLinear(hidden_features, hidden_features),
                               nl))

        if outermost_linear:
            self.net.append(
                MetaSequential(BatchLinear(hidden_features, out_features)))
        else:
            self.net.append(
                MetaSequential(BatchLinear(hidden_features, out_features), nl))

        self.net = MetaSequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None:  # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(coords, params=get_subdict(params, 'net'))
        return output

    def forward_with_activations(self, coords, params=None, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.'''
        if params is None:
            params = OrderedDict(self.named_parameters())

        activations = OrderedDict()

        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            subdict = get_subdict(params, 'net.%d' % i)
            for j, sublayer in enumerate(layer):
                if isinstance(sublayer, BatchLinear):
                    x = sublayer(x, params=get_subdict(subdict, '%d' % j))
                else:
                    x = sublayer(x)

                if retain_grad:
                    x.retain_grad()
                activations['_'.join((str(sublayer.__class__), "%d" % i))] = x
        return activations


class BatchRenorm(torch.jit.ScriptModule):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-3,
        momentum: float = 0.01,
        affine: bool = True,
    ):
        super().__init__()
        self.register_buffer("running_mean",
                             torch.zeros(num_features, dtype=torch.float))
        self.register_buffer("running_std",
                             torch.ones(num_features, dtype=torch.float))
        self.register_buffer("num_batches_tracked",
                             torch.tensor(0, dtype=torch.long))
        self.weight = torch.nn.Parameter(
            torch.ones(num_features, dtype=torch.float))
        self.bias = torch.nn.Parameter(
            torch.zeros(num_features, dtype=torch.float))
        self.affine = affine
        self.eps = eps
        self.step = 0
        self.momentum = momentum

    def _check_input_dim(self, x: torch.Tensor) -> None:
        raise NotImplementedError()  # pragma: no cover

    @property
    def rmax(self) -> torch.Tensor:
        return (2 / 35000 * self.num_batches_tracked + 25 / 35).clamp_(
            1.0, 3.0)

    @property
    def dmax(self) -> torch.Tensor:
        return (5 / 20000 * self.num_batches_tracked - 25 / 20).clamp_(
            0.0, 5.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)
        if x.dim() > 2:
            x = x.transpose(1, -1)
        if self.training:
            dims = [i for i in range(x.dim() - 1)]
            batch_mean = x.mean(dims)
            batch_std = torch.sqrt(x.var(dims, unbiased=False) + self.eps)
            r = (batch_std.detach() /
                 self.running_std.view_as(batch_std)).clamp_(
                     1 / self.rmax, self.rmax)
            d = (
                (batch_mean.detach() - self.running_mean.view_as(batch_mean)) /
                self.running_std.view_as(batch_std)).clamp_(
                    -self.dmax, self.dmax)
            x = (x - batch_mean) / batch_std * r + d
            self.running_mean += self.momentum * (batch_mean.detach() -
                                                  self.running_mean)
            self.running_std += self.momentum * (batch_std.detach() -
                                                 self.running_std)
            self.num_batches_tracked += 1
        else:
            x = (x - self.running_mean) / self.running_std
        if self.affine:
            x = self.weight * x + self.bias
        if x.dim() > 2:
            x = x.transpose(1, -1)
        return x


class BatchRenorm1d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() not in [2, 3]:
            raise ValueError("expected 2D or 3D input (got {x.dim()}D input)")


class BatchRenorm2d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError("expected 4D input (got {x.dim()}D input)")


class BatchRenorm3d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 5:
            raise ValueError("expected 5D input (got {x.dim()}D input)")
