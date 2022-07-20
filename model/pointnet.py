import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import ResnetBlockFC


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class OccNetSimplePointNet(nn.Module):
    ''' PointNet-based encoder network.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''
    def __init__(self,
                 c_dim=128,
                 dim=3,
                 hidden_dim=128,
                 depth=4,
                 learnable_attention=False,
                 attention_type='none',
                 attention_dim=750,
                 attention_reduction_type='max'):
        super().__init__()
        self.c_dim = c_dim
        self.out_dim = c_dim
        self.depth = depth

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.fcs = nn.ModuleList(
            [nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(self.depth)])
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p, **kwargs):
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)

        for idx, fc in enumerate(self.fcs):
            if idx == len(self.fcs) - 1:
                net = fc(self.actvn(net))
            else:
                net = fc(self.actvn(net))
                pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
                net = torch.cat([net, pooled], dim=2)

        # Recude to  B x F
        net = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(net))

        return c


class TreeGANPointNet(nn.Module):
    def __init__(self,
                 dim=3,
                 depth=5,
                 learnable_attention=False,
                 attention_type='none',
                 attention_dim=750,
                 attention_reduction_type='max',
                 c_dim=None,
                 hidden_size=None):
        super(TreeGANPointNet, self).__init__()
        self.layer_num = depth
        self.attention_type = attention_type
        assert attention_type in ['none', 'softmax', 'proper']
        self.attention_reduction_type = attention_reduction_type
        assert attention_reduction_type in ['max', 'sum']

        self.fc_layers = nn.ModuleList([])

        self.fc_layers.append(nn.Conv1d(dim, 64, kernel_size=1, stride=1))

        for inx in range(self.layer_num - 1):
            self.fc_layers.append(
                nn.Conv1d(64 * (2**inx),
                          64 * (2**(inx + 1)),
                          kernel_size=1,
                          stride=1))
        if c_dim is not None:
            self.out_dim = c_dim
        else:
            self.out_dim = 64 * (2**(self.layer_num - 1))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.learnable_attention = learnable_attention
        if self.learnable_attention:
            self.attention_dim = attention_dim
            self.att_layers = nn.ModuleList([])
            self.att_layers.append(
                nn.Linear(attention_dim, int(attention_dim // 2)))
            self.att_layers.append(
                nn.Linear(int(attention_dim // 2), attention_dim))
        if self.attention_type == 'proper':
            self.proper_attention_squash_layer = nn.Conv1d(self.out_dim,
                                                           1,
                                                           kernel_size=1,
                                                           stride=1)

    def forward(self, f, mask=None):
        # B, dim = 3 or 2, points
        feat = f.transpose(1, 2)
        vertex_num = feat.size(2)

        for layer in self.fc_layers:
            feat = layer(feat)
            feat = self.leaky_relu(feat)

        if mask is not None:
            assert mask.ndim == 2, mask.shape
            assert mask.size(1) == feat.size(-1)
            if self.learnable_attention:
                for l in self.att_layers:
                    mask = self.leaky_relu(l(mask.float()))
            mask = mask.unsqueeze(1)
            if self.attention_type in ['none', 'proper']:
                mask = mask
            elif self.attention_type == 'softmax':
                mask = torch.softmax(mask, dim=-1)
            else:
                raise NotImplementedError
            if self.attention_type == 'proper':
                queried = feat * mask
                queried = torch.softmax(queried, dim=-1)
                feat = self.leaky_relu(
                    self.proper_attention_squash_layer(feat)) * queried
            else:
                feat = feat * mask

            #feat = feat + mask
        if self.attention_reduction_type == 'max':
            out = F.max_pool1d(input=feat, kernel_size=vertex_num).squeeze(-1)
        elif self.attention_reduction_type == 'sum':
            out = feat.sum(-1)
        else:
            raise NotImplementedError
        return out


class ResnetPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''
    def __init__(self, c_dim=128, dim=3, hidden_dim=128, depth=5):
        super().__init__()
        self.c_dim = c_dim
        self.depth = depth

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.blocks = nn.ModuleList(
            [ResnetBlockFC(2 * hidden_dim, hidden_dim) for _ in range(depth)])
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        feats = {}
        for idx in range(len(self.blocks) - 1):
            net = self.blocks[idx](net)
            pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
            feats['feat_{}'.format(idx)] = pooled
            net = torch.cat([net, pooled], dim=2)

        net = self.blocks[-1](net)

        # Recude to  B x F
        net = self.pool(net, dim=1)
        feats['feat_{}'.format(self.depth - 1)] = net
        c = self.fc_c(self.actvn(net))

        feats['latent'] = c
        return feats


class TreeGANPointNetSNELU(nn.Module):
    def __init__(self,
                 dim=3,
                 depth=5,
                 learnable_attention=False,
                 attention_type='none',
                 attention_dim=750,
                 attention_reduction_type='max',
                 act_type='elu',
                 special_weight_init=False):
        super().__init__()
        self.layer_num = depth
        self.attention_type = attention_type
        assert attention_type in ['none', 'softmax', 'proper']
        self.attention_reduction_type = attention_reduction_type
        assert attention_reduction_type in ['max', 'sum']

        self.fc_layers = nn.ModuleList([])

        self.fc_layers.append(
            nn.utils.spectral_norm(nn.Conv1d(dim, 64, kernel_size=1,
                                             stride=1)))

        for inx in range(self.layer_num - 1):
            self.fc_layers.append(
                nn.utils.spectral_norm(
                    nn.Conv1d(64 * (2**inx),
                              64 * (2**(inx + 1)),
                              kernel_size=1,
                              stride=1)))

        if special_weight_init:
            for layer in self.fc_layers:
                layer.weight.data.normal_(0.0, 0.02)
        self.out_dim = 64 * (2**(self.layer_num - 1))
        if act_type == 'elu':
            self.elu = nn.ELU()
        elif act_type == 'selu':
            self.elu = nn.SELU()
        else:
            raise NotImplementedError

        self.learnable_attention = learnable_attention
        if self.learnable_attention:
            self.attention_dim = attention_dim
            self.att_layers = nn.ModuleList([])
            self.att_layers.append(
                nn.utils.spectral_norm(
                    nn.Linear(attention_dim, int(attention_dim // 2))))
            self.att_layers.append(
                nn.utils.spectral_norm(
                    nn.Linear(int(attention_dim // 2), attention_dim)))
        if self.attention_type == 'proper':
            self.proper_attention_squash_layer = nn.utils.spectral_norm(
                nn.Conv1d(self.out_dim, 1, kernel_size=1, stride=1))

    def forward(self, f, mask=None):
        # B, dim = 3 or 2, points
        feat = f.transpose(1, 2)
        vertex_num = feat.size(2)

        for layer in self.fc_layers:
            feat = layer(feat)
            feat = self.elu(feat)

        if mask is not None:
            assert mask.ndim == 2, mask.shape
            assert mask.size(1) == feat.size(-1)
            if self.learnable_attention:
                for l in self.att_layers:
                    mask = self.elu(l(mask.float()))
            mask = mask.unsqueeze(1)
            if self.attention_type in ['none', 'proper']:
                mask = mask
            elif self.attention_type == 'softmax':
                mask = torch.softmax(mask, dim=-1)
            else:
                raise NotImplementedError
            if self.attention_type == 'proper':
                queried = feat * mask
                queried = torch.softmax(queried, dim=-1)
                feat = self.elu(
                    self.proper_attention_squash_layer(feat)) * queried
            else:
                feat = feat * mask

            #feat = feat + mask
        if self.attention_reduction_type == 'max':
            out = F.max_pool1d(input=feat, kernel_size=vertex_num).squeeze(-1)
        elif self.attention_reduction_type == 'sum':
            out = feat.sum(-1)
        else:
            raise NotImplementedError
        return out

