import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import ResnetBlockConv1d


class PointNetDiscriminator(nn.Module):
    def __init__(self, dim=3, depth=5, final_depth=4):
        super(PointNetDiscriminator, self).__init__()
        self.layer_num = depth

        self.fc_layers = nn.ModuleList([])

        self.fc_layers.append(nn.Conv1d(dim, 64, kernel_size=1, stride=1))

        for inx in range(self.layer_num - 1):
            self.fc_layers.append(
                nn.Conv1d(64 * (2**inx),
                          64 * (2**(inx + 1)),
                          kernel_size=1,
                          stride=1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.final_layers = nn.ModuleList([])
        in_ch = 64 * 2**(self.layer_num - 1)
        out_ch = 64 * 2**(self.layer_num - 1)
        for inx in range(final_depth - 1):
            if inx % 2 == 0:
                in_ch = out_ch
                self.final_layers.append(nn.Linear(in_ch, out_ch))
            else:
                in_ch = out_ch
                out_ch = out_ch // 2
                self.final_layers.append(nn.Linear(in_ch, out_ch))
        self.final_layers.append(nn.Linear(out_ch, 1))

    def forward(self, f):
        feat = f.transpose(1, 2)
        vertex_num = feat.size(2)

        for layer in self.fc_layers:
            feat = layer(feat)
            feat = self.leaky_relu(feat)

        out = F.max_pool1d(input=feat, kernel_size=vertex_num).squeeze(-1)
        for layer in self.final_layers:
            out = layer(out)

        return out


class ResNetDiscriminator(nn.Module):
    def __init__(self, in_channel, param_dim, layer_depth=4, hidden_size=128):
        super(ResNetDiscriminator, self).__init__()
        final_layers = []
        for idx in range(layer_depth):
            if idx == 0:
                size_in = in_channel
                size_out = hidden_size
            else:
                size_in = int(hidden_size / 2**(idx - 1))
                size_out = int(hidden_size / 2**idx)
            final_layers.extend(
                [ResnetBlockConv1d(size_in, size_out=size_out)])
        self.bn = nn.BatchNorm1d(size_out)

        self.fc = nn.Conv1d(size_out, param_dim, 1)
        self.actvn = nn.ReLU()

        self.net = nn.Sequential(*final_layers)

    def forward(self, out):
        outs = {}
        out = self.net(out.unsqueeze(-1))
        out = self.fc(self.actvn(self.bn(out)))
        outs['out'] = out.squeeze(-1)
        return outs


class VoxelDiscriminatorSNELU(nn.Module):
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
        layers = []
        c = nn.utils.spectral_norm(
            nn.Conv3d(in_channels=1,
                      out_channels=64,
                      kernel_size=4,
                      stride=2,
                      padding=1))
        c.weight.data.normal_(0.0, 0.02)
        layers.extend([c, nn.SELU()])
        c = nn.utils.spectral_norm(
            nn.Conv3d(in_channels=64,
                      out_channels=128,
                      kernel_size=4,
                      stride=2,
                      padding=1))
        c.weight.data.normal_(0.0, 0.02)
        layers.extend([c, nn.SELU()])
        c = nn.utils.spectral_norm(
            nn.Conv3d(in_channels=128,
                      out_channels=256,
                      kernel_size=4,
                      stride=2,
                      padding=1))
        c.weight.data.normal_(0.0, 0.02)
        layers.extend([c, nn.SELU()])
        c = nn.utils.spectral_norm(
            nn.Conv3d(in_channels=256, out_channels=1, kernel_size=2,
                      stride=1))
        c.weight.data.normal_(0.0, 0.02)
        layers.extend([c])
        self.layers = nn.Sequential(*layers)
        self.out_dim = 1

    def forward(self, x, *args, **kwargs):
        B = x.size(0)
        for layer in self.layers:
            x = layer(x)
        return x.view(B, 1)


class DummyDiscriminatorDecoder(nn.Module):
    def __init__(self,
                 in_channel,
                 param_dim,
                 layer_depth=4,
                 canonical_additional_head=False,
                 canonical_additional_head_dim=1,
                 slide_additional_head=False,
                 slide_additional_head_dim=1,
                 special_weight_init=False):
        super().__init__()

    def forward(self, x):
        return {'out': x}
