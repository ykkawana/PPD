from model.imnet.nsd_decoder import GroupWiseResnetBlockConv1dWoBN
import torch
from torch import batch_norm, nn
from torch.nn import functional as F
from model.layers import GroupWiseResnetBlockConv1d, GroupWiseResnetBlockConv1dGenericNorm, ResnetBlockConv1d, ResnetBlockConv1dWoBN, GroupWiseResnetBlockConv1dGBN, ResnetBlockConv1dLayerNorm
from model import layers


class NSDParamNet(nn.Module):
    def __init__(self,
                 in_channel,
                 param_dim,
                 dense=True,
                 layer_depth=0,
                 hidden_size=128,
                 out_act='tanh',
                 **kwargs):
        super().__init__()
        self.param_dim = param_dim
        self.dense = dense

        assert out_act in ['tanh', 'tanhshrink', None]
        self.out_act = out_act
        if layer_depth == 0:
            self.conv1d = nn.Linear(in_channel, in_channel)
            self.out_conv1d = nn.Linear(in_channel, param_dim)
            self.convs = []
        else:
            self.conv1d = nn.Linear(in_channel, hidden_size)
            self.convs = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size)
                for _ in range(layer_depth - 2)
            ])
            self.out_conv1d = nn.Linear(hidden_size, param_dim)
        #self.conv1d = nn.Conv1d(in_channel, in_channel, 1)
        self.act = nn.LeakyReLU(0.2, True)
        #self.out_conv1d = nn.Conv1d(in_channel, n_primitives * param_dim, 1)
        self.tanhshrink = torch.nn.Tanhshrink()

    def get_feature(self, x):
        if self.dense:
            x = self.act(self.conv1d(x))

        for conv in self.convs:
            x = self.act(conv(x))

        return x

    def forward(self, x):
        B = x.shape[0]
        x = self.get_feature(x)

        out = self.out_conv1d(x).view(B, self.param_dim)
        if self.out_act == None:
            return out
        elif self.out_act == 'tanh':
            return torch.tanh(out)
        elif self.out_act == 'tanhshrink':
            return self.tanhshrink(out)
        else:
            raise NotImplementedError


class TreeGANParamNet(nn.Module):
    def __init__(self,
                 in_channel,
                 param_dim,
                 layer_depth=4,
                 canonical_additional_head=False,
                 canonical_additional_head_dim=1,
                 slide_additional_head=False,
                 slide_additional_head_dim=1):
        super(TreeGANParamNet, self).__init__()
        self.additional_head = canonical_additional_head or slide_additional_head
        self.final_layers = nn.ModuleList([])
        in_ch = in_channel
        out_ch = in_channel
        for inx in range(layer_depth - 1):
            if inx % 2 == 0:
                in_ch = out_ch
                self.final_layers.append(nn.Linear(in_ch, out_ch))
            else:
                in_ch = out_ch
                out_ch = out_ch // 2
                self.final_layers.append(nn.Linear(in_ch, out_ch))
        self.final_layers.append(nn.Linear(out_ch, param_dim))
        self.additional_head_layers = nn.ModuleDict({})
        if canonical_additional_head:
            self.additional_head_layers['D_canonical'] = nn.Linear(
                out_ch, canonical_additional_head_dim)
        if slide_additional_head:
            self.additional_head_layers['D_slide'] = nn.Linear(
                out_ch, slide_additional_head_dim)

    def forward(self, out):
        outs = {}
        for idx, layer in enumerate(self.final_layers):
            if idx == len(self.final_layers) - 1:
                outs['out'] = layer(out)
                if self.additional_head:
                    for layer_name, layer in self.additional_head_layers.items(
                    ):
                        outs[layer_name] = layer(out)
            else:
                out = layer(out)

        return outs


class ParamNetBatchNormAtlasNetV2(nn.Module):
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
                                self.out_dim * self.param_out_dim,
                                1,
                                groups=self.out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, c, **kwargs):
        batch_size, D = c.size()

        net = self.fc_c(c).unsqueeze(2)
        net = net.unsqueeze(1).expand(-1, self.out_dim, -1,
                                      -1).reshape(batch_size, -1, 1)
        net = self.decoder(net)
        out = self.fc_out(self.actvn(self.bn(net)))

        # batch, points, occupancy=out_dim (primitive_num)
        out = out.reshape(batch_size, self.out_dim * self.param_out_dim,
                          1).transpose(1, 2).contiguous().reshape(
                              batch_size, 1, self.out_dim, self.param_out_dim)
        return out


class TreeGANParamNetSN(nn.Module):
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
        self.additional_head = canonical_additional_head or slide_additional_head
        self.final_layers = nn.ModuleList([])
        in_ch = in_channel
        out_ch = in_channel
        for inx in range(layer_depth - 1):
            if inx % 2 == 0:
                in_ch = out_ch
                self.final_layers.append(
                    nn.utils.spectral_norm(nn.Linear(in_ch, out_ch)))
            else:
                in_ch = out_ch
                out_ch = out_ch // 2
                self.final_layers.append(
                    nn.utils.spectral_norm(nn.Linear(in_ch, out_ch)))
        self.final_layers.append(
            nn.utils.spectral_norm(nn.Linear(out_ch, param_dim)))
        if special_weight_init:
            for layer in self.final_layers:
                layer.weight.data.normal_(0.0, 0.02)
        self.additional_head_layers = nn.ModuleDict({})

        if canonical_additional_head:
            self.additional_head_layers[
                'D_canonical'] = nn.utils.spectral_norm(
                    nn.Linear(out_ch, canonical_additional_head_dim))
        if slide_additional_head:
            self.additional_head_layers['D_slide'] = nn.utils.spectral_norm(
                nn.Linear(out_ch, slide_additional_head_dim))

    def forward(self, out):
        outs = {}
        for idx, layer in enumerate(self.final_layers):
            if idx == len(self.final_layers) - 1:
                outs['out'] = layer(out)
                if self.additional_head:
                    for layer_name, layer in self.additional_head_layers.items(
                    ):
                        outs[layer_name] = layer(out)
            else:
                out = layer(out)

        return outs


class NSDParamNetBN(nn.Module):
    def __init__(self,
                 in_channel,
                 param_dim,
                 dense=True,
                 layer_depth=0,
                 hidden_size=128,
                 out_act='tanh',
                 **kwargs):
        super().__init__()
        self.param_dim = param_dim
        self.dense = dense

        assert out_act in ['tanh', 'tanhshrink', None]
        self.out_act = out_act
        if layer_depth == 0:
            self.conv1d = nn.Conv1d(in_channel, in_channel, 1)
            self.out_conv1d = nn.Conv1d(in_channel, param_dim, 1)
            self.convs = []
        else:
            self.conv1d = nn.Conv1d(in_channel, hidden_size, 1)
            self.convs = nn.ModuleList([
                ResnetBlockConv1d(hidden_size) for _ in range(layer_depth - 2)
            ])
            self.out_conv1d = nn.Conv1d(hidden_size, param_dim, 1)
        #self.conv1d = nn.Conv1d(in_channel, in_channel, 1)
        self.act = nn.LeakyReLU(0.2, True)

        self.bn = nn.BatchNorm1d(hidden_size)
        #self.out_conv1d = nn.Conv1d(in_channel, n_primitives * param_dim, 1)
        self.tanhshrink = torch.nn.Tanhshrink()

    def get_feature(self, x):
        if self.dense:
            x = self.conv1d(x)

        for conv in self.convs:
            x = conv(x)

        return x

    def forward(self, x):
        B = x.shape[0]
        x = x.unsqueeze(-1)
        x = self.get_feature(x)

        out = self.out_conv1d(self.act(self.bn(x))).view(B, self.param_dim)
        if self.out_act == None:
            return out
        elif self.out_act == 'tanh':
            return torch.tanh(out)
        elif self.out_act == 'tanhshrink':
            return self.tanhshrink(out)
        else:
            raise NotImplementedError


class NSDParamNetBNRotTrans(nn.Module):
    def __init__(self,
                 out_dim=2,
                 param_out_dim=0,
                 c_dim=128,
                 z_dim=0,
                 dim=0,
                 leaky=True,
                 depth=0,
                 hidden_size=128,
                 rotation_head_dim=9,
                 translation_head_dim=3,
                 rotation_primitive_num=2,
                 translation_primitive_num=1,
                 **kwargs):
        super().__init__()
        self.rotation_head_dim = rotation_head_dim
        self.rotation_primtive_num = rotation_primitive_num
        self.translation_head_dim = translation_head_dim
        self.translation_primitive_num = translation_primitive_num
        layer_depth = depth
        in_channel = c_dim

        self.conv1d = nn.Conv1d(in_channel, hidden_size, 1)
        self.convs = nn.ModuleList(
            [ResnetBlockConv1d(hidden_size) for _ in range(layer_depth - 2)])
        self.out_trans = nn.Conv1d(
            hidden_size, translation_primitive_num * translation_head_dim, 1)
        self.out_rot = nn.Conv1d(hidden_size,
                                 rotation_primitive_num * rotation_head_dim, 1)
        #self.conv1d = nn.Conv1d(in_channel, in_channel, 1)
        self.act = nn.LeakyReLU(0.2, True)

        self.rot_bn = nn.BatchNorm1d(hidden_size)
        self.trans_bn = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        B = x.shape[0]
        x = x.unsqueeze(-1)
        x = self.conv1d(x)

        rot_out = self.out_rot(self.act(self.rot_bn(x))).view(
            B, self.rotation_primtive_num * self.rotation_head_dim)
        trans_out = self.out_trans(self.act(self.trans_bn(x))).view(
            B, self.translation_primitive_num * self.translation_head_dim)
        out = torch.cat([trans_out, rot_out], dim=1)
        return out


class NSDParamNetBN2(nn.Module):
    def __init__(self,
                 out_dim=0,
                 param_out_dim=0,
                 c_dim=128,
                 z_dim=0,
                 dim=0,
                 leaky=True,
                 depth=0,
                 hidden_size=128,
                 out_conv_scale=1,
                 init_out_conv_with_zeros=False):
        super().__init__()
        layer_depth = depth
        in_channel = c_dim
        self.param_out_dim = param_out_dim
        self.out_conv_scale = out_conv_scale

        self.conv1d = nn.Conv1d(in_channel, hidden_size, 1)
        self.convs = nn.ModuleList(
            [ResnetBlockConv1d(hidden_size) for _ in range(layer_depth - 2)])
        self.out_conv1d = nn.Conv1d(hidden_size, param_out_dim, 1)
        if init_out_conv_with_zeros:
            self.out_conv1d.weight.data.fill_(0.0)
            self.out_conv1d.bias.data.fill_(0.0)
        #self.conv1d = nn.Conv1d(in_channel, in_channel, 1)
        self.act = nn.LeakyReLU(0.2, True)

        self.bn = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        B = x.shape[0]
        x = x.unsqueeze(-1)
        x = self.conv1d(x)

        out = self.out_conv1d(self.act(self.bn(x))).view(
            B, self.param_out_dim) * self.out_conv_scale
        return out


class NSDParamNet2(nn.Module):
    def __init__(self,
                 out_dim=0,
                 param_out_dim=0,
                 c_dim=128,
                 z_dim=0,
                 dim=0,
                 leaky=True,
                 depth=0,
                 hidden_size=128,
                 init_out_conv_with_zeros=False):
        super().__init__()
        layer_depth = depth
        in_channel = c_dim
        self.param_out_dim = param_out_dim

        self.conv1d = nn.Conv1d(in_channel, hidden_size, 1)
        self.convs = nn.ModuleList(
            [ResnetBlockConv1d(hidden_size) for _ in range(layer_depth - 2)])
        self.out_conv1d = nn.Conv1d(hidden_size, param_out_dim, 1)
        if init_out_conv_with_zeros:
            self.out_conv1d.weight.data.fill_(0.0)
            self.out_conv1d.bias.data.fill_(0.0)
        #self.conv1d = nn.Conv1d(in_channel, in_channel, 1)
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        B = x.shape[0]
        x = x.unsqueeze(-1)
        x = self.conv1d(x)

        out = self.out_conv1d(self.act(x)).view(B, self.param_out_dim)
        return out


class NSDParamNetBN2MoreLayers(nn.Module):
    def __init__(self,
                 out_dim=0,
                 param_out_dim=0,
                 c_dim=128,
                 z_dim=0,
                 dim=0,
                 leaky=True,
                 depth=0,
                 hidden_size=128,
                 init_out_conv_with_zeros=False,
                 out_conv_scale=1):
        super().__init__()
        layer_depth = depth
        in_channel = c_dim
        self.param_out_dim = param_out_dim
        self.out_conv_scale = out_conv_scale

        self.conv1d = nn.Conv1d(in_channel, hidden_size, 1)
        self.convs = nn.Sequential(
            *[ResnetBlockConv1d(hidden_size) for _ in range(layer_depth - 2)])
        self.out_conv1d = nn.Conv1d(hidden_size, param_out_dim, 1)
        if init_out_conv_with_zeros:
            self.out_conv1d.weight.data.fill_(0.0)
            self.out_conv1d.bias.data.fill_(0.0)
        #self.conv1d = nn.Conv1d(in_channel, in_channel, 1)
        self.act = nn.LeakyReLU(0.2, True)

        self.bn = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        B = x.shape[0]
        x = x.unsqueeze(-1)
        x = self.conv1d(x)
        x = self.convs(x)

        out = self.out_conv1d(self.act(self.bn(x))).view(
            B, self.param_out_dim) * self.out_conv_scale
        return out


class NSDParamNet2MoreLayers(nn.Module):
    def __init__(self,
                 out_dim=0,
                 param_out_dim=0,
                 c_dim=128,
                 z_dim=0,
                 dim=0,
                 leaky=True,
                 depth=0,
                 hidden_size=128,
                 init_out_conv_with_zeros=False):
        super().__init__()
        layer_depth = depth
        in_channel = c_dim
        self.param_out_dim = param_out_dim

        self.conv1d = nn.Conv1d(in_channel, hidden_size, 1)
        self.convs = nn.Sequential(*[
            ResnetBlockConv1dWoBN(hidden_size) for _ in range(layer_depth - 2)
        ])
        self.out_conv1d = nn.Conv1d(hidden_size, param_out_dim, 1)
        if init_out_conv_with_zeros:
            self.out_conv1d.weight.data.fill_(0.0)
            self.out_conv1d.bias.data.fill_(0.0)
        #self.conv1d = nn.Conv1d(in_channel, in_channel, 1)
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        B = x.shape[0]
        x = x.unsqueeze(-1)
        x = self.conv1d(x)
        x = self.convs(x)

        out = self.out_conv1d(self.act(x)).view(B, self.param_out_dim)
        return out


class QuantizedEncoderFromSpec(nn.Module):
    def __init__(self,
                 hidden_size,
                 const_latent_dim,
                 spec,
                 batch_norm_type='batch_norm',
                 output_layer_init_method=False,
                 disable_learnable_quantized_latent_vectors=False):
        super().__init__()
        self.spec = spec
        canonical_loc_enc_layers = []
        const_out_dim = spec['out_dim']
        const_groups = spec['groups']
        self.const_groups = const_groups
        const_hidden_dim = int(hidden_size / const_groups)

        self.act = nn.LeakyReLU(0.2, False)
        if batch_norm_type == 'batch_norm':
            bn_c = nn.BatchNorm1d(const_latent_dim)
        elif batch_norm_type == 'layer_norm':
            bn_c = nn.LayerNorm([const_latent_dim, 1])
        elif batch_norm_type == 'no_norm':
            bn_c = nn.Identity()
        canonical_loc_enc_layers.append(bn_c)
        canonical_loc_enc_layers.append(self.act)

        fc_c = nn.Conv1d(const_latent_dim, const_hidden_dim * const_groups, 1)
        canonical_loc_enc_layers.append(fc_c)

        const_quantizer = layers.Quantize(
            const_hidden_dim * const_groups,
            spec['quantize_num'],
            disable_learnable_quantized_latent_vectors=
            disable_learnable_quantized_latent_vectors)
        canonical_loc_enc_layers.append(const_quantizer)

        idx = 0
        while True:
            size_out = int(const_hidden_dim / 2**(idx + 1))
            if size_out <= const_out_dim:
                break

            if batch_norm_type == 'batch_norm':
                block = GroupWiseResnetBlockConv1d(const_groups,
                                                   int(const_hidden_dim /
                                                       2**idx),
                                                   size_out=size_out)
            elif batch_norm_type == 'layer_norm':
                block = GroupWiseResnetBlockConv1dGBN(const_groups,
                                                      int(const_hidden_dim /
                                                          2**idx),
                                                      size_out=size_out)
            elif batch_norm_type == 'no_norm':
                block = GroupWiseResnetBlockConv1dWoBN(const_groups,
                                                       int(const_hidden_dim /
                                                           2**idx),
                                                       size_out=size_out)
            canonical_loc_enc_layers.append(block)

            idx += 1

        canonical_loc_enc_layers.append(self.act)

        canonical_loc_fc_out = nn.Conv1d(int(const_hidden_dim / 2**idx) *
                                         const_groups,
                                         const_groups * const_out_dim,
                                         1,
                                         groups=const_groups)
        if output_layer_init_method == 'zero':
            canonical_loc_fc_out.weight.data.fill_(0.0)
            canonical_loc_fc_out.bias.data.fill_(0.0)
        elif output_layer_init_method == 'uniform_01':
            canonical_loc_fc_out.weight.data.uniform_(-0.1, 0.1)
            canonical_loc_fc_out.bias.data.uniform_(-0.1, 0.1)
        canonical_loc_enc_layers.append(canonical_loc_fc_out)

        self.const_out_layers = nn.ModuleList(canonical_loc_enc_layers)

    def forward(self, x):
        batch_size = x.size(0)
        for idx, layer in enumerate(self.const_out_layers):
            if idx == 3:
                if self.spec.get('is_multiple_quantize', False):
                    x = x.view(batch_size * self.const_groups, -1, 1)
                x, diff, indices = layer(x.squeeze(-1))
                if self.spec.get('is_multiple_quantize', False):
                    x = x.view(batch_size, -1)
                    indices = indices.view(batch_size, self.const_groups)

                x = x.unsqueeze(-1)
            elif idx == len(self.const_out_layers) - 1:
                ret = layer(self.act(x)).view(x.size(0), self.spec['groups'],
                                              self.spec['out_dim'])
            else:
                x = layer(x)
        return ret, diff, indices


class ConstEncoderFromSpec(nn.Module):
    def __init__(self, hidden_size, spec, output_layer_init_method=False):
        super().__init__()
        self.spec = spec
        canonical_loc_enc_layers = []
        const_out_dim = spec['out_dim']
        const_groups = spec['groups']
        const_hidden_dim = int(hidden_size / const_groups)

        self.act = nn.LeakyReLU(0.2, False)
        const_quantizer = layers.ConstantInput(const_hidden_dim * const_groups)
        canonical_loc_enc_layers.append(const_quantizer)

        idx = 0
        while True:
            size_out = int(const_hidden_dim / 2**(idx + 1))
            if size_out <= const_out_dim * 2:
                break
            canonical_loc_enc_layers.append(
                GroupWiseResnetBlockConv1d(const_groups,
                                           int(const_hidden_dim / 2**idx),
                                           size_out=size_out))
            idx += 1

        canonical_loc_enc_layers.append(self.act)

        canonical_loc_fc_out = nn.Conv1d(int(const_hidden_dim / 2**idx) *
                                         const_groups,
                                         const_groups * const_out_dim,
                                         1,
                                         groups=const_groups)
        if output_layer_init_method == 'zero':
            canonical_loc_fc_out.weight.data.fill_(0.0)
            canonical_loc_fc_out.bias.data.fill_(0.0)
        elif output_layer_init_method == 'uniform_01':
            canonical_loc_fc_out.weight.data.uniform_(-0.1, 0.1)
            canonical_loc_fc_out.bias.data.uniform_(-0.1, 0.1)

        canonical_loc_enc_layers.append(canonical_loc_fc_out)
        self.const_out_layers = nn.ModuleList(canonical_loc_enc_layers)

    def forward(self, x):
        for idx, layer in enumerate(self.const_out_layers):
            if idx == len(self.const_out_layers) - 1:
                ret = layer(self.act(x)).view(x.size(0), self.spec['groups'],
                                              self.spec['out_dim'])
            if idx == 1:
                x = layer(x.unsqueeze(-1))
            else:
                x = layer(x)
        return ret


class AttentionEncoderFromSpec(nn.Module):
    def __init__(self,
                 hidden_size,
                 const_latent_dim,
                 spec,
                 output_layer_init_method=False):
        super().__init__()
        self.spec = spec
        canonical_loc_enc_layers = []
        const_out_dim = spec['out_dim']
        const_groups = spec['groups']
        self.const_groups = const_groups
        const_hidden_dim = int(hidden_size / const_groups)
        self.const_hidden_dim = const_hidden_dim

        self.act = nn.LeakyReLU(0.2, False)

        attention_module = layers.AttentionConstMemory(const_latent_dim,
                                                       const_groups,
                                                       spec['quantize_num'],
                                                       const_hidden_dim)

        canonical_loc_enc_layers.append(attention_module)

        idx = 0
        while True:
            size_out = int(const_hidden_dim / 2**(idx + 1))
            if size_out <= const_out_dim:
                break
            canonical_loc_enc_layers.append(
                GroupWiseResnetBlockConv1d(const_groups,
                                           int(const_hidden_dim / 2**idx),
                                           size_out=size_out))
            idx += 1

        canonical_loc_enc_layers.append(self.act)

        canonical_loc_fc_out = nn.Conv1d(int(const_hidden_dim / 2**idx) *
                                         const_groups,
                                         const_groups * const_out_dim,
                                         1,
                                         groups=const_groups)
        if output_layer_init_method == 'zero':
            canonical_loc_fc_out.weight.data.fill_(0.0)
            canonical_loc_fc_out.bias.data.fill_(0.0)
        elif output_layer_init_method == 'uniform_01':
            canonical_loc_fc_out.weight.data.uniform_(-0.1, 0.1)
            canonical_loc_fc_out.bias.data.uniform_(-0.1, 0.1)
        canonical_loc_enc_layers.append(canonical_loc_fc_out)

        self.const_out_layers = nn.ModuleList(canonical_loc_enc_layers)

    def forward(self, x):
        batch_size = x.size(0)
        for idx, layer in enumerate(self.const_out_layers):
            if idx == 0:
                x = layer(x).view(batch_size,
                                  self.const_groups * self.const_hidden_dim, 1)

            elif idx == len(self.const_out_layers) - 1:
                ret = layer(self.act(x)).view(x.size(0), self.spec['groups'],
                                              self.spec['out_dim'])
            else:
                x = layer(x)
        return ret


class ParamNetV2QuantizedCanonicalMotion(nn.Module):
    def __init__(self,
                 latent_dim=128,
                 depth=0,
                 hidden_size=128,
                 batch_norm_type='batch_norm',
                 features={},
                 motion_template_latent_dim=64,
                 disable_learnable_quantized_latent_vectors=False,
                 init_methods=[],
                 out_conv_scales={}):
        """
        features:
            e.g. {
                    name: {
                        out_dim: int,
                        groups: int
                        },
                    name: {
                        out_dim: int,
                        is_const: bool,
                        groups: int,
                        quantize_num: int
                        }
                    }
        """
        super().__init__()
        layer_depth = depth
        in_channel = latent_dim
        self.out_conv_scales = out_conv_scales

        self.conv1d = nn.Conv1d(in_channel, hidden_size, 1)

        if batch_norm_type == 'batch_norm':
            self.convs = nn.ModuleList([
                ResnetBlockConv1d(hidden_size) for _ in range(layer_depth - 2)
            ])
        elif batch_norm_type == 'layer_norm':
            self.convs = nn.ModuleList([
                ResnetBlockConv1dLayerNorm(hidden_size)
                for _ in range(layer_depth - 2)
            ])
        elif batch_norm_type == 'no_norm':
            self.convs = nn.ModuleList([
                ResnetBlockConv1dWoBN(hidden_size)
                for _ in range(layer_depth - 2)
            ])

        self.act = nn.LeakyReLU(0.2, False)
        self.features = features
        out_layers = {}
        const_out_layers = {}

        for name, spec in features.items():
            if spec.get('is_const', False):
                const_out_layers[name] = QuantizedEncoderFromSpec(
                    hidden_size,
                    motion_template_latent_dim,
                    spec,
                    batch_norm_type=batch_norm_type,
                    output_layer_init_method=init_methods.get(name, 'none'),
                    disable_learnable_quantized_latent_vectors=
                    disable_learnable_quantized_latent_vectors)
            else:
                out_layers[name] = nn.Conv1d(hidden_size,
                                             spec['out_dim'] * spec['groups'],
                                             1)

                if init_methods.get(name, 'none') == 'zero':
                    out_layers[name].weight.data.fill_(0.0)
                    out_layers[name].bias.data.fill_(0.0)
                elif init_methods.get(name, 'none') == 'uniform_01':
                    out_layers[name].weight.data.uniform_(-0.1, 0.1)
                    out_layers[name].bias.data.uniform_(-0.1, 0.1)

        self.out_layers = nn.ModuleDict(out_layers)
        self.const_out_layers = nn.ModuleDict(const_out_layers)

    def forward(self, inputs, motion_template_latent=None):
        B = inputs.shape[0]
        inputs = inputs.unsqueeze(-1)
        x = self.conv1d(inputs)
        for conv in self.convs:
            x = conv(x)

        ret = {}
        for name, layer in self.out_layers.items():
            spec = self.features[name]
            ret[name] = layer(self.act(x)).view(
                B, spec['groups'], spec['out_dim']) * self.out_conv_scales.get(
                    name, 1)

        diff_total = 0.
        indices_ret = {}
        for idx, (name, layer) in enumerate(self.const_out_layers.items()):
            out, diff, indices = layer(motion_template_latent.unsqueeze(-1))
            indices_ret[name] = indices
            ret[name] = out * self.out_conv_scales.get(name, 1)
            diff_total = diff_total + diff

        if not isinstance(diff, torch.Tensor):
            ret['latent_quantize_diff'] = torch.zeros([1],
                                                      dtype=x.dtype,
                                                      device=x.device)
        else:
            ret['latent_quantize_diff'] = diff_total

        ret['quantize_indices'] = indices_ret

        return ret


class ParamNetV2ConstCanonicalMotion(nn.Module):
    def __init__(self,
                 latent_dim=128,
                 depth=0,
                 hidden_size=128,
                 features={},
                 motion_template_latent_dim=64,
                 init_methods=[],
                 out_conv_scales={}):
        """
        features:
            e.g. {
                    name: {
                        out_dim: int,
                        groups: int
                        },
                    name: {
                        out_dim: int,
                        is_const: bool,
                        groups: int,
                        quantize_num: int
                        }
                    }
        """
        super().__init__()
        layer_depth = depth
        in_channel = latent_dim

        self.conv1d = nn.Conv1d(in_channel, hidden_size, 1)
        self.convs = nn.ModuleList(
            [ResnetBlockConv1d(hidden_size) for _ in range(layer_depth - 2)])

        self.act = nn.LeakyReLU(0.2, False)
        self.features = features
        self.out_conv_scales = out_conv_scales
        out_layers = {}
        const_out_layers = {}

        for name, spec in features.items():
            if spec.get('is_const', False):
                const_out_layers[name] = ConstEncoderFromSpec(
                    hidden_size,
                    spec,
                    output_layer_init_method=init_methods.get(name, 'none'))
            else:
                out_layers[name] = nn.Conv1d(hidden_size,
                                             spec['out_dim'] * spec['groups'],
                                             1)
                if init_methods.get(name, 'none') == 'zero':
                    out_layers[name].weight.data.fill_(0.0)
                    out_layers[name].bias.data.fill_(0.0)
                elif init_methods.get(name, 'none') == 'uniform_01':
                    out_layers[name].weight.data.uniform_(-0.1, 0.1)
                    out_layers[name].bias.data.uniform_(-0.1, 0.1)

        self.out_layers = nn.ModuleDict(out_layers)
        self.const_out_layers = nn.ModuleDict(const_out_layers)

    def forward(self, inputs):
        B = inputs.shape[0]
        inputs = inputs.unsqueeze(-1)
        x = self.conv1d(inputs)
        for conv in self.convs:
            x = conv(x)

        ret = {}
        for name, layer in self.out_layers.items():
            spec = self.features[name]
            ret[name] = layer(self.act(x)).view(
                B, spec['groups'], spec['out_dim']) * self.out_conv_scales.get(
                    name, 1)

        for idx, (name, layer) in enumerate(self.const_out_layers.items()):
            out = layer(inputs)
            ret[name] = out * self.out_conv_scales.get(name, 1)

        return ret


class ParamNetV2AttentionCanonicalMotion(nn.Module):
    def __init__(self,
                 latent_dim=128,
                 depth=0,
                 hidden_size=128,
                 features={},
                 motion_template_latent_dim=64,
                 init_methods=[],
                 out_conv_scales={}):
        """
        features:
            e.g. {
                    name: {
                        out_dim: int,
                        groups: int
                        },
                    name: {
                        out_dim: int,
                        is_const: bool,
                        groups: int,
                        quantize_num: int
                        }
                    }
        """
        super().__init__()
        layer_depth = depth
        in_channel = latent_dim
        self.out_conv_scales = out_conv_scales

        self.conv1d = nn.Conv1d(in_channel, hidden_size, 1)
        self.convs = nn.ModuleList(
            [ResnetBlockConv1d(hidden_size) for _ in range(layer_depth - 2)])

        self.act = nn.LeakyReLU(0.2, False)
        self.features = features
        out_layers = {}
        const_out_layers = {}

        for name, spec in features.items():
            if spec.get('is_const', False):
                const_out_layers[name] = AttentionEncoderFromSpec(
                    hidden_size,
                    motion_template_latent_dim,
                    spec,
                    output_layer_init_method=init_methods.get(name, 'none'))
            else:
                out_layers[name] = nn.Conv1d(hidden_size,
                                             spec['out_dim'] * spec['groups'],
                                             1)

                if init_methods.get(name, 'none') == 'zero':
                    out_layers[name].weight.data.fill_(0.0)
                    out_layers[name].bias.data.fill_(0.0)
                elif init_methods.get(name, 'none') == 'uniform_01':
                    out_layers[name].weight.data.uniform_(-0.1, 0.1)
                    out_layers[name].bias.data.uniform_(-0.1, 0.1)

        self.out_layers = nn.ModuleDict(out_layers)
        self.const_out_layers = nn.ModuleDict(const_out_layers)

    def forward(self, inputs, motion_template_latent=None):
        B = inputs.shape[0]
        inputs = inputs.unsqueeze(-1)
        x = self.conv1d(inputs)
        for conv in self.convs:
            x = conv(x)

        ret = {}
        for name, layer in self.out_layers.items():
            spec = self.features[name]
            ret[name] = layer(self.act(x)).view(
                B, spec['groups'], spec['out_dim']) * self.out_conv_scales.get(
                    name, 1)

        for idx, (name, layer) in enumerate(self.const_out_layers.items()):
            out = layer(motion_template_latent)
            ret[name] = out * self.out_conv_scales.get(name, 1)

        return ret


class ParamNetV2QuantizedCanonicalMotionGenericNorm(nn.Module):
    def __init__(self,
                 latent_dim=128,
                 depth=0,
                 hidden_size=128,
                 batch_norm_type='batch_norm',
                 batch_norm_type_const=None,
                 features={},
                 motion_template_latent_dim=64,
                 disable_learnable_quantized_latent_vectors=False,
                 init_methods={},
                 is_simple_constant_mode=False,
                 is_direct_constant_mode=False,
                 out_depths={},
                 out_depth=1,
                 group_norm_groups_per_channel=1,
                 batch_norm_momentum=0.1,
                 batch_norm_types_per_const={},
                 input_consts=[],
                 spec_overwrite={},
                 out_conv_scales={}):
        """
        features:
            e.g. {
                    name: {
                        out_dim: int,
                        groups: int
                        },
                    name: {
                        out_dim: int,
                        is_const: bool,
                        groups: int,
                        quantize_num: int
                        }
                    }
        """
        super().__init__()
        layer_depth = depth
        in_channel = latent_dim
        self.features = features
        self.out_conv_scales = out_conv_scales
        if batch_norm_type_const is None:
            batch_norm_type_const = batch_norm_type

        extra_in_channel = 0
        for name in input_consts:
            spec = self.features[name]
            if name in spec_overwrite:
                spec.update(spec_overwrite[name])
            if is_direct_constant_mode or spec['is_direct_constant']:
                extra_in_channel += spec['groups'] * spec['out_dim']
            else:
                extra_in_channel += int(
                    spec['groups'] / hidden_size) * spec['groups']
        self.conv1d = layers.NormDependentConv1d(in_channel + extra_in_channel,
                                                 hidden_size,
                                                 1,
                                                 norm_type=batch_norm_type)

        self.convs = nn.ModuleList([
            layers.GroupWiseResnetBlockConv1dGenericNorm(
                1,
                hidden_size,
                batch_norm_type=batch_norm_type,
                batch_norm_momentum=batch_norm_momentum)
            for _ in range(layer_depth - 2)
        ])
        self.act = nn.LeakyReLU(0.2, False)
        self.is_direct_constant_mode = is_direct_constant_mode
        self.input_consts = input_consts
        out_layers = {}
        const_out_layers = {}

        for name, spec in features.items():
            if spec.get('is_const', False):
                batch_norm_type_per_const = batch_norm_types_per_const.get(
                    name, batch_norm_type_const)
                if name in spec_overwrite:
                    spec.update(spec_overwrite[name])
                if self.is_direct_constant_mode or spec.get(
                        'is_direct_constant', False):
                    const_out_layers[name] = layers.ConstantInput(
                        spec['out_dim'] * spec['groups'],
                        init_method=init_methods.get(name, 'none'))
                else:
                    const_out_layers[
                        name] = QuantizedEncoderFromSpecGenericNorm(
                            hidden_size,
                            motion_template_latent_dim,
                            spec,
                            batch_norm_type=batch_norm_type_per_const,
                            batch_norm_momentum=batch_norm_momentum,
                            output_layer_init_method=init_methods.get(
                                name, 'none'),
                            is_return_quantize_latent=(name in input_consts),
                            group_norm_groups_per_channel=
                            group_norm_groups_per_channel,
                            is_simple_constant_mode=is_simple_constant_mode,
                            disable_learnable_quantized_latent_vectors=
                            disable_learnable_quantized_latent_vectors)
            else:
                if name in spec_overwrite:
                    spec.update(spec_overwrite[name])
                last_conv = nn.Conv1d(hidden_size,
                                      spec['out_dim'] * spec['groups'], 1)
                if out_depths.get(name, out_depth) == 1:
                    out_layers[name] = last_conv
                    if init_methods.get(name, 'none') == 'zero':
                        out_layers[name].weight.data.fill_(0.0)
                        out_layers[name].bias.data.fill_(0.0)
                    elif init_methods.get(name, 'none') == 'uniform_01':
                        out_layers[name].weight.data.uniform_(-0.1, 0.1)
                        out_layers[name].bias.data.uniform_(-0.1, 0.1)
                else:
                    outs = [
                        layers.InputDependentConvGenericNormRelu(
                            hidden_size,
                            hidden_size,
                            1,
                            momentum=batch_norm_momentum,
                            norm_type=batch_norm_type)
                        for _ in range(out_depths.get(name, 1) - 1)
                    ]

                    outs.append(last_conv)
                    out_layers[name] = nn.Sequential(*outs)

                    if init_methods.get(name, 'none') == 'zero':
                        outs[-1].weight.data.fill_(0.0)
                        outs[-1].bias.data.fill_(0.0)
                        #out_layers[name].weight.data.fill_(0.0)
                        #out_layers[name].bias.data.fill_(0.0)
                    elif init_methods.get(name, 'none') == 'uniform_01':
                        outs[-1].weight.data.uniform_(-0.1, 0.1)
                        outs[-1].bias.data.uniform_(-0.1, 0.1)
                        #out_layers[name].weight.data.uniform_(-0.1, 0.1)
                        #out_layers[name].bias.data.uniform_(-0.1, 0.1)

        self.out_layers = nn.ModuleDict(out_layers)
        self.const_out_layers = nn.ModuleDict(const_out_layers)

    def forward(self, inputs, motion_template_latent=None):
        B = inputs.shape[0]

        diff_total = 0.
        indices_ret = {}
        latents = {}
        ret = {}
        for idx, (name, layer) in enumerate(self.const_out_layers.items()):
            spec = self.features[name]
            if self.is_direct_constant_mode or spec.get(
                    'is_direct_constant', False):
                out = layer(motion_template_latent)
                out = out.view(B, spec['groups'], spec['out_dim'])
                diff = torch.zeros_like(out).mean()
                indices = torch.zeros_like(out)[..., 0].long()
                if name in self.input_consts:
                    latent = out.view(B, -1)
                    latents[name] = latent
            else:
                lret = layer(motion_template_latent.unsqueeze(-1))
                if name in self.input_consts:
                    out, diff, indices, latent = lret
                    latents[name] = latent
                else:
                    out, diff, indices = lret
            indices_ret[name] = indices
            ret[name] = out * self.out_conv_scales.get(name, 1)
            diff_total = diff_total + diff

        inputs = inputs.unsqueeze(-1)

        if len(latents) > 0:
            inputs = torch.cat([inputs, *list(latents.values())], -1)
        x = self.conv1d(inputs)
        for conv in self.convs:
            x = conv(x)

        for name, layer in self.out_layers.items():
            spec = self.features[name]
            ret[name] = layer(self.act(x)).view(
                B, spec['groups'], spec['out_dim']) * self.out_conv_scales.get(
                    name, 1)

        if not isinstance(diff, torch.Tensor):
            ret['latent_quantize_diff'] = torch.zeros([1],
                                                      dtype=x.dtype,
                                                      device=x.device)
        else:
            ret['latent_quantize_diff'] = diff_total

        ret['quantize_indices'] = indices_ret

        return ret


class QuantizedEncoderFromSpecGenericNorm(nn.Module):
    def __init__(self,
                 hidden_size,
                 const_latent_dim,
                 spec,
                 batch_norm_type='batch_norm',
                 output_layer_init_method=False,
                 batch_norm_momentum=0.1,
                 is_simple_constant_mode=False,
                 is_return_quantize_latent=False,
                 group_norm_groups_per_channel=1,
                 disable_learnable_quantized_latent_vectors=False):
        super().__init__()
        self.spec = spec
        canonical_loc_enc_layers = []
        const_out_dim = spec['out_dim']
        const_groups = spec['groups']
        self.const_groups = const_groups
        self.group_norm_groups_per_channel = group_norm_groups_per_channel
        self.is_return_quantize_latent = is_return_quantize_latent
        const_hidden_dim = int(hidden_size / const_groups /
                               group_norm_groups_per_channel)

        self.act = nn.LeakyReLU(0.2, False)
        bn_c = layers.GenericNorm1d(const_latent_dim,
                                    norm_type=batch_norm_type,
                                    groups=1,
                                    momentum=batch_norm_momentum)
        canonical_loc_enc_layers.append(bn_c)
        canonical_loc_enc_layers.append(self.act)

        fc_c = nn.Conv1d(
            const_latent_dim,
            const_hidden_dim * const_groups * group_norm_groups_per_channel, 1)
        canonical_loc_enc_layers.append(fc_c)

        const_quantizer = layers.Quantize(
            const_hidden_dim * const_groups * group_norm_groups_per_channel,
            spec['quantize_num'],
            is_simple_constant_mode=is_simple_constant_mode,
            disable_learnable_quantized_latent_vectors=
            disable_learnable_quantized_latent_vectors)
        canonical_loc_enc_layers.append(const_quantizer)

        idx = 0
        while True:
            size_out = int(const_hidden_dim / 2**(idx + 1))
            if size_out <= const_out_dim:
                last_const_size_out = int(const_hidden_dim / 2**idx)
                break

            block = layers.GroupWiseResnetBlockConv1dGenericNorm(
                const_groups,
                int(const_hidden_dim / 2**idx),
                size_out=size_out,
                batch_norm_type=batch_norm_type,
                group_norm_groups_per_channel=group_norm_groups_per_channel,
                batch_norm_momentum=batch_norm_momentum)
            canonical_loc_enc_layers.append(block)

            idx += 1

        #canonical_loc_enc_layers.append(self.act)
        self.last_bn = layers.GenericNorm1d(
            last_const_size_out * const_groups * group_norm_groups_per_channel,
            norm_type=batch_norm_type,
            groups=const_groups * group_norm_groups_per_channel,
            momentum=batch_norm_momentum)

        canonical_loc_fc_out = nn.Conv1d(int(const_hidden_dim / 2**idx) *
                                         const_groups *
                                         group_norm_groups_per_channel,
                                         const_groups * const_out_dim,
                                         1,
                                         groups=const_groups)
        if output_layer_init_method == 'zero':
            canonical_loc_fc_out.weight.data.fill_(0.0)
            canonical_loc_fc_out.bias.data.fill_(0.0)
        elif output_layer_init_method == 'uniform_01':
            canonical_loc_fc_out.weight.data.uniform_(-0.1, 0.1)
            canonical_loc_fc_out.bias.data.uniform_(-0.1, 0.1)
        canonical_loc_enc_layers.append(canonical_loc_fc_out)
        self.const_out_layers = nn.ModuleList(canonical_loc_enc_layers)

    def forward(self, x):
        batch_size = x.size(0)
        for idx, layer in enumerate(self.const_out_layers):
            if idx == 3:
                if self.spec.get('is_multiple_quantize', False):
                    x = x.view(
                        batch_size * self.const_groups *
                        self.group_norm_groups_per_channel, -1, 1)
                x, diff, indices = layer(x.squeeze(-1))
                latent = x
                if self.spec.get('is_multiple_quantize', False):
                    x = x.view(batch_size, -1)
                    indices = indices.view(
                        batch_size,
                        self.const_groups * self.group_norm_groups_per_channel)

                x = x.unsqueeze(-1)
            elif idx == len(self.const_out_layers) - 1:
                ret = layer(self.act(self.last_bn(x))).view(
                    x.size(0), self.spec['groups'], self.spec['out_dim'])
            else:
                x = layer(x)
        if self.is_return_quantize_latent:
            return ret, diff, indices, latent
        return ret, diff, indices
