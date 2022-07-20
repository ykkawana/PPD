import torch
from torch import nn
from torch.nn import functional as F
from model import decoder, pointnet, paramnet, atlasnet, discriminator


class MoveNetAtlasNetAutoEncoder(nn.Module):
    def __init__(self,
                 latent_dim=64,
                 primitive_num=2,
                 is_atlasnet_template_sphere=False,
                 pointnet_hidden_size=128,
                 pointnet_depth=3,
                 paramnet_hidden_size=128,
                 paramnet_out_act='tanh',
                 paramnet_depth=3,
                 discriminator_depth=3,
                 discriminator_hidden_size=128,
                 discriminator_encoder_depth=3,
                 param_dim=1,
                 is_shared_occnet_pointnet_as_D_encoder=False,
                 is_treegan_pointnet_as_D_encoder=True,
                 is_nsd_paramnet_as_D=False,
                 is_treegan_paramnet_as_D=True,
                 is_atlasnetv2=False,
                 is_adversarial=False,
                 use_discriminator_moving=False,
                 use_metric_learning=False):
        super().__init__()
        self.primitive_num = primitive_num
        self.is_atlasnet_template_sphere = is_atlasnet_template_sphere
        self.is_atlasnetv2 = is_atlasnetv2
        self.is_adversarial = is_adversarial
        self.is_nsd_paramnet_as_D = is_nsd_paramnet_as_D

        self.parameter_groups = {}
        self.model_groups = {}
        self.use_discriminator_moving = use_discriminator_moving
        self.use_metric_learning = use_metric_learning

        if self.is_atlasnetv2:
            self.decoder = atlasnet.AtlasNetV2(
                c_dim=latent_dim,
                dim=2,
                primitives_num=self.primitive_num,
                template_sphere=self.is_atlasnet_template_sphere)
        else:
            self.decoder = atlasnet.AtlasNet(
                c_dim=latent_dim,
                dim=2,
                primitives_num=self.primitive_num,
                template_sphere=self.is_atlasnet_template_sphere)
        self.pointnet = pointnet.OccNetSimplePointNet(
            c_dim=latent_dim,
            dim=2,
            hidden_dim=pointnet_hidden_size,
            depth=pointnet_depth)
        self.paramnet = paramnet.NSDParamNet(in_channel=latent_dim,
                                             param_dim=param_dim,
                                             layer_depth=paramnet_depth,
                                             out_act=paramnet_out_act,
                                             hidden_size=paramnet_hidden_size)

        self.model_groups['generator'] = [
            self.decoder, self.pointnet, self.paramnet
        ]
        self.parameter_groups['generator'] = []
        for model in self.model_groups['generator']:
            self.parameter_groups['generator'].extend(model.parameters())

        if self.is_adversarial:

            def init_discriminator():
                if is_treegan_pointnet_as_D_encoder:
                    D_encoder = pointnet.TreeGANPointNet(
                        dim=2, depth=discriminator_encoder_depth)
                    in_discriminator_channel = D_encoder.out_dim
                elif is_shared_occnet_pointnet_as_D_encoder:
                    D_encoder = self.pointnet
                    in_discriminator_channel = latent_dim
                else:
                    raise NotImplementedError

                if is_nsd_paramnet_as_D:
                    net_D = paramnet.NSDParamNet(
                        in_channel=in_discriminator_channel,
                        param_dim=1,
                        layer_depth=discriminator_depth,
                        hidden_size=discriminator_hidden_size)
                elif is_treegan_paramnet_as_D:
                    net_D = paramnet.TreeGANParamNet(
                        in_channel=in_discriminator_channel,
                        param_dim=1,
                        layer_depth=discriminator_depth)
                else:
                    raise NotImplementedError
                return D_encoder, net_D

            self.D_encoder, self.net_D = init_discriminator()
            self.parameter_groups['discriminator'] = [
                *self.D_encoder.parameters(), *self.net_D.parameters()
            ]
            self.model_groups['discriminator'] = [self.D_encoder, self.net_D]
            self.parameter_groups['discriminator'] = []
            for model in self.model_groups['discriminator']:
                self.parameter_groups['discriminator'].extend(
                    model.parameters())

            if self.use_discriminator_moving:
                self.D_encoder_moving, self.net_D_moving = init_discriminator()
                self.parameter_groups['discriminator'] = [
                    *self.D_encoder_moving.parameters(),
                    *self.net_D_moving.parameters()
                ]
                self.model_groups['discriminator'] = [
                    self.D_encoder_moving, self.net_D_moving
                ]
                self.parameter_groups['discriminator'] = []
                for model in self.model_groups['discriminator']:
                    self.parameter_groups['discriminator'].extend(
                        model.parameters())

    def forward(self,
                inputs,
                mode='generator',
                return_param=True,
                return_points=True):
        if mode == 'generator':
            ret = {}
            z = self.pointnet(inputs)
            ret['latent'] = z
            if return_points:
                point_set = self.decoder(z)
                ret['point_set'] = point_set
            if return_param:
                ret['param'] = self.paramnet(z)
            return ret
        elif mode == 'discriminator' and self.is_adversarial:
            z = self.D_encoder(inputs)
            param = self.net_D(z)
            if self.is_nsd_paramnet_as_D:
                return {'latent': z, 'D': param}
            else:
                return {'latent': z, 'D': param['out']}
        elif mode == 'discriminator_moving' and self.is_adversarial and self.use_discriminator_moving:
            z = self.D_encoder_moving(inputs)
            param = self.net_D_moving(z)
            if self.is_nsd_paramnet_as_D:
                return {'latent': z, 'D': param}
            else:
                return {'latent': z, 'D': param['out']}
        else:
            raise NotImplementedError


class MoveNetAutoEncoder(nn.Module):
    def __init__(self,
                 decoder_hidden_size=128,
                 decoder_depth=3,
                 decoder_leaky_relu=False,
                 is_atlasnet_template_sphere=False,
                 latent_dim=64,
                 primitive_num=2,
                 pointnet_hidden_size=128,
                 pointnet_depth=3,
                 paramnet_hidden_size=128,
                 paramnet_out_act='tanh',
                 paramnet_depth=3,
                 discriminator_depth=3,
                 discriminator_hidden_size=128,
                 discriminator_encoder_depth=3,
                 param_dim=1,
                 dim=2,
                 is_treegan_pointnet_as_D_encoder=True,
                 is_nsd_paramnet_as_D=False,
                 is_treegan_paramnet_as_D=True,
                 is_adversarial=False,
                 is_explicit=False,
                 is_implicit=False,
                 is_atlasnetv2=False,
                 use_discriminator_moving=False,
                 is_classification_head=False,
                 decode_sdf=False,
                 sdf_decoder_depth=6,
                 sdf_decoder_last_tanh=False,
                 learnable_attention=False,
                 attention_dim=750,
                 attention_type='none',
                 discriminator_canonical_additional_head=False,
                 discriminator_canonical_additional_head_dim=1,
                 discriminator_slide_additional_head=False,
                 discriminator_slide_additional_head_dim=1,
                 attention_reduction_type='max'):

        super().__init__()
        self.primitive_num = primitive_num
        self.is_adversarial = is_adversarial
        self.is_atlasnet_template_sphere = is_atlasnet_template_sphere
        self.parameter_groups = {}
        self.model_groups = {}
        self.use_discriminator_moving = use_discriminator_moving
        self.is_explicit = is_explicit
        self.is_implicit = is_implicit
        self.dim = dim
        self.is_atlasnetv2 = is_atlasnetv2
        self.is_classification_head = is_classification_head
        self.decode_sdf = decode_sdf
        self.param_dim = param_dim
        assert is_explicit != is_implicit
        self.discriminator_additional_head = discriminator_canonical_additional_head or discriminator_slide_additional_head
        if self.discriminator_additional_head:
            assert not is_nsd_paramnet_as_D

        if is_explicit:
            if self.is_atlasnetv2:
                self.decoder = atlasnet.AtlasNetV2(
                    c_dim=latent_dim,
                    dim=dim,
                    primitives_num=self.primitive_num,
                    template_sphere=self.is_atlasnet_template_sphere)
            else:
                self.decoder = atlasnet.AtlasNet(
                    c_dim=latent_dim,
                    dim=dim,
                    primitives_num=self.primitive_num,
                    template_sphere=self.is_atlasnet_template_sphere)
        elif is_implicit:
            if self.decode_sdf:
                self.decoder = decoder.SDFGenerator(
                    out_dim=primitive_num,
                    c_dim=latent_dim,
                    dim=dim,
                    hidden_size=decoder_hidden_size,
                    depth=sdf_decoder_depth,
                    last_tanh=sdf_decoder_last_tanh)
            else:
                self.decoder = decoder.DecoderBatchNorm(
                    out_dim=primitive_num +
                    (1 if self.is_classification_head else 0),
                    c_dim=latent_dim,
                    z_dim=0,
                    dim=dim,
                    hidden_size=decoder_hidden_size,
                    leaky=decoder_leaky_relu,
                    depth=decoder_depth)

        self.pointnet = pointnet.OccNetSimplePointNet(
            c_dim=latent_dim,
            dim=dim,
            hidden_dim=pointnet_hidden_size,
            depth=pointnet_depth)
        self.paramnet = paramnet.NSDParamNet(in_channel=latent_dim,
                                             param_dim=self.param_dim,
                                             layer_depth=paramnet_depth,
                                             out_act=paramnet_out_act,
                                             hidden_size=paramnet_hidden_size)

        self.model_groups['generator'] = [
            self.decoder, self.pointnet, self.paramnet
        ]
        self.parameter_groups['generator'] = []
        for model in self.model_groups['generator']:
            self.parameter_groups['generator'].extend(model.parameters())

        if self.is_adversarial:

            def init_discriminator():
                if self.is_implicit:
                    if self.is_classification_head:
                        discriminator_input_dim = dim + 1 + 1
                    else:
                        discriminator_input_dim = dim + 1
                else:
                    discriminator_input_dim = dim
                if is_treegan_pointnet_as_D_encoder:
                    D_encoder = pointnet.TreeGANPointNet(
                        dim=discriminator_input_dim,
                        depth=discriminator_encoder_depth,
                        learnable_attention=learnable_attention,
                        attention_type=attention_type,
                        attention_reduction_type=attention_reduction_type,
                        attention_dim=attention_dim)
                    in_discriminator_channel = D_encoder.out_dim
                else:
                    raise NotImplementedError

                if is_nsd_paramnet_as_D:
                    net_D = paramnet.NSDParamNet(
                        in_channel=in_discriminator_channel,
                        param_dim=1,
                        layer_depth=discriminator_depth,
                        hidden_size=discriminator_hidden_size)
                elif is_treegan_paramnet_as_D:
                    net_D = paramnet.TreeGANParamNet(
                        in_channel=in_discriminator_channel,
                        param_dim=1,
                        layer_depth=discriminator_depth,
                        canonical_additional_head=
                        discriminator_canonical_additional_head,
                        canonical_additional_head_dim=
                        discriminator_canonical_additional_head_dim,
                        slide_additional_head=
                        discriminator_slide_additional_head,
                        slide_additional_head_dim=
                        discriminator_slide_additional_head_dim)
                else:
                    raise NotImplementedError
                return D_encoder, net_D

            self.D_encoder, self.net_D = init_discriminator()
            self.parameter_groups['discriminator'] = [
                *self.D_encoder.parameters(), *self.net_D.parameters()
            ]
            self.model_groups['discriminator'] = [self.D_encoder, self.net_D]
            self.parameter_groups['discriminator'] = []
            for model in self.model_groups['discriminator']:
                self.parameter_groups['discriminator'].extend(
                    model.parameters())

            if self.use_discriminator_moving:
                self.D_encoder_moving, self.net_D_moving = init_discriminator()
                self.parameter_groups['discriminator'] = [
                    *self.D_encoder_moving.parameters(),
                    *self.net_D_moving.parameters()
                ]
                self.model_groups['discriminator'] = [
                    self.D_encoder_moving, self.net_D_moving
                ]
                self.parameter_groups['discriminator'] = []
                for model in self.model_groups['discriminator']:
                    self.parameter_groups['discriminator'].extend(
                        model.parameters())

    def forward(self,
                inputs,
                coord=None,
                mode='generator',
                mask=None,
                return_param=True,
                return_points=True,
                direct_input_to_D=False,
                return_occupancy=True):

        if mode == 'generator':
            ret = {}
            z = self.pointnet(inputs)
            ret['latent'] = z
            if return_points and self.is_explicit:
                point_set = self.decoder(z)
                ret['point_set'] = point_set
            if return_occupancy and self.is_implicit:
                occ = self.decoder(coord, None, z)
                ret['occupancy'] = occ
            if return_param:
                ret['param'] = self.paramnet(z)
            return ret
        elif mode.startswith('discriminator'):
            if mode == 'discriminator' and self.is_adversarial:
                if direct_input_to_D:
                    discriminator_z = inputs
                else:
                    discriminator_z = self.D_encoder(inputs, mask=mask)
                param = self.net_D(discriminator_z)
                ret = {'D_latent': discriminator_z, 'D': param['out']}
                ret.update(param)
                return ret
            elif mode == 'discriminator_moving' and self.is_adversarial and self.use_discriminator_moving:
                raise NotImplementedError
                discriminator_moving_z = self.D_encoder_moving(inputs,
                                                               mask=mask)
                param = self.net_D_moving(discriminator_moving_z)
                ret = {
                    'D_moving_latent': discriminator_moving_z,
                    'D': param['out']
                }
                ret.update(param)
                return ret
            else:
                raise NotImplementedError
        elif mode.startswith('occupancy_points_encoder'):
            x = self.D_encoder(inputs, mask=mask)
            x = F.normalize(x, p=2, dim=1)
            ret = {'latent_normalized': x}
            return ret
        else:
            raise NotImplementedError
