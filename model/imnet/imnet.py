from torch import nn
from model.imnet import encoder, generator
from model import paramnet, pointnet, decoder
import torch
from torch.nn import functional as F
from utils import geometry
import numpy as np
from collections import OrderedDict, defaultdict
from model.imnet import gumbel_softmax
import importlib
import trimesh


class IMNetAutoEncoder(nn.Module):
    def __init__(
        self,
        primitive_num=5,
        latent_dim=256,
        dim=3,
        decoder_hidden_size=128,
        decoder_last_act_type='imnet_original',
        decoder_type='imnet',
        decoder_depth=3,
        decoder_leaky_relu=False,
        voxel_encoder_hidden_size=32,
        paramnet_hidden_size=128,
        paramnet_out_act='tanh',
        is_adversarial=False,
        paramnet_depth=3,
        learnable_attention=False,
        attention_dim=750,
        attention_type='none',
        discriminator_depth=3,
        discriminator_hidden_size=128,
        discriminator_encoder_depth=3,
        attention_reduction_type='max',
        encoder_type='voxelnet',
        pointnet_hidden_size=128,
        pointnet_depth=3,
        is_split_latent_dim_for_motion=False,
        splitted_latent_dim_for_motion=128,
        motion_decoding_type='default',
        decode_rotation_axis_type='default',
        axis_weighting_type='softmax',
        rotation_primitive_num=3,
        anchor_point_estimation_type='default',
        occnet_decoder_type='model.decoder.DecoderBatchNorm',
        is_input_motion_latent_to_shape_decoder=False,
        use_motion_shape_integrated_decoder=False,
        paramnet_decoder_type='model.paramnet.NSDParamNet',
        paramnet_leaky_relu=False,
        use_diff_decoder=False,
        discriminator_encoder_type='model.pointnet.TreeGANPointNet',
        discriminator_decoder_type='model.paramnet.TreeGANParamNet',
        param_type='default',
        use_canonical_location_from_generator=False,
        use_canonical_direction_from_generator=False,
        is_skip_direction_normalization_in_canonical_direction_from_generator=False,
        disable_sigmoid_for_amount=False,
        is_decode_canonical_direction_as_rot_matrix=False,
        is_generator_input_motion_and_shape_template_latent=False,
        skip_learning_models=[],
        paramnet_version='v1',
        use_canonical_motion_from_paramnet_and_shape_from_generator=False,
        canonical_direction_decode_type='svd',
        rotation_decode_type='svd',
        canonical_direction_init_directions=[],
        is_expand_rotation_for_euler_angle=False,
        expand_rotation_for_euler_angle_sample_num=4,
        is_expand_rotation_for_euler_angle_width_angle=20,
        is_expand_rotation_for_euler_angle_decay=False,
        expand_rotation_for_euler_angle_decay_per_iter=0,
        skip_apply_tanh_to_canonical_location=False,
        paramnet_kwargs={},
        generator_kwargs={},
        discriminator_encoder_kwargs={},
        discriminator_decoder_kwargs={},
        encoder_kwargs={},
        bound_amount_to_pi_and_one=False,
        model_optimizer_group_names={
            'generator': ['generator', 'encoder', 'paramnet'],
            'discriminator': ['net_D', 'D_encoder']
        }):
        super().__init__()
        self.voxel_encoder_hidden_size = voxel_encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.latent_dim = latent_dim
        self.total_latent_dim = self.latent_dim
        self.dim = dim
        self.primitive_num = primitive_num
        self.rotation_primitive_num = rotation_primitive_num
        self.is_adversarial = is_adversarial
        self.pointnet_hidden_size = pointnet_hidden_size
        self.pointnet_depth = pointnet_depth
        self.decoder_type = decoder_type
        assert self.decoder_type in ['imnet', 'occnet', 'nsd']
        if decoder_last_act_type == 'imnet_original':
            assert self.decoder_type == 'imnet'

        self.is_split_latent_dim_for_motion = is_split_latent_dim_for_motion
        self.splitted_latent_dim_for_motion = splitted_latent_dim_for_motion
        if self.is_split_latent_dim_for_motion:
            self.total_latent_dim = self.latent_dim + self.splitted_latent_dim_for_motion
            self.motion_latent_dim = self.splitted_latent_dim_for_motion
        else:
            self.motion_latent_dim = self.latent_dim
        self.is_input_motion_latent_to_shape_decoder = is_input_motion_latent_to_shape_decoder
        if self.is_input_motion_latent_to_shape_decoder:
            self.latent_dim = self.total_latent_dim

        self.motion_decoding_type = motion_decoding_type
        assert self.motion_decoding_type in [
            'default', 'one_joint_type_per_primitive',
            'one_joint_type_per_primitive_rot_pm_num_specified'
        ]
        if self.motion_decoding_type == 'default':
            self.rotation_primitive_num = self.primitive_num - 1
            self.translation_primitive_num = self.primitive_num - 1
        if self.motion_decoding_type == 'one_joint_type_per_primitive':
            assert (self.primitive_num - 1) % 2 == 0
            self.rotation_primitive_num = (self.primitive_num - 1) / 2
            self.translation_primitive_num = self.primitive_num - self.rotation_primitive_num - 1

        if self.motion_decoding_type == 'one_joint_type_per_primitive_rot_pm_num_specified':
            assert self.primitive_num > self.rotation_primitive_num
            self.translation_primitive_num = self.primitive_num - self.rotation_primitive_num - 1

        self.decode_rotation_axis_type = decode_rotation_axis_type
        assert self.decode_rotation_axis_type in [
            'default', 'axis_direction_classification'
        ]
        self.axis_weighting_type = axis_weighting_type
        assert self.axis_weighting_type in ['softmax', 'gumbel', 'gumbel_hard']

        self.anchor_point_estimation_type = anchor_point_estimation_type
        assert self.anchor_point_estimation_type in [
            'default', 'direct_regression', 'direct_regression_fixed',
            'sphere_regression'
        ]
        self.occnet_decoder_type = occnet_decoder_type
        self.use_motion_shape_integrated_decoder = use_motion_shape_integrated_decoder
        if self.use_motion_shape_integrated_decoder:
            assert self.decoder_type == 'occnet'

        self.use_diff_decoder = use_diff_decoder
        self.is_generator_return_dict = self.use_motion_shape_integrated_decoder or self.use_diff_decoder
        self.paramnet_decoder_type = paramnet_decoder_type
        self.is_primitive_wise_motion_output = self.use_motion_shape_integrated_decoder or self.paramnet_decoder_type not in [
            'model.paramnet.NSDParamNet',
            'model.paramnet.NSDParamNetBN',
            'model.paramnet.NSDParamNetBNRotTrans',
            'model.paramnet.NSDParamNetBN2',
            'model.paramnet.NSDParamNet2',
            'model.paramnet.NSDParamNetBN2MoreLayers',
            'model.paramnet.NSDParamNet2MoreLayers',
        ]
        self.param_type = param_type
        assert self.param_type in [
            'default', 'affine', 'motion_separate_affine_rotation_quaternion',
            'only_amount', 'only_amount_as_matrix',
            'only_amount_as_matrix_loc_offset',
            'only_amount_as_matrix_loc_offset_canonical_motion',
            'only_amount_as_matrix_loc_offset_canonical_motion_euler_angle',
            'only_amount_as_matrix_loc_offset_canonical_motion_euler_angle_plus_direction_offset',
            'only_amount_as_matrix_loc_offset_canonical_motion_euler_angle_screw',
            'only_amount_as_matrix_loc_offset_canonical_motion_euler_angle_multiple_quantized_encoder'
        ]
        if self.param_type not in [
                'only_amount_as_matrix_loc_offset_canonical_motion_euler_angle_screw'
        ]:
            assert self.translation_primitive_num > 0

        self.use_canonical_location_from_generator = use_canonical_location_from_generator
        self.use_canonical_direction_from_generator = use_canonical_direction_from_generator
        self.is_skip_direction_normalization_in_canonical_direction_from_generator = is_skip_direction_normalization_in_canonical_direction_from_generator
        self.disable_sigmoid_for_amount = disable_sigmoid_for_amount
        self.is_decode_canonical_direction_as_rot_matrix = is_decode_canonical_direction_as_rot_matrix
        self.is_generator_input_motion_and_shape_template_latent = is_generator_input_motion_and_shape_template_latent
        self.use_canonical_motion_from_paramnet_and_shape_from_generator = use_canonical_motion_from_paramnet_and_shape_from_generator
        self.use_canonical_motion_and_shape_latent = self.is_generator_input_motion_and_shape_template_latent or self.use_canonical_motion_from_paramnet_and_shape_from_generator
        if self.use_canonical_motion_and_shape_latent:
            if self.is_generator_input_motion_and_shape_template_latent:
                self.motion_template_latent_dim = generator_kwargs[
                    'motion_template_latent_dim']
                self.shape_template_latent_dim = generator_kwargs[
                    'shape_template_latent_dim']
            elif self.use_canonical_motion_from_paramnet_and_shape_from_generator:
                self.motion_template_latent_dim = paramnet_kwargs[
                    'motion_template_latent_dim']
                self.shape_template_latent_dim = generator_kwargs[
                    'shape_template_latent_dim']
            self.total_latent_dim += (self.motion_template_latent_dim +
                                      self.shape_template_latent_dim)
        self.paramnet_version = paramnet_version
        self.canonical_direction_decode_type = canonical_direction_decode_type
        assert self.canonical_direction_decode_type in ['svd', 'euler_angle']
        self.canonical_direction_init_directions = canonical_direction_init_directions

        self.rotation_decode_type = rotation_decode_type
        assert self.rotation_decode_type in ['svd', 'euler_angle']
        self.is_expand_rotation_for_euler_angle = is_expand_rotation_for_euler_angle
        if self.is_expand_rotation_for_euler_angle:
            assert self.rotation_decode_type in ['euler_angle']
        self.expand_rotation_for_euler_angle_sample_num = expand_rotation_for_euler_angle_sample_num
        self.is_expand_rotation_for_euler_angle_width_angle = is_expand_rotation_for_euler_angle_width_angle
        self.skip_apply_tanh_to_canonical_location = skip_apply_tanh_to_canonical_location
        if paramnet_version in ['v2']:
            assert not self.skip_apply_tanh_to_canonical_location
        self.is_expand_rotation_for_euler_angle_decay = is_expand_rotation_for_euler_angle_decay
        self.expand_rotation_for_euler_angle_decay_per_iter = expand_rotation_for_euler_angle_decay_per_iter
        if self.param_type == 'only_amount_as_matrix_loc_offset_canonical_motion_euler_angle_multiple_quantized_encoder':
            self.canonical_direction_quantize_num = paramnet_kwargs.get(
                'canonical_direction_quantize_num', 4)
        assert len(self.canonical_direction_init_directions) == 0 or len(
            self.canonical_direction_init_directions
        ) == self.primitive_num - 1 or len(
            self.canonical_direction_init_directions
        ) == self.canonical_direction_quantize_num

        self.bound_amount_to_pi_and_one = bound_amount_to_pi_and_one

        self.init_param_dim()

        if encoder_type == 'voxelnet':
            self.encoder = encoder.encoder(self.voxel_encoder_hidden_size,
                                           self.total_latent_dim)
        else:
            if encoder_type == 'pointnet':
                encoder_type = 'model.pointnet.OccNetSimplePointNet'
            tmp = encoder_type.split('.')
            module_path = '.'.join(tmp[:-1])
            class_name = tmp[-1]
            self.encoder = getattr(importlib.import_module(module_path),
                                   class_name)(
                                       c_dim=self.total_latent_dim,
                                       dim=self.dim,
                                       hidden_dim=self.pointnet_hidden_size,
                                       depth=self.pointnet_depth,
                                       **encoder_kwargs)

        if self.decoder_type == 'imnet':
            self.generator = generator.generator(
                self.latent_dim,
                self.dim,
                self.decoder_hidden_size,
                out_dim=self.primitive_num,
                last_act_type=decoder_last_act_type)
        elif self.decoder_type == 'occnet':
            tmp = self.occnet_decoder_type.split('.')
            module_path = '.'.join(tmp[:-1])
            class_name = tmp[-1]
            if self.use_motion_shape_integrated_decoder:
                self.generator = getattr(importlib.import_module(module_path),
                                         class_name)(
                                             out_dim=primitive_num,
                                             param_out_dim=self.param_dim,
                                             c_dim=self.latent_dim,
                                             z_dim=0,
                                             dim=dim,
                                             hidden_size=decoder_hidden_size,
                                             leaky=decoder_leaky_relu,
                                             depth=decoder_depth,
                                             **generator_kwargs)
            else:
                self.generator = getattr(importlib.import_module(module_path),
                                         class_name)(
                                             out_dim=primitive_num,
                                             c_dim=self.latent_dim,
                                             z_dim=0,
                                             dim=dim,
                                             hidden_size=decoder_hidden_size,
                                             leaky=decoder_leaky_relu,
                                             depth=decoder_depth,
                                             **generator_kwargs)
        if not self.use_motion_shape_integrated_decoder:
            if paramnet_version == 'v2':
                paramnet_tmp = self.paramnet_decoder_type.split('.')
                paramnet_module_path = '.'.join(paramnet_tmp[:-1])
                paramnet_class_name = paramnet_tmp[-1]
                self.paramnet = getattr(
                    importlib.import_module(paramnet_module_path),
                    paramnet_class_name)(latent_dim=self.motion_latent_dim,
                                         hidden_size=paramnet_hidden_size,
                                         depth=paramnet_depth,
                                         features=self.paramnet_features,
                                         **paramnet_kwargs)
            elif paramnet_version == 'v1':
                if self.paramnet_decoder_type in [
                        'model.paramnet.NSDParamNet',
                        'model.paramnet.NSDParamNetBN'
                ]:
                    self.paramnet = paramnet.NSDParamNet(
                        in_channel=self.motion_latent_dim,
                        param_dim=self.param_dim,
                        layer_depth=paramnet_depth,
                        out_act=paramnet_out_act,
                        hidden_size=paramnet_hidden_size)

                else:
                    paramnet_tmp = self.paramnet_decoder_type.split('.')
                    paramnet_module_path = '.'.join(paramnet_tmp[:-1])
                    paramnet_class_name = paramnet_tmp[-1]
                    self.paramnet = getattr(
                        importlib.import_module(paramnet_module_path),
                        paramnet_class_name)(out_dim=primitive_num,
                                             param_out_dim=self.param_dim,
                                             c_dim=self.motion_latent_dim,
                                             z_dim=0,
                                             dim=dim,
                                             hidden_size=paramnet_hidden_size,
                                             leaky=paramnet_leaky_relu,
                                             depth=paramnet_depth,
                                             **paramnet_kwargs)
            else:
                raise NotImplementedError

        if self.is_adversarial:

            def init_discriminator():
                discriminator_input_dim = dim + 1
                tmp = discriminator_encoder_type.split('.')
                module_path = '.'.join(tmp[:-1])
                class_name = tmp[-1]
                #D_encoder = pointnet.TreeGANPointNet
                D_encoder = getattr(
                    importlib.import_module(module_path), class_name)(
                        dim=discriminator_input_dim,
                        depth=discriminator_encoder_depth,
                        learnable_attention=learnable_attention,
                        attention_type=attention_type,
                        attention_reduction_type=attention_reduction_type,
                        attention_dim=attention_dim,
                        **discriminator_encoder_kwargs)
                in_discriminator_channel = D_encoder.out_dim

                tmp = discriminator_decoder_type.split('.')
                module_path = '.'.join(tmp[:-1])
                class_name = tmp[-1]
                net_D = getattr(importlib.import_module(module_path),
                                class_name)(
                                    #net_D = paramnet.TreeGANParamNet(
                                    in_channel=in_discriminator_channel,
                                    param_dim=1,
                                    layer_depth=discriminator_depth,
                                    **discriminator_decoder_kwargs)
                return D_encoder, net_D

            self.D_encoder, self.net_D = init_discriminator()

        self.model_groups = defaultdict(lambda: [])
        self.parameter_groups = defaultdict(lambda: [])

        for key, name_list in model_optimizer_group_names.items():
            if key == 'discriminator' and not self.is_adversarial:
                continue
            for name in name_list:
                if name in skip_learning_models:
                    continue
                if name == 'paramnet' and self.use_motion_shape_integrated_decoder:
                    continue
                model = getattr(self, name)
                self.model_groups[key].append(model)
                self.parameter_groups[key].extend(model.parameters())

    def forward(self,
                inputs,
                coord=None,
                mode='generator',
                mask=None,
                return_param=True,
                direct_input_to_decoder=False,
                direct_input_to_D=False,
                return_occupancy=True,
                generator_kwargs={},
                paramnet_kwargs={}):
        if mode == 'generator':
            ret = {}
            if direct_input_to_decoder:
                assert inputs.ndim == 2, inputs.ndim
                assert inputs.size(1) == self.latent_dim, (
                    self.total_latent_dim, self.latent_dim,
                    self.motion_latent_dim)
                z = inputs
            else:
                z_temp = self.encoder(inputs)
                if self.use_canonical_motion_and_shape_latent:
                    z = z_temp[:, :-(self.shape_template_latent_dim +
                                     self.motion_template_latent_dim)]
                    z_temp2 = z_temp[:, -(self.shape_template_latent_dim +
                                          self.motion_template_latent_dim):]
                    shape_template_latent = z_temp2[:, :self.
                                                    shape_template_latent_dim]
                    motion_template_latent = z_temp2[:, self.
                                                     shape_template_latent_dim:]
                    generator_kwargs.update(
                        dict(shape_template_latent=shape_template_latent))
                    if self.is_generator_input_motion_and_shape_template_latent:
                        generator_kwargs.update(
                            dict(
                                motion_template_latent=motion_template_latent))
                    if self.use_canonical_motion_from_paramnet_and_shape_from_generator:
                        paramnet_kwargs.update(
                            dict(
                                motion_template_latent=motion_template_latent))
                else:
                    z = z_temp
            if self.is_split_latent_dim_for_motion and not direct_input_to_decoder:
                motion_z = z[:, :self.splitted_latent_dim_for_motion]
                ret['motion_latent'] = motion_z
                if self.is_input_motion_latent_to_shape_decoder:
                    z = z
                else:
                    z = z[:, self.splitted_latent_dim_for_motion:]
            ret['latent'] = z
            if return_occupancy:
                if self.is_expand_rotation_for_euler_angle:
                    if z.size(0) != coord.size(0):
                        expanded_z = z.unsqueeze(1).expand(
                            -1,
                            self.expand_rotation_for_euler_angle_sample_num +
                            1, -1
                        ).contiguous().view(
                            z.size(0) *
                            (self.expand_rotation_for_euler_angle_sample_num +
                             1), z.size(-1))
                        expanded_generator_kwargs = {}
                        for key, value in generator_kwargs.items():
                            expanded_generator_kwargs[key] = value.unsqueeze(
                                1
                            ).expand(
                                -1,
                                self.expand_rotation_for_euler_angle_sample_num
                                + 1, -1).contiguous().view(
                                    value.size(0) *
                                    (self.
                                     expand_rotation_for_euler_angle_sample_num
                                     + 1), value.size(-1))
                    else:
                        expanded_z = z
                        expanded_generator_kwargs = generator_kwargs
                else:
                    expanded_z = z
                    expanded_generator_kwargs = generator_kwargs
                if self.decoder_type == 'imnet':
                    occ = self.generator(coord, expanded_z,
                                         **expanded_generator_kwargs)
                elif self.decoder_type == 'occnet':
                    if self.is_generator_return_dict:
                        gen_ret = self.generator(coord, None, expanded_z,
                                                 **expanded_generator_kwargs)
                        occ = gen_ret['occupancy']
                        if self.is_expand_rotation_for_euler_angle and coord.size(
                                0) == z.size(0):
                            cd = occ
                            cd = cd.unsqueeze(-3).expand(
                                -1,
                                self.expand_rotation_for_euler_angle_sample_num
                                + 1, -1, -1).contiguous().view(
                                    cd.size(0) *
                                    (self.
                                     expand_rotation_for_euler_angle_sample_num
                                     + 1), cd.size(1), cd.size(2))

                            occ = cd

                        if self.use_diff_decoder:
                            if self.is_expand_rotation_for_euler_angle and coord.size(
                                    0) == z.size(0):
                                cd = gen_ret['canonical_occupancy']
                                cd = cd.unsqueeze(-3).expand(
                                    -1, self.
                                    expand_rotation_for_euler_angle_sample_num
                                    + 1, -1, -1
                                ).contiguous().view(
                                    cd.size(0) *
                                    (self.
                                     expand_rotation_for_euler_angle_sample_num
                                     + 1), cd.size(1), cd.size(2))

                                gen_ret['canonical_occupancy'] = cd

                            ret['canonical_occupancy'] = gen_ret[
                                'canonical_occupancy']
                        if self.use_canonical_location_from_generator:
                            if self.is_expand_rotation_for_euler_angle and coord.size(
                                    0) == z.size(0):
                                cd = gen_ret['canonical_location']
                                cd = cd.unsqueeze(-3).expand(
                                    -1, self.
                                    expand_rotation_for_euler_angle_sample_num
                                    + 1, -1, -1
                                ).contiguous().view(
                                    cd.size(0) *
                                    (self.
                                     expand_rotation_for_euler_angle_sample_num
                                     + 1), cd.size(1), cd.size(2))

                                gen_ret['canonical_location'] = cd
                            if not self.skip_apply_tanh_to_canonical_location:
                                canonical_location = torch.tanh(
                                    gen_ret['canonical_location']) * 0.5
                            else:
                                canonical_location = gen_ret[
                                    'canonical_location']

                            ret['canonical_location'] = canonical_location
                        if self.use_canonical_direction_from_generator:
                            if self.is_skip_direction_normalization_in_canonical_direction_from_generator:
                                canonical_direction = gen_ret[
                                    'canonical_direction']
                            elif self.is_decode_canonical_direction_as_rot_matrix:
                                if self.is_expand_rotation_for_euler_angle and coord.size(
                                        0) == z.size(0):
                                    cd = gen_ret['canonical_direction']
                                    cd = cd.unsqueeze(-3).expand(
                                        -1, self.
                                        expand_rotation_for_euler_angle_sample_num
                                        + 1, -1, -1
                                    ).contiguous().view(
                                        cd.size(0) *
                                        (self.
                                         expand_rotation_for_euler_angle_sample_num
                                         + 1), cd.size(1), cd.size(2))

                                    gen_ret['canonical_direction'] = cd

                                rot_matrix = self.decode_R(
                                    gen_ret['canonical_direction'],
                                    decode_R_type=self.
                                    canonical_direction_decode_type)
                                rot_matrix = torch.matmul(
                                    self.decode_init_canonical_direction(
                                        self.
                                        canonical_direction_init_directions,
                                        device=rot_matrix.device,
                                        dtype=rot_matrix.dtype), rot_matrix)
                                canonical_direction = torch.zeros(
                                    rot_matrix.shape[:3],
                                    dtype=rot_matrix.dtype,
                                    device=rot_matrix.device)
                                canonical_direction[:, :, 2] = 1.
                                canonical_direction = torch.matmul(
                                    rot_matrix,
                                    canonical_direction.unsqueeze(-1)).squeeze(
                                        -1)
                                ret['canonical_rotation_matrix'] = rot_matrix
                            else:
                                canonical_direction = F.normalize(
                                    gen_ret['canonical_direction'], dim=-1)
                            ret['canonical_direction'] = canonical_direction
                        if 'latent_quantize_diff' in gen_ret:
                            ret['latent_quantize_diff'] = gen_ret[
                                'latent_quantize_diff']
                        if 'surface_points' in gen_ret:
                            ret['surface_points'] = gen_ret['surface_points']
                        if 'quantize_indices' in gen_ret:
                            ret['shape_quantize_indices'] = gen_ret[
                                'quantize_indices']
                    else:
                        occ = self.generator(coord, None, z,
                                             **generator_kwargs)
                # occ[:, :, 1:] = occ[:, :, 1:] * 100
                ret['occupancy'] = occ
            else:
                ret['occupancy'] = None
            ret['generator_kwargs'] = generator_kwargs
            if return_param:
                assert not direct_input_to_decoder
                if self.use_motion_shape_integrated_decoder:
                    if not return_occupancy:
                        gen_ret = self.generator(coord, None, z)
                    raw_param = gen_ret['param']
                else:
                    if self.is_split_latent_dim_for_motion:
                        raw_param = self.paramnet(motion_z, **paramnet_kwargs)
                    else:
                        raw_param = self.paramnet(z, **paramnet_kwargs)
                ret['raw_param'] = raw_param
                ret['param'] = self.shape_param(raw_param)
                if isinstance(raw_param,
                              dict) and 'latent_quantize_diff' in raw_param:
                    latent_quantize_diff = ret.get(
                        'latent_quantize_diff',
                        0) + raw_param['latent_quantize_diff']
                    ret['latent_quantize_diff'] = latent_quantize_diff

            else:
                ret['param'] = None

            if self.is_expand_rotation_for_euler_angle_decay:
                self.is_expand_rotation_for_euler_angle_width_angle = max(
                    1, self.is_expand_rotation_for_euler_angle_width_angle -
                    self.expand_rotation_for_euler_angle_decay_per_iter)
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
            else:
                return {}
        elif mode == 'get_act':
            z = self.encoder(inputs)
            ret = self.generator(coord, z, return_act=True)
            return ret
        elif mode.startswith('occupancy_points_encoder'):
            x = self.D_encoder(inputs, mask=mask)
            x = F.normalize(x, p=2, dim=1)
            ret = {'latent_normalized': x}
            return ret
        else:
            raise NotImplementedError

    def shape_param(self, raw_param):

        ret = {}

        if self.paramnet_version == 'v2':
            if self.param_type == 'only_amount_as_matrix_loc_offset_canonical_motion':
                canonical_rot_matrix = self.decode_R(
                    raw_param['canonical_rotation_matrix'])
                ret['canonical_rotation_matrix'] = canonical_rot_matrix
                canonical_direction = torch.zeros(
                    canonical_rot_matrix.shape[:3],
                    dtype=canonical_rot_matrix.dtype,
                    device=canonical_rot_matrix.device)
                canonical_direction[:, :, 2] = 1.
                canonical_direction = torch.matmul(
                    canonical_rot_matrix,
                    canonical_direction.unsqueeze(-1)).squeeze(-1)
                ret['rotation_direction'] = canonical_direction[:, 1:1 + self.
                                                                rotation_primitive_num, :]

                rot_matrix_z_dim2 = self.decode_R(raw_param['rotation_matrix'],
                                                  dim=2)
                rot_matrix = torch.zeros(rot_matrix_z_dim2.size(0),
                                         rot_matrix_z_dim2.size(1),
                                         3,
                                         3,
                                         dtype=rot_matrix_z_dim2.dtype,
                                         device=rot_matrix_z_dim2.device)
                rot_matrix[..., -1, -1] = 1
                rot_matrix[..., :2, :2] = rot_matrix_z_dim2

                ret['rotation_matrix'] = torch.matmul(
                    canonical_rot_matrix[:, 1:1 + self.rotation_primitive_num,
                                         ...], rot_matrix)

                ret['rotation_amount'] = torch.acos(
                    rot_matrix[:, :, 0, 0].unsqueeze(-1)) / np.pi
                ret['rotation_scale'] = torch.ones_like(ret['rotation_amount'])

                ret['translation_direction'] = canonical_direction[:, 1 + self.
                                                                   rotation_primitive_num:, :]
                ret['translation_amount'] = raw_param['translation_amount']
                ret['translation_scale'] = torch.ones_like(
                    ret['translation_amount'])

                ret['canonical_location'] = raw_param[
                    'canonical_location'] + raw_param[
                        'canonical_location_offset']

            elif self.param_type in [
                    'only_amount_as_matrix_loc_offset_canonical_motion_euler_angle',
                    'only_amount_as_matrix_loc_offset_canonical_motion_euler_angle_screw',
                    'only_amount_as_matrix_loc_offset_canonical_motion_euler_angle_plus_direction_offset'
            ]:
                if self.param_type == 'only_amount_as_matrix_loc_offset_canonical_motion_euler_angle_screw':
                    if self.bound_amount_to_pi_and_one:
                        rotation_amount = (
                            ((torch.tanh(raw_param['motion_time']) + 1.) / 2.)
                        ) * torch.sigmoid(raw_param['motion_spectrum']) * np.pi
                    else:
                        rotation_amount = raw_param[
                            'motion_time'] * torch.sigmoid(
                                raw_param['motion_spectrum'])
                else:
                    if self.bound_amount_to_pi_and_one:
                        rotation_amount = (
                            (torch.tanh(raw_param['rotation_amount']) + 1.) /
                            2.) * np.pi
                    else:
                        rotation_amount = raw_param['rotation_amount']

                batch_size = rotation_amount.size(0)
                expanded_batch_size = batch_size * (
                    self.expand_rotation_for_euler_angle_sample_num + 1)
                if self.is_expand_rotation_for_euler_angle:
                    rand_val = torch.rand_like(
                        rotation_amount.unsqueeze(-3).expand(
                            -1,
                            self.expand_rotation_for_euler_angle_sample_num,
                            -1, -1))
                    rand_val = torch.cat([
                        torch.zeros_like(rotation_amount.unsqueeze(-3)),
                        rand_val
                    ],
                                         dim=-3)
                    rotation_amount_expand_amount = self.is_expand_rotation_for_euler_angle_width_angle / 180 * np.pi * (
                        rand_val - 0.5) * 2

                    rotation_amount = rotation_amount.unsqueeze(
                        -3) + rotation_amount_expand_amount
                    rotation_amount = rotation_amount.contiguous().view(
                        expanded_batch_size, rotation_amount.size(-2),
                        rotation_amount.size(-1))
                ret['rotation_amount'] = rotation_amount

                rot_matrix_z_dim2 = self.decode_R(
                    rotation_amount,
                    dim=2,
                    decode_R_type=self.rotation_decode_type)
                rot_matrix = torch.zeros(rot_matrix_z_dim2.size(0),
                                         rot_matrix_z_dim2.size(1),
                                         3,
                                         3,
                                         dtype=rot_matrix_z_dim2.dtype,
                                         device=rot_matrix_z_dim2.device)
                rot_matrix[..., -1, -1] = 1
                rot_matrix[..., :2, :2] = rot_matrix_z_dim2

                cd = raw_param['canonical_direction']
                if self.param_type in [
                        'only_amount_as_matrix_loc_offset_canonical_motion_euler_angle_plus_direction_offset',
                        'only_amount_as_matrix_loc_offset_canonical_motion_euler_angle_screw'
                ] and 'canonical_direction_offset' in raw_param:
                    cd = cd + raw_param['canonical_direction_offset']
                if self.is_expand_rotation_for_euler_angle:
                    cd = cd.unsqueeze(-3).expand(
                        -1,
                        self.expand_rotation_for_euler_angle_sample_num + 1,
                        -1, -1).contiguous().view(expanded_batch_size,
                                                  cd.size(1), cd.size(2))
                ret['canonical_direction'] = cd

                canonical_rot_matrix = self.decode_R(
                    cd, decode_R_type=self.rotation_decode_type)

                canonical_rot_matrix = torch.matmul(
                    self.decode_init_canonical_direction(
                        self.canonical_direction_init_directions,
                        device=canonical_rot_matrix.device,
                        dtype=canonical_rot_matrix.dtype),
                    canonical_rot_matrix)
                ret['canonical_rotation_matrix'] = canonical_rot_matrix
                canonical_direction = torch.zeros(
                    canonical_rot_matrix.shape[:3],
                    dtype=canonical_rot_matrix.dtype,
                    device=canonical_rot_matrix.device)
                canonical_direction[:, :, 2] = 1.
                canonical_direction = torch.matmul(
                    canonical_rot_matrix,
                    canonical_direction.unsqueeze(-1)).squeeze(-1)
                ret['rotation_direction'] = canonical_direction[:, 1:1 + self.
                                                                rotation_primitive_num, :]

                ret['rotation_scale'] = torch.ones_like(ret['rotation_amount'])

                ret['rotation_matrix'] = torch.matmul(
                    canonical_rot_matrix[:, 1:1 + self.rotation_primitive_num,
                                         ...], rot_matrix)

                if self.param_type in [
                        'only_amount_as_matrix_loc_offset_canonical_motion_euler_angle_screw'
                ]:
                    ret['translation_direction'] = ret['rotation_direction']
                else:
                    ret['translation_direction'] = canonical_direction[:, 1 +
                                                                       self.
                                                                       rotation_primitive_num:, :]
                if self.param_type in [
                        'only_amount_as_matrix_loc_offset_canonical_motion_euler_angle_screw'
                ]:
                    if self.bound_amount_to_pi_and_one:
                        ta = (
                            (torch.tanh(raw_param['motion_time']) + 1) / 2.
                        ) * (1 - torch.sigmoid(raw_param['motion_spectrum']))
                    else:
                        ta = raw_param['motion_time'] * (
                            1 - torch.sigmoid(raw_param['motion_spectrum']))
                else:
                    if self.bound_amount_to_pi_and_one:
                        ta = (
                            (torch.tanh(raw_param['translation_amount']) + 1) /
                            2.)
                    else:
                        ta = raw_param['translation_amount']
                if self.is_expand_rotation_for_euler_angle:
                    ta = ta.unsqueeze(-3).expand(
                        -1,
                        self.expand_rotation_for_euler_angle_sample_num + 1,
                        -1, -1).contiguous().view(expanded_batch_size,
                                                  ta.size(1), ta.size(2))
                ret['translation_amount'] = ta
                ret['translation_scale'] = torch.ones_like(
                    ret['translation_amount'])

                cl = raw_param['canonical_location'] + raw_param[
                    'canonical_location_offset']
                if self.is_expand_rotation_for_euler_angle:
                    cl = cl.unsqueeze(-3).expand(
                        -1,
                        self.expand_rotation_for_euler_angle_sample_num + 1,
                        -1, -1).contiguous().view(expanded_batch_size,
                                                  cl.size(1), cl.size(2))
                if self.skip_apply_tanh_to_canonical_location:
                    ret['canonical_location'] = cl
                else:
                    ret['canonical_location'] = torch.tanh(cl) * 0.5

                if 'quantize_indices' in raw_param:
                    ret['motion_quantize_indices'] = raw_param[
                        'quantize_indices']

            elif self.param_type == 'only_amount_as_matrix_loc_offset_canonical_motion_euler_angle_multiple_quantized_encoder':
                rotation_amount = raw_param['rotation_amount']
                batch_size = rotation_amount.size(0)
                expanded_batch_size = batch_size * (
                    self.expand_rotation_for_euler_angle_sample_num + 1)
                if self.is_expand_rotation_for_euler_angle:
                    rand_val = torch.rand_like(
                        rotation_amount.unsqueeze(-3).expand(
                            -1,
                            self.expand_rotation_for_euler_angle_sample_num,
                            -1, -1))
                    rand_val = torch.cat([
                        torch.zeros_like(rotation_amount.unsqueeze(-3)),
                        rand_val
                    ],
                                         dim=-3)
                    rotation_amount_expand_amount = self.is_expand_rotation_for_euler_angle_width_angle / 180 * np.pi * (
                        rand_val - 0.5) * 2

                    rotation_amount = rotation_amount.unsqueeze(
                        -3) + rotation_amount_expand_amount
                    rotation_amount = rotation_amount.contiguous().view(
                        expanded_batch_size, rotation_amount.size(-2),
                        rotation_amount.size(-1))
                ret['rotation_amount'] = rotation_amount

                rot_matrix_z_dim2 = self.decode_R(
                    rotation_amount,
                    dim=2,
                    decode_R_type=self.rotation_decode_type)
                rot_matrix = torch.zeros(rot_matrix_z_dim2.size(0),
                                         rot_matrix_z_dim2.size(1),
                                         3,
                                         3,
                                         dtype=rot_matrix_z_dim2.dtype,
                                         device=rot_matrix_z_dim2.device)
                rot_matrix[..., -1, -1] = 1
                rot_matrix[..., :2, :2] = rot_matrix_z_dim2

                cd = raw_param['canonical_direction']
                if self.is_expand_rotation_for_euler_angle:
                    cd = cd.unsqueeze(-3).expand(
                        -1,
                        self.expand_rotation_for_euler_angle_sample_num + 1,
                        -1, -1).contiguous().view(expanded_batch_size,
                                                  cd.size(1), cd.size(2))
                ret['canonical_direction'] = cd

                canonical_rot_matrix = self.decode_R(
                    cd, decode_R_type=self.rotation_decode_type)

                canonical_rot_matrix = torch.matmul(
                    self.decode_init_canonical_direction(
                        self.canonical_direction_init_directions,
                        device=canonical_rot_matrix.device,
                        dtype=canonical_rot_matrix.dtype),
                    canonical_rot_matrix)
                ret['canonical_rotation_matrix'] = canonical_rot_matrix
                canonical_direction = torch.zeros(
                    canonical_rot_matrix.shape[:3],
                    dtype=canonical_rot_matrix.dtype,
                    device=canonical_rot_matrix.device)
                canonical_direction[:, :, 2] = 1.
                canonical_direction = torch.matmul(
                    canonical_rot_matrix,
                    canonical_direction.unsqueeze(-1)).squeeze(-1)
                ret['rotation_direction'] = canonical_direction[:, 1:1 + self.
                                                                rotation_primitive_num, :]

                ret['rotation_scale'] = torch.ones_like(ret['rotation_amount'])

                ret['rotation_matrix'] = torch.matmul(
                    canonical_rot_matrix[:, 1:1 + self.rotation_primitive_num,
                                         ...], rot_matrix)

                ret['translation_direction'] = canonical_direction[:, 1 + self.
                                                                   rotation_primitive_num:, :]
                ta = raw_param['translation_amount']
                if self.is_expand_rotation_for_euler_angle:
                    ta = ta.unsqueeze(-3).expand(
                        -1,
                        self.expand_rotation_for_euler_angle_sample_num + 1,
                        -1, -1).contiguous().view(expanded_batch_size,
                                                  ta.size(1), ta.size(2))
                ret['translation_amount'] = ta
                ret['translation_scale'] = torch.ones_like(
                    ret['translation_amount'])

                cl = raw_param['canonical_location'] + raw_param[
                    'canonical_location_offset']
                if self.is_expand_rotation_for_euler_angle:
                    cl = cl.unsqueeze(-3).expand(
                        -1,
                        self.expand_rotation_for_euler_angle_sample_num + 1,
                        -1, -1).contiguous().view(expanded_batch_size,
                                                  cl.size(1), cl.size(2))
                if self.skip_apply_tanh_to_canonical_location:
                    ret['canonical_location'] = cl
                else:
                    ret['canonical_location'] = torch.tanh(cl) * 0.5
            else:
                raise NotImplementedError

        elif self.paramnet_version == 'v1':
            if self.is_primitive_wise_motion_output:
                raw_param = raw_param.mean(1)
            if self.motion_decoding_type == 'default':
                start_pos = 0

                if self.is_primitive_wise_motion_output:
                    raw_param = raw_param[:, 1:, :]
                raw_param = raw_param.reshape(
                    raw_param.size(0), self.primitive_num - 1,
                    sum(list(self.param_dims.values())))
                for key, value in self.param_dims.items():
                    end_pos = start_pos + value
                    ret[key] = raw_param[..., start_pos:end_pos]
                    if self.param_type == 'default':
                        if key == 'translation_direction':
                            ret[key] = F.normalize(ret[key], dim=-1)
                        elif key == 'rotation_direction':
                            if self.decode_rotation_axis_type == 'axis_direction_classification':
                                ret[key] = self.decode_direction_classification(
                                    ret[key])
                            ret[key] = F.normalize(ret[key], dim=-1)
                        elif key.endswith('_amount'):
                            if self.disable_sigmoid_for_amount:
                                ret[key] = ret[key]
                            else:
                                ret[key] = torch.sigmoid(ret[key])
                        elif key == 'rotation_scale':
                            ret[key] = torch.sigmoid(ret[key]) * np.pi
                        elif key == 'translation_scale':
                            ret[key] = torch.sigmoid(ret[key]) * 0.5
                        elif key == 'rotation_plane_pos':
                            ret[key] = torch.tanh(ret[key]) * 0.5
                    else:
                        raise NotImplementedError
            elif self.motion_decoding_type in [
                    'one_joint_type_per_primitive',
                    'one_joint_type_per_primitive_rot_pm_num_specified'
            ]:
                if self.is_primitive_wise_motion_output:
                    translation_raw_param = raw_param[:, (
                        1 + self.rotation_primitive_num):, self.rotation_dim:]
                else:
                    translation_raw_param = raw_param[:, :self.
                                                      translation_primitive_num
                                                      * self.
                                                      translation_dim].reshape(
                                                          -1, self.
                                                          translation_primitive_num,
                                                          self.translation_dim)

                start_pos = 0
                for key, value in self.translation_param_dims.items():
                    end_pos = start_pos + value
                    ret[key] = translation_raw_param[..., start_pos:end_pos]
                    if self.param_type == 'default':
                        if key.endswith('_direction'):
                            ret[key] = F.normalize(ret[key], dim=-1)
                        elif key.endswith('_amount'):
                            ret[key] = torch.sigmoid(ret[key])
                        elif key == 'translation_scale':
                            ret[key] = torch.sigmoid(ret[key]) * 0.5
                    elif self.param_type == 'affine':
                        if key.endswith('R'):
                            ret[key] = self.decode_R(ret[key])
                        elif key.endswith('T'):
                            ret[key] = torch.tanh(ret[key]) * 0.5
                        elif key.endswith('amount'):
                            if self.disable_sigmoid_for_amount:
                                ret[key] = ret[key]
                            else:
                                ret[key] = torch.sigmoid(ret[key])
                    elif self.param_type.startswith('motion_separate_affine'):
                        if key == 'translation_vector':
                            ret[key] = torch.tanh(ret[key]) * 0.5
                            ret['translation_direction'] = F.normalize(
                                ret[key], dim=-1)
                            ret['translation_amount'] = torch.norm(ret[key],
                                                                   dim=-1)
                            ret['translation_scale'] = torch.ones_like(
                                ret['translation_amount'])
                    elif self.param_type.startswith('only_amount'):
                        if key.endswith('amount'):
                            if self.disable_sigmoid_for_amount:
                                ret[key] = ret[key]
                            else:
                                ret[key] = torch.sigmoid(ret[key])
                            if self.is_expand_rotation_for_euler_angle:
                                ret[key] = ret[key].unsqueeze(-3).expand(
                                    -1, self.
                                    expand_rotation_for_euler_angle_sample_num
                                    + 1, -1, -1
                                ).contiguous().view(
                                    ret[key].size(0) *
                                    (self.
                                     expand_rotation_for_euler_angle_sample_num
                                     + 1), ret[key].size(1), ret[key].size(2))

                            ret['translation_scale'] = torch.ones_like(
                                ret['translation_amount'])
                        if key.endswith('loc_offset'):
                            ret[key] = ret[key]
                            if self.is_expand_rotation_for_euler_angle:
                                ret[key] = ret[key].unsqueeze(-3).expand(
                                    -1, self.
                                    expand_rotation_for_euler_angle_sample_num
                                    + 1, -1, -1
                                ).contiguous().view(
                                    ret[key].size(0) *
                                    (self.
                                     expand_rotation_for_euler_angle_sample_num
                                     + 1), ret[key].size(1), ret[key].size(2))
                    else:
                        raise NotImplementedError

                if self.is_primitive_wise_motion_output:
                    rotation_raw_param = raw_param[:, 1:(
                        1 + self.rotation_primitive_num), :self.rotation_dim]
                else:
                    rotation_raw_param = raw_param[:, self.
                                                   translation_primitive_num *
                                                   self.
                                                   translation_dim:].reshape(
                                                       -1, self.
                                                       rotation_primitive_num,
                                                       self.rotation_dim)
                start_pos = 0
                for key, value in self.rotation_param_dims.items():
                    end_pos = start_pos + value
                    ret[key] = rotation_raw_param[..., start_pos:end_pos]
                    if self.param_type == 'default':
                        if key.endswith('_direction'):
                            if self.decode_rotation_axis_type == 'axis_direction_classification':
                                ret[key] = self.decode_direction_classification(
                                    ret[key])
                            ret[key] = F.normalize(ret[key], dim=-1)
                        elif key.endswith('_amount'):
                            if self.disable_sigmoid_for_amount:
                                ret[key] = ret[key]
                            else:
                                ret[key] = torch.sigmoid(ret[key])
                        elif key == 'rotation_scale':
                            ret[key] = torch.sigmoid(ret[key]) * np.pi
                        elif key == 'rotation_plane_pos':
                            ret[key] = torch.tanh(ret[key]) * 0.5
                    elif self.param_type == 'affine':
                        if key.endswith('R'):
                            ret[key] = self.decode_R(ret[key])
                        elif key.endswith('T'):
                            ret[key] = torch.tanh(ret[key]) * 0.5
                        elif key.endswith('amount'):
                            if self.disable_sigmoid_for_amount:
                                ret[key] = ret[key]
                            else:
                                ret[key] = torch.sigmoid(ret[key])
                    elif self.param_type == 'motion_separate_affine_rotation_quaternion':
                        if key == 'rotation_vector':
                            axis, angle = geometry.convert_axis_angle_from_quaternion(
                                ret[key])
                            normalized_angle = angle / np.pi
                            ret['rotation_direction'] = axis
                            ret['rotation_amount'] = normalized_angle
                            ret['rotation_scale'] = torch.ones_like(
                                ret['rotation_amount'])
                    elif self.param_type == 'only_amount':
                        if key.endswith('amount'):
                            if self.disable_sigmoid_for_amount:
                                ret[key] = ret[key]
                            else:
                                ret[key] = torch.sigmoid(ret[key])
                            ret['rotation_scale'] = torch.ones_like(
                                ret['rotation_amount'])
                    elif self.param_type.startswith('only_amount_as_matrix'):
                        if key.endswith('amount'):

                            rotation_amount = ret[key]
                            if self.is_expand_rotation_for_euler_angle:
                                rand_val = torch.rand_like(
                                    rotation_amount.unsqueeze(-3).expand(
                                        -1, self.
                                        expand_rotation_for_euler_angle_sample_num,
                                        -1, -1))
                                rand_val = torch.cat([
                                    torch.zeros_like(
                                        rotation_amount.unsqueeze(-3)),
                                    rand_val
                                ],
                                                     dim=-3)
                                rotation_amount_expand_amount = self.is_expand_rotation_for_euler_angle_width_angle / 180 * np.pi * (
                                    rand_val - 0.5) * 2

                                rotation_amount = rotation_amount.unsqueeze(
                                    -3) + rotation_amount_expand_amount
                                rotation_amount = rotation_amount.contiguous(
                                ).view(
                                    ret[key].size(0) *
                                    (self.
                                     expand_rotation_for_euler_angle_sample_num
                                     + 1), ret[key].size(-2),
                                    ret[key].size(-1))
                            rot_matrix_z_dim2 = self.decode_R(
                                rotation_amount,
                                dim=2,
                                decode_R_type=self.rotation_decode_type)
                            rot_matrix = torch.zeros(
                                rot_matrix_z_dim2.size(0),
                                rot_matrix_z_dim2.size(1),
                                3,
                                3,
                                dtype=rot_matrix_z_dim2.dtype,
                                device=rot_matrix_z_dim2.device)
                            rot_matrix[..., -1, -1] = 1
                            rot_matrix[..., :2, :2] = rot_matrix_z_dim2

                            ret['rotation_matrix'] = rot_matrix
                            """
                            if self.is_expand_rotation_for_euler_angle:
                                ret['rotation_amount'] = torch.acos(
                                    rot_matrix.view(
                                        ret[key].size(0), self.
                                        is_expand_rotation_for_euler_angle_width_angle,
                                        ret[key].size(-2), 3,
                                        3)[:, 0, :, 0,
                                           0].unsqueeze(-1)) / np.pi

                            else:
                            """
                            ret['rotation_amount'] = torch.acos(
                                rot_matrix[:, :, 0, 0].unsqueeze(-1)) / np.pi
                            if self.disable_sigmoid_for_amount:
                                ret[key] = ret[key]
                            else:
                                ret[key] = torch.sigmoid(ret[key])
                            ret['rotation_scale'] = torch.ones_like(
                                ret['rotation_amount'])
                        if key.endswith('loc_offset'):
                            ret[key] = ret[key]
                            if self.is_expand_rotation_for_euler_angle:
                                ret[key] = ret[key].unsqueeze(-3).expand(
                                    -1, self.
                                    expand_rotation_for_euler_angle_sample_num
                                    + 1, -1, -1
                                ).contiguous().view(
                                    ret[key].size(0) *
                                    (self.
                                     expand_rotation_for_euler_angle_sample_num
                                     + 1), ret[key].size(1), ret[key].size(2))
                    else:
                        raise NotImplementedError

            if self.param_type == 'default':
                # Get anchor point
                rotation_direction = ret['rotation_direction']
                # B, pm, dim - 1
                rot_plane_pos = ret['rotation_plane_pos']
                ret['rotation_anchor_point'] = self.get_anchor_point(
                    rotation_direction, rot_plane_pos)
        return ret

    def get_anchor_point(self, rotation_direction, rot_plane_pos):
        if self.anchor_point_estimation_type == 'direct_regression':
            return torch.tanh(rot_plane_pos) * 0.5
        if self.anchor_point_estimation_type == 'direct_regression_fixed':
            return rot_plane_pos
        if self.anchor_point_estimation_type == 'sphere_regression':
            z_axis = torch.zeros_like(rotation_direction)
            # B, pm, dim
            z_axis[..., -1] = 1
            pos3d = torch.cat([
                rot_plane_pos,
                torch.zeros(rot_plane_pos.shape[:-1],
                            dtype=rot_plane_pos.dtype,
                            device=rot_plane_pos.device).unsqueeze(-1)
            ],
                              dim=-1)
            rotation_anchor_points = []
            for idx in range(self.rotation_primitive_num):
                z = z_axis[:, idx, :]
                d = rotation_direction[:, idx, :]
                mat = geometry.get_align_rotation_between_two(z, d)
                pos = pos3d[:, idx, :].unsqueeze(-1)
                anchor_point = torch.matmul(mat, pos).reshape(-1,
                                                              3).unsqueeze(1)
                rotation_anchor_points.append(anchor_point)
            return torch.cat(rotation_anchor_points, dim=1)

        elif self.anchor_point_estimation_type == 'default':
            z_axis = torch.zeros_like(rotation_direction)
            # B, pm, dim
            z_axis[..., -1] = 1
            anchor_axis_rotation_direction = torch.cross(
                z_axis, rotation_direction)
            anchor_axis_rotation_direction_normalized = F.normalize(
                anchor_axis_rotation_direction, dim=-1)
            # B, pm, 1
            anchor_axis_rotation_angle = torch.acos(
                F.cosine_similarity(z_axis, rotation_direction,
                                    dim=-1)).unsqueeze(-1)

            # B, pm, dim + 1
            quat = geometry.get_quaternion(
                anchor_axis_rotation_direction_normalized,
                anchor_axis_rotation_angle)

            # B, pm, dim
            rot_plane_pos_zero_padded = torch.cat([
                rot_plane_pos,
                torch.zeros(rot_plane_pos.shape[:-1],
                            dtype=rot_plane_pos.dtype,
                            device=rot_plane_pos.device).unsqueeze(-1)
            ],
                                                  dim=-1)
            # B, pm, dim
            return geometry.apply_3d_rotation(
                rot_plane_pos_zero_padded.unsqueeze(-2), quat).squeeze(-2)

    def init_param_dim(self):

        if self.param_type == 'default':
            axis = [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [-1, 1, 0],
                [-1, 0, 1],
                [0, -1, 1],
                [1, -1, 0],
                [1, 0, -1],
                [0, 1, -1],
                [-1, -1, 0],
                [-1, 0, -1],
                [0, -1, -1],
            ]
            axis = np.array(axis).astype(np.float32)
            self.rotation_direction_axis = torch.from_numpy(axis / np.sqrt(
                (axis**2).sum(-1, keepdims=True)))

            rotation_direction_dim = self.rotation_direction_axis.size(
                0
            ) if self.decode_rotation_axis_type == 'axis_direction_classification' else self.dim
            rotation_plane_pos_dim = self.dim if self.anchor_point_estimation_type in [
                'direct_regression', 'direct_regression_fixed'
            ] else self.dim - 1

            self.param_dims = OrderedDict({
                'rotation_direction': rotation_direction_dim,
                'rotation_plane_pos': rotation_plane_pos_dim,
                'rotation_amount': 1,
                'rotation_scale': 1,
                'translation_direction': self.dim,
                'translation_amount': 1,
                'translation_scale': 1,
            })
            self.translation_param_dims = {
                key: value
                for key, value in self.param_dims.items()
                if key.startswith('translation')
            }
            self.translation_dim = sum(
                list(self.translation_param_dims.values()))

            self.rotation_param_dims = {
                key: value
                for key, value in self.param_dims.items()
                if key.startswith('rotation')
            }
            self.rotation_dim = sum(list(self.rotation_param_dims.values()))

        elif self.param_type == 'affine':
            self.param_dims = OrderedDict({'R': 9, 'T': 3, 'amount': 1})
            self.translation_dim = sum(list(self.param_dims.values()))
            self.rotation_dim = sum(list(self.param_dims.values()))

        elif self.param_type == 'motion_separate_affine_rotation_quaternion':
            self.param_dims = OrderedDict({
                'rotation_vector': 4,
                'translation_vector': 3
            })
            self.translation_param_dims = {
                key: value
                for key, value in self.param_dims.items()
                if key.startswith('translation')
            }
            self.translation_dim = sum(
                list(self.translation_param_dims.values()))

            self.rotation_param_dims = {
                key: value
                for key, value in self.param_dims.items()
                if key.startswith('rotation')
            }
            self.rotation_dim = sum(list(self.rotation_param_dims.values()))

        elif self.param_type == 'only_amount':
            self.param_dims = OrderedDict({
                'rotation_amount': 1,
                'translation_amount': 1
            })
            self.translation_param_dims = {
                key: value
                for key, value in self.param_dims.items()
                if key.startswith('translation')
            }
            self.translation_dim = sum(
                list(self.translation_param_dims.values()))

            self.rotation_param_dims = {
                key: value
                for key, value in self.param_dims.items()
                if key.startswith('rotation')
            }
            self.rotation_dim = sum(list(self.rotation_param_dims.values()))

        elif self.param_type == 'only_amount_as_matrix':
            self.param_dims = OrderedDict({
                'rotation_amount': 4,
                'translation_amount': 1
            })
            self.translation_param_dims = {
                key: value
                for key, value in self.param_dims.items()
                if key.startswith('translation')
            }
            self.translation_dim = sum(
                list(self.translation_param_dims.values()))

            self.rotation_param_dims = {
                key: value
                for key, value in self.param_dims.items()
                if key.startswith('rotation')
            }
            self.rotation_dim = sum(list(self.rotation_param_dims.values()))

        elif self.param_type == 'only_amount_as_matrix_loc_offset':
            self.param_dims = OrderedDict({
                'rotation_amount': 4,
                'rotation_location_offset': 3,
                'translation_amount': 1,
                'translation_location_offset': 3,
            })
            self.translation_param_dims = {
                key: value
                for key, value in self.param_dims.items()
                if key.startswith('translation')
            }
            self.translation_dim = sum(
                list(self.translation_param_dims.values()))

            self.rotation_param_dims = {
                key: value
                for key, value in self.param_dims.items()
                if key.startswith('rotation')
            }
            self.rotation_dim = sum(list(self.rotation_param_dims.values()))
        elif self.param_type == 'only_amount_as_matrix_loc_offset_canonical_motion':
            self.paramnet_features = {
                'rotation_matrix': {
                    'out_dim': 4,
                    'groups': self.rotation_primitive_num
                },
                'canonical_location_offset': {
                    'out_dim': 3,
                    'groups': self.primitive_num
                },
                'translation_amount': {
                    'out_dim': 1,
                    'groups': self.translation_primitive_num
                },
                'canonical_rotation_matrix': {
                    'out_dim': 9,
                    'is_const': True,
                    'groups': self.primitive_num,
                    'quantize_num': 4
                },
                'canonical_location': {
                    'out_dim': 3,
                    'is_const': True,
                    'groups': self.primitive_num,
                    'quantize_num': 4
                }
            }
            self.param_dims = {}

        elif self.param_type in [
                'only_amount_as_matrix_loc_offset_canonical_motion_euler_angle',
                'only_amount_as_matrix_loc_offset_canonical_motion_euler_angle_plus_direction_offset'
        ]:
            self.paramnet_features = {
                'rotation_amount': {
                    'out_dim': 1,
                    'groups': self.rotation_primitive_num
                },
                'canonical_location_offset': {
                    'out_dim': 3,
                    'groups': self.primitive_num
                },
                'translation_amount': {
                    'out_dim': 1,
                    'groups': self.translation_primitive_num
                },
                'canonical_direction': {
                    'out_dim': 3,
                    'is_const': True,
                    'groups': self.primitive_num,
                    'quantize_num': 4
                },
                'canonical_location': {
                    'out_dim': 3,
                    'is_const': True,
                    'groups': self.primitive_num,
                    'quantize_num': 4
                }
            }
            if self.param_type == 'only_amount_as_matrix_loc_offset_canonical_motion_euler_angle_plus_direction_offset':
                self.paramnet_features.update({
                    'canonical_direction_offset': {
                        'out_dim': 3,
                        'groups': self.primitive_num
                    }
                })
        elif self.param_type in [
                'only_amount_as_matrix_loc_offset_canonical_motion_euler_angle_screw'
        ]:
            self.paramnet_features = {
                'motion_time': {
                    'out_dim': 1,
                    'groups': self.rotation_primitive_num
                },
                'canonical_location_offset': {
                    'out_dim': 3,
                    'groups': self.primitive_num
                },
                'canonical_direction_offset': {
                    'out_dim': 3,
                    'groups': self.primitive_num
                },
                'canonical_direction': {
                    'out_dim': 3,
                    'is_const': True,
                    'groups': self.primitive_num,
                    'quantize_num': 4
                },
                'canonical_location': {
                    'out_dim': 3,
                    'is_const': True,
                    'groups': self.primitive_num,
                    'quantize_num': 4
                },
                'motion_spectrum': {
                    'out_dim': 1,
                    'is_const': True,
                    'groups': self.rotation_primitive_num,
                    'quantize_num': 1
                },
            }
        elif self.param_type == 'only_amount_as_matrix_loc_offset_canonical_motion_euler_angle_multiple_quantized_encoder':
            self.paramnet_features = {
                'rotation_amount': {
                    'out_dim':
                    1,
                    'groups':
                    self.rotation_primitive_num *
                    self.canonical_direction_quantize_num
                },
                'canonical_location_offset': {
                    'out_dim': 3,
                    'groups': self.primitive_num
                },
                'translation_amount': {
                    'out_dim':
                    1,
                    'groups':
                    self.translation_primitive_num *
                    self.canonical_direction_quantize_num
                },
                'canonical_direction': {
                    'out_dim': 3,
                    'is_const': True,
                    'groups': self.primitive_num,
                    'quantize_num': self.canonical_direction_quantize_num,
                    'is_multiple_quantize': True
                },
                'canonical_location': {
                    'out_dim': 3,
                    'is_const': True,
                    'groups': self.primitive_num,
                    'quantize_num': 4
                }
            }
        #else:
        if self.paramnet_version in ['v1']:
            if self.motion_decoding_type == 'default':
                self.param_dim = sum(list(
                    self.param_dims.values())) * (self.primitive_num - 1)
            elif self.motion_decoding_type == 'one_joint_type_per_primitive':
                self.param_dim = sum(list(self.param_dims.values())) * int(
                    (self.primitive_num - 1) // 2)
            elif self.motion_decoding_type == 'one_joint_type_per_primitive_rot_pm_num_specified':
                self.param_dim = self.rotation_dim * self.rotation_primitive_num + self.translation_dim * self.translation_primitive_num

    def decode_direction_classification(self, axis_raw_weights):
        # B, pm, 18
        if self.axis_weighting_type == 'gumbel':
            weight = gumbel_softmax.gumbel_softmax(axis_raw_weights, dim=-1)
        elif self.axis_weighting_type == 'gumbel_hard':
            weight = gumbel_softmax.gumbel_softmax(axis_raw_weights,
                                                   dim=-1,
                                                   hard=True)
        elif self.axis_weighting_type == 'softmax':
            weight = F.softmax(axis_raw_weights, dim=-1)

        # B, pm, 18, 1
        weight = weight.unsqueeze(-1)

        # 18, 3
        axis = self.rotation_direction_axis.to(weight.device)
        # 1, 1, 18, 3
        axis = axis.unsqueeze(0).unsqueeze(0)

        # B, pm, 18, 3
        weighted_axis = axis * weight

        # B, pm, 3
        weighted_axis = weighted_axis.sum(-2)

        return weighted_axis

    def decode_R(self, raw_R, dim=3, decode_R_type='svd'):
        """Maps 9D input vectors onto SO(3) via symmetric orthogonalization.

        x: should have size [batch_size, 9]

        Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
        """
        batch_size, pm_num, odim = raw_R.shape
        if decode_R_type == 'svd':
            assert odim == dim * dim
            m = raw_R.contiguous().view(-1, dim, dim)
            u, s, v = torch.svd(m)
            vt = torch.transpose(v, 1, 2)
            det = torch.det(torch.matmul(u, vt))
            det = det.view(-1, 1, 1)
            vt = torch.cat((vt[:, :(dim - 1), :], vt[:, -1:, :] * det), 1)
            r = torch.matmul(u, vt)
            return r.contiguous().view(batch_size, pm_num, dim, dim)
        elif decode_R_type == 'euler_angle':
            if dim == 3:
                return geometry.compute_rotation_matrix_from_euler(
                    raw_R[..., :3])
            elif dim == 2:
                return geometry.compute_rotation_matrix_from_euler(
                    raw_R[..., 0].unsqueeze(-1))
            else:
                raise NotImplementedError

    def decode_init_canonical_direction(self, init_directions, device, dtype):
        if len(init_directions) == 0:
            rot_matrices = [
                torch.eye(3).type(dtype) for _ in range(self.primitive_num)
            ]
        else:
            rot_matrices = [torch.eye(3).type(dtype)]
            for direction_str in init_directions:
                if direction_str.startswith('y'):
                    if len(direction_str) > 1:
                        angle = int(direction_str[1:]) / 180 * np.pi
                    else:
                        angle = np.pi / 2
                    rot_matrix = trimesh.transformations.rotation_matrix(
                        angle, [1, 0, 0])[:3, :3]
                    rot_matrix = torch.from_numpy(rot_matrix).type(dtype)
                elif direction_str.startswith('x'):
                    if len(direction_str) > 1:
                        angle = int(direction_str[1:]) / 180 * np.pi
                    else:
                        angle = np.pi / 2
                    rot_matrix = trimesh.transformations.rotation_matrix(
                        angle, [0, 1, 0])[:3, :3]
                    rot_matrix = torch.from_numpy(rot_matrix).type(dtype)
                elif direction_str.startswith('z'):
                    if len(direction_str) > 1:
                        angle = int(direction_str[1:]) / 180 * np.pi
                    else:
                        angle = 0
                    rot_matrix = trimesh.transformations.rotation_matrix(
                        angle, [0, 1, 0])[:3, :3]
                    rot_matrix = torch.from_numpy(rot_matrix).type(dtype)
                else:
                    raise NotImplementedError
                rot_matrices.append(rot_matrix)
        return torch.stack(rot_matrices).to(device).unsqueeze(0)
