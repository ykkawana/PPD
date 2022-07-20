from trainer import imex_trainer
import torch
from utils import visualizer_util
import numpy as np
import torch.nn.functional as F
from utils import geometry
import random
from pytorch_metric_learning import losses as metric_losses
import numpy as np
from collections import defaultdict
from utils import train_util
import time
from loss import chamfer_loss
from model.imnet import gumbel_softmax
from utils import common

EPS = 1e-7


class Trainer(imex_trainer.Trainer):
    def __init__(
            self,
            model,
            optimizers,
            device,
            disable_induction_by_moving=True,
            disable_gan_training=True,
            learn_moving_self_seg_reg=False,
            self_supervised_seg_loss_weight=1.,
            disable_translation=False,
            model_input_type='voxel',
            anchor_position_loss_margin=0.01,
            anchor_position_loss_weight=1.,
            use_anchor_position_loss=False,
            use_self_supervised_canonical_learning=False,
            use_learn_generator_with_randomly_moved_shape=False,
            skip_learn_generator_with_recon_shape=False,
            use_volume_preserving_loss=False,
            volume_preserving_loss_weight=1,
            use_learn_only_generator_with_canonical_shape=False,
            disable_part_similarity_loss_in_discriminator=False,
            use_primitive_number_loss=False,
            primitive_number_loss_weight=1.,
            primitive_number_loss_margin=0.2,
            use_canonical_similarity_loss=False,
            canonical_similarity_loss_weight=1.0,
            part_shape_canonical_similarity_loss_type='TripletMarginLoss',
            canonical_triplet_margin_loss_margin=0.05,
            is_check_gradient_scale=False,
            use_gradient_penalty=False,
            apply_gradient_penalty_every=1,
            gradient_penalty_type='wgangp',
            is_apply_gradient_penalty_as_independent_step=False,
            is_constant_motion_range=False,
            use_pretrained_feature_canonical_loss=False,
            pretrained_feature_canonical_loss_weight=1.,
            which_pretrained_feature_canonical_loss_feat='feat_4',
            is_input_pm0_to_motion_net_too=False,
            is_scale_pm0_values_after_motion=False,
            scale_pm0_values_after_motion=1.,
            use_soft_volume_preserving_loss=False,
            soft_volume_preserving_loss_weight=1.,
            use_motion_gan_loss=False,
            use_adaptive_max_motion_range_sampling=False,
            use_moved_pretrained_feature_canonical_loss=False,
            is_freeze_generator_in_motion_gan_training=False,
            soft_volume_preserving_loss_type='default',
            use_anchor_position_near_gt_shape_loss=False,
            anchor_position_near_gt_shape_loss_weight=1.,
            use_self_supervised_motion_learning=False,
            self_supervised_motion_learning_line_distance_loss_weight=1.,
            self_supervised_motion_learning_line_direction_loss_weight=1.,
            self_supervised_motion_learning_line_rotation_loss_weight=1.,
            rotation_anchor_point_scale=1.,
            use_alternate_train_generator_and_paramnet=False,
            is_freeze_recon_loss_during_paramnet_training=False,
            train_paramnet_every=1.,
            train_generator_every=1.,
            use_explicit_pointcloud_loss_for_motion=False,
            explicit_pointcloud_loss_for_motion_weight=1.,
            is_detach_transformation_for_occupancy_reconstruction=False,
            is_sample_rotation_anchor_point_by_occupancy=False,
            sample_rotation_anchor_point_by_occupancy_type='gumbel',
            use_rotation_anchor_point_similarity_loss=False,
            rotation_anchor_point_similarity_loss_type='gaussian_logprob',
            rotation_anchor_point_similarity_loss_weight=1.,
            use_rotation_anchor_point_occupancy_relation_loss=False,
            rotation_anchor_point_occupancy_relation_loss_weight=1.,
            use_bone_occupancy_loss=False,
            bone_occupancy_loss_sampling_num=100,
            bone_occupancy_loss_weight=1.,
            use_canonical_location_near_gt_loss=False,
            canonical_location_near_gt_loss_weight=1,
            is_canonical_location_as_anchor_point=False,
            tsdf_loss_weight=1.,
            use_occ_recon_loss_for_occ_and_tsdf_for_param=False,
            occ_recon_loss_for_occ_and_tsdf_for_param_occinize_scale=1.,
            use_canonical_shape_for_gan_training=False,
            disable_detach_canonical_location_as_pred_values_in_generator=False,
            is_randomly_disable_rotation_per_part=False,
            is_continous_axis_angle_rotation=False,
            latent_quantize_diff_loss_weight=1.,
            use_recon_canonical_shape_loss=False,
            recon_canonical_shape_loss_weight=1.,
            sample_transform_param_rotation_max=np.pi / 2,
            sample_transform_param_translation_max=0.3,
            surface_quasi_sdf_points_per_point=8,
            surface_quasi_sdf_input_subsampling_num=None,
            surface_quasi_sdf_std=1e-2,
            use_surface_quasi_sdf_loss=False,
            surface_quasi_sdf_loss_weight=1.,
            is_correct_continuous_rotation=False,
            use_explicit_pointcloud_loss_for_motion_with_surface_points=False,
            explicit_pointcloud_loss_for_motion_with_surface_points_weight=1.,
            is_add_tie_breaking_noise_in_occ_recon=False,
            tie_breaking_noise_in_occ_recon_scale=0.02,
            use_weight_for_positive_values_for_implicit_reconstruction_loss=False,
            weight_for_positive_values_for_implicit_reconstruction_loss_threshold=0,
            weight_for_positive_values_for_implicit_reconstruction_loss_scale=1.,
            is_only_visualize_surface_points_prediction=False,
            is_apply_tanh_to_merged_canonical_location=False,
            use_motion_amount_inverse_std_loss=False,
            motion_amount_inverse_std_loss_weight=1,
            motion_amount_inverse_std_loss_rotation_threshold=0.,
            motion_amount_inverse_std_loss_translation_threshold=0.,
            motion_amount_inverse_std_loss_aggregation_type='min',
            use_moved_overlap_regularizer_loss=False,
            use_anchor_position_near_gt_shape_loss_near_primitive=False,
            use_anchor_position_near_gt_shape_loss_union_to_static_primitive=False,
            is_canonical_location_as_anchor_point_not_part_centric=False,
            use_randomly_moved_overlap_regularizer_loss=False,
            location_offset_regulariztion_loss_gausian_scale=0.03,
            location_offset_regulariztion_loss_weight=1.,
            use_location_offset_regulariztion_loss=False,
            use_canonical_diff_prior_recon_loss=False,
            use_minimize_raw_canonical_location_to_offset_loss=False,
            use_raw_anchor_position_near_gt_shape_loss=False,
            minimize_raw_canonical_location_to_offset_loss_weight=1,
            location_offset_regulariztion_loss_type='gaussian',
            minimize_raw_canonical_location_to_offset_loss_detach_canonical_location_grad=False,
            use_anchor_position_chain_assumption_loss=False,
            anchor_position_chain_assumption_loss_weight=1.0,
            minimize_raw_canonical_direction_to_offset_loss_weight=1.0,
            use_minimize_raw_canonical_direction_to_offset_loss=False,
            minimize_raw_canonical_direction_to_offset_loss_detach_canonical_direction_grad=False,
            use_screw_motion_spectrum_entropy_loss=False,
            screw_motion_spectrum_entropy_loss_weight=1.0,
            disable_canonical_location_for_translation=False,
            use_voxel_discriminator=False,
            anchor_position_chain_assumption_loss_only_self=False,
            anchor_position_chain_assumption_loss_threshold=None,
            is_sample_transform_param_random_additive_to_original=False,
            **kwargs):
        kwargs['disable_induction_by_moving'] = disable_induction_by_moving
        kwargs['disable_gan_training'] = disable_gan_training
        super().__init__(model, optimizers, device, **kwargs)
        assert self.train_implicit
        assert not self.is_movenet_classification_head
        assert not self.is_movenet_decode_sdf
        assert not self.use_acgan
        assert not self.use_canonical_self_supervised_slide_loss
        assert not self.use_canonical_occ_loss
        self.model_input_type = model_input_type
        assert self.model_input_type in ['voxel', 'surface_points']
        if self.learn_moving_self_slide_reg:
            assert self.model_input_type in ['surface_points']
        if self.model_input_type in ['voxel']:
            assert self.disable_gan_training
            assert self.disable_induction_by_moving
        self.self_supervised_seg_loss_weight = self_supervised_seg_loss_weight
        self.learn_moving_self_seg_reg = learn_moving_self_seg_reg
        self.disable_translation = disable_translation

        self.anchor_position_loss_margin = anchor_position_loss_margin
        self.anchor_position_loss_weight = anchor_position_loss_weight
        self.use_anchor_position_loss = use_anchor_position_loss

        self.use_self_supervised_canonical_learning = use_self_supervised_canonical_learning
        if self.use_self_supervised_canonical_learning:
            assert not self.disable_induction_by_moving
            assert not self.learn_moving_self_seg_reg
            assert not self.learn_moving_self_slide_reg
            assert not self.is_move_points
        self.use_learn_generator_with_randomly_moved_shape = use_learn_generator_with_randomly_moved_shape
        if self.use_learn_generator_with_randomly_moved_shape:
            assert self.use_self_supervised_canonical_learning
        self.skip_learn_generator_with_recon_shape = skip_learn_generator_with_recon_shape
        self.use_canonical_shape_for_gan_training = use_canonical_shape_for_gan_training
        if not self.disable_gan_training:
            assert (
                self.use_learn_generator_with_randomly_moved_shape
            ) or not self.skip_learn_generator_with_recon_shape or self.use_canonical_shape_for_gan_training
        if self.overlap_regularizer_type == 'non_top_primitives':
            assert self.occupancy_reduction_loss_type == 'occnet'
        self.use_volume_preserving_loss = use_volume_preserving_loss
        if self.use_volume_preserving_loss:
            assert self.occupancy_reduction_loss_type == 'occnet'
        self.volume_preserving_loss_weight = volume_preserving_loss_weight
        self.use_learn_only_generator_with_canonical_shape = use_learn_only_generator_with_canonical_shape
        self.disable_part_similarity_loss_in_discriminator = disable_part_similarity_loss_in_discriminator
        self.use_primitive_number_loss = use_primitive_number_loss
        if self.use_primitive_number_loss:
            assert self.occupancy_reduction_loss_type == 'occnet'
        self.primitive_number_loss_weight = primitive_number_loss_weight
        self.primitive_number_loss_margin = primitive_number_loss_margin
        self.use_canonical_similarity_loss = use_canonical_similarity_loss
        self.canonical_similarity_loss_weight = canonical_similarity_loss_weight
        self.part_shape_canonical_similarity_loss_type = part_shape_canonical_similarity_loss_type
        if self.part_shape_canonical_similarity_loss_type == 'TripletMarginLoss':
            self.part_shape_canonical_similarity_loss_func = metric_losses.TripletMarginLoss(
                margin=canonical_triplet_margin_loss_margin)
        else:
            raise NotImplementedError
        self.is_check_gradient_scale = is_check_gradient_scale
        self.use_gradient_penalty = use_gradient_penalty
        if self.use_gradient_penalty:
            assert self.use_self_supervised_canonical_learning
        if self.gan_type == 'wgangp' and not self.disable_gan_training:
            assert not self.use_discriminator_input_mask
            assert self.use_gradient_penalty
            assert not self.use_wgangp_latent_code_interpolation
        self.gradient_penalty_type = gradient_penalty_type
        assert self.gradient_penalty_type in ['wgangp', 'real_input']
        if self.gradient_penalty_type == 'wgangp' and self.use_gradient_penalty and not self.disable_gan_training:
            assert self.gan_type == 'wgangp'
        self.is_apply_gradient_penalty_as_independent_step = is_apply_gradient_penalty_as_independent_step
        if self.is_apply_gradient_penalty_as_independent_step:
            assert not self.gradient_penalty_type == 'wgangp'
        self.apply_gradient_penalty_every = apply_gradient_penalty_every
        if self.apply_gradient_penalty_every > 1:
            assert self.is_apply_gradient_penalty_as_independent_step
            assert not self.gan_type == 'wgangp'
        self.is_constant_motion_range = is_constant_motion_range
        self.use_pretrained_feature_canonical_loss = use_pretrained_feature_canonical_loss
        self.pretrained_feature_canonical_loss_weight = pretrained_feature_canonical_loss_weight
        if self.use_pretrained_feature_canonical_loss:
            assert self.model.use_diff_decoder
        self.which_pretrained_feature_canonical_loss_feat = which_pretrained_feature_canonical_loss_feat
        self.is_input_pm0_to_motion_net_too = is_input_pm0_to_motion_net_too
        if self.is_input_pm0_to_motion_net_too:
            assert self.use_self_supervised_canonical_learning
        self.is_scale_pm0_values_after_motion = is_scale_pm0_values_after_motion
        if self.is_scale_pm0_values_after_motion:
            assert self.use_self_supervised_canonical_learning
        self.scale_pm0_values_after_motion = scale_pm0_values_after_motion
        self.use_soft_volume_preserving_loss = use_soft_volume_preserving_loss
        self.soft_volume_preserving_loss_weight = soft_volume_preserving_loss_weight
        self.use_motion_gan_loss = use_motion_gan_loss
        self.use_adaptive_max_motion_range_sampling = use_adaptive_max_motion_range_sampling
        if self.use_motion_gan_loss:
            assert self.use_adaptive_max_motion_range_sampling
            assert self.use_learn_generator_with_randomly_moved_shape
            assert self.skip_learn_generator_with_recon_shape
        self.use_moved_pretrained_feature_canonical_loss = use_moved_pretrained_feature_canonical_loss
        self.is_freeze_generator_in_motion_gan_training = is_freeze_generator_in_motion_gan_training
        self.soft_volume_preserving_loss_type = soft_volume_preserving_loss_type
        assert self.soft_volume_preserving_loss_type in [
            'default', 'primitive_wise_mean'
        ]
        self.use_anchor_position_near_gt_shape_loss = use_anchor_position_near_gt_shape_loss
        self.anchor_position_near_gt_shape_loss_weight = anchor_position_loss_weight
        self.use_self_supervised_motion_learning = use_self_supervised_motion_learning
        if self.use_self_supervised_motion_learning:
            assert not self.model.use_canonical_location_from_generator
        self.self_supervised_motion_learning_line_distance_loss_weight = self_supervised_motion_learning_line_distance_loss_weight
        self.self_supervised_motion_learning_line_direction_loss_weight = self_supervised_motion_learning_line_direction_loss_weight
        self.self_supervised_motion_learning_line_rotation_loss_weight = self_supervised_motion_learning_line_rotation_loss_weight
        self.rotation_anchor_point_scale = rotation_anchor_point_scale

        self.use_alternate_train_generator_and_paramnet = use_alternate_train_generator_and_paramnet
        self.is_freeze_recon_loss_during_paramnet_training = is_freeze_recon_loss_during_paramnet_training
        self.train_paramnet_every = train_paramnet_every
        self.train_generator_every = train_generator_every
        if self.use_alternate_train_generator_and_paramnet:
            self.train_generator_every != self.train_paramnet_every

        self.use_explicit_pointcloud_loss_for_motion = use_explicit_pointcloud_loss_for_motion
        if self.use_explicit_pointcloud_loss_for_motion:
            assert self.occupancy_reduction_loss_type == 'occnet'
            assert self.model.decoder_type == 'occnet'
        self.explicit_pointcloud_loss_for_motion_weight = explicit_pointcloud_loss_for_motion_weight
        self.is_detach_transformation_for_occupancy_reconstruction = is_detach_transformation_for_occupancy_reconstruction

        self.is_sample_rotation_anchor_point_by_occupancy = is_sample_rotation_anchor_point_by_occupancy
        if self.is_sample_rotation_anchor_point_by_occupancy:
            assert self.model.motion_decoding_type in [
                'one_joint_type_per_primitive',
                'one_joint_type_per_primitive_rot_pm_num_specified'
            ]
        self.sample_rotation_anchor_point_by_occupancy_type = sample_rotation_anchor_point_by_occupancy_type
        assert self.sample_rotation_anchor_point_by_occupancy_type in [
            'gumbel', 'softmax'
        ]
        self.use_rotation_anchor_point_similarity_loss = use_rotation_anchor_point_similarity_loss
        self.rotation_anchor_point_similarity_loss_type = rotation_anchor_point_similarity_loss_type
        self.rotation_anchor_point_similarity_loss_weight = rotation_anchor_point_similarity_loss_weight
        self.use_rotation_anchor_point_occupancy_relation_loss = use_rotation_anchor_point_occupancy_relation_loss
        if self.use_rotation_anchor_point_occupancy_relation_loss:
            assert self.occupancy_reduction_loss_type == 'occnet'
            assert self.model.decoder_type == 'occnet'
        self.rotation_anchor_point_occupancy_relation_loss_weight = rotation_anchor_point_occupancy_relation_loss_weight
        self.use_bone_occupancy_loss = use_bone_occupancy_loss
        if self.use_bone_occupancy_loss:
            assert self.model.use_canonical_location_from_generator
        self.bone_occupancy_loss_sampling_num = bone_occupancy_loss_sampling_num
        self.bone_occupancy_loss_weight = bone_occupancy_loss_weight

        self.use_canonical_location_near_gt_loss = use_canonical_location_near_gt_loss
        self.canonical_location_near_gt_loss_weight = canonical_location_near_gt_loss_weight
        self.is_canonical_location_as_anchor_point = is_canonical_location_as_anchor_point
        if self.model.param_type == 'motion_separate_affine_rotation_quaternion':
            assert self.is_canonical_location_as_anchor_point
            assert self.model.use_canonical_location_from_generator
            assert self.disable_gan_training
            assert not self.use_learn_generator_with_randomly_moved_shape
            assert self.is_constant_motion_range
        if self.model.param_type == 'only_amount':
            assert self.is_canonical_location_as_anchor_point
            assert self.model.use_canonical_location_from_generator
            assert self.model.use_canonical_direction_from_generator
            assert not self.use_learn_generator_with_randomly_moved_shape
            assert self.is_constant_motion_range
        self.tsdf_loss_weight = tsdf_loss_weight
        self.use_occ_recon_loss_for_occ_and_tsdf_for_param = use_occ_recon_loss_for_occ_and_tsdf_for_param
        self.occ_recon_loss_for_occ_and_tsdf_for_param_occinize_scale = occ_recon_loss_for_occ_and_tsdf_for_param_occinize_scale
        self.disable_detach_canonical_location_as_pred_values_in_generator = disable_detach_canonical_location_as_pred_values_in_generator
        self.is_randomly_disable_rotation_per_part = is_randomly_disable_rotation_per_part
        self.is_continous_axis_angle_rotation = is_continous_axis_angle_rotation
        if self.is_continous_axis_angle_rotation:
            assert self.model.param_type == 'only_amount_as_matrix'
            assert self.model.is_decode_canonical_direction_as_rot_matrix
        self.latent_quantize_diff_loss_weight = latent_quantize_diff_loss_weight
        self.use_recon_canonical_shape_loss = use_recon_canonical_shape_loss
        self.recon_canonical_shape_loss_weight = recon_canonical_shape_loss_weight
        self.sample_transform_param_rotation_max = sample_transform_param_rotation_max
        self.sample_transform_param_translation_max = sample_transform_param_translation_max
        self.surface_quasi_sdf_points_per_point = surface_quasi_sdf_points_per_point
        self.surface_quasi_sdf_std = surface_quasi_sdf_std
        self.use_surface_quasi_sdf_loss = use_surface_quasi_sdf_loss
        self.surface_quasi_sdf_loss_weight = surface_quasi_sdf_loss_weight
        if self.use_surface_quasi_sdf_loss:
            assert not self.use_occ_recon_loss_for_occ_and_tsdf_for_param
        self.surface_quasi_sdf_input_subsampling_num = surface_quasi_sdf_input_subsampling_num
        self.is_correct_continuous_rotation = is_correct_continuous_rotation
        self.use_explicit_pointcloud_loss_for_motion_with_surface_points = use_explicit_pointcloud_loss_for_motion_with_surface_points
        self.explicit_pointcloud_loss_for_motion_with_surface_points_weight = explicit_pointcloud_loss_for_motion_with_surface_points_weight
        self.is_add_tie_breaking_noise_in_occ_recon = is_add_tie_breaking_noise_in_occ_recon
        self.tie_breaking_noise_in_occ_recon_scale = tie_breaking_noise_in_occ_recon_scale
        self.use_weight_for_positive_values_for_implicit_reconstruction_loss = use_weight_for_positive_values_for_implicit_reconstruction_loss
        self.weight_for_positive_values_for_implicit_reconstruction_loss_threshold = weight_for_positive_values_for_implicit_reconstruction_loss_threshold
        self.weight_for_positive_values_for_implicit_reconstruction_loss_scale = weight_for_positive_values_for_implicit_reconstruction_loss_scale
        self.is_only_visualize_surface_points_prediction = is_only_visualize_surface_points_prediction
        self.is_apply_tanh_to_merged_canonical_location = is_apply_tanh_to_merged_canonical_location
        self.use_motion_amount_inverse_std_loss = use_motion_amount_inverse_std_loss
        self.motion_amount_inverse_std_loss_weight = motion_amount_inverse_std_loss_weight
        self.motion_amount_inverse_std_loss_rotation_threshold = motion_amount_inverse_std_loss_rotation_threshold
        self.motion_amount_inverse_std_loss_translation_threshold = motion_amount_inverse_std_loss_translation_threshold
        self.motion_amount_inverse_std_loss_aggregation_type = motion_amount_inverse_std_loss_aggregation_type
        assert self.motion_amount_inverse_std_loss_aggregation_type in [
            'min', 'max', 'mean', 'exact'
        ]
        self.use_moved_overlap_regularizer_loss = use_moved_overlap_regularizer_loss
        assert not (self.use_moved_overlap_regularizer_loss
                    and self.use_overlap_regularizer)
        self.use_anchor_position_near_gt_shape_loss_near_primitive = use_anchor_position_near_gt_shape_loss_near_primitive
        self.use_anchor_position_near_gt_shape_loss_union_to_static_primitive = use_anchor_position_near_gt_shape_loss_union_to_static_primitive
        self.is_canonical_location_as_anchor_point_not_part_centric = is_canonical_location_as_anchor_point_not_part_centric
        if self.is_canonical_location_as_anchor_point_not_part_centric:
            assert self.is_canonical_location_as_anchor_point
        self.use_randomly_moved_overlap_regularizer_loss = use_randomly_moved_overlap_regularizer_loss
        if self.use_randomly_moved_overlap_regularizer_loss:
            assert not self.disable_gan_training and self.use_learn_generator_with_randomly_moved_shape
        self.use_location_offset_regulariztion_loss = use_location_offset_regulariztion_loss
        self.location_offset_regulariztion_loss_gausian_scale = location_offset_regulariztion_loss_gausian_scale
        self.location_offset_regulariztion_loss_weight = location_offset_regulariztion_loss_weight
        self.use_canonical_diff_prior_recon_loss = use_canonical_diff_prior_recon_loss
        self.use_minimize_raw_canonical_location_to_offset_loss = use_minimize_raw_canonical_location_to_offset_loss
        self.minimize_raw_canonical_location_to_offset_loss_weight = minimize_raw_canonical_location_to_offset_loss_weight
        self.use_raw_anchor_position_near_gt_shape_loss = use_raw_anchor_position_near_gt_shape_loss
        self.location_offset_regulariztion_loss_type = location_offset_regulariztion_loss_type
        self.minimize_raw_canonical_location_to_offset_loss_detach_canonical_location_grad = minimize_raw_canonical_location_to_offset_loss_detach_canonical_location_grad
        self.use_anchor_position_chain_assumption_loss = use_anchor_position_chain_assumption_loss
        self.anchor_position_chain_assumption_loss_weight = anchor_position_chain_assumption_loss_weight
        self.minimize_raw_canonical_direction_to_offset_loss_weight = minimize_raw_canonical_direction_to_offset_loss_weight
        self.use_minimize_raw_canonical_direction_to_offset_loss = use_minimize_raw_canonical_direction_to_offset_loss
        self.minimize_raw_canonical_direction_to_offset_loss_detach_canonical_direction_grad = minimize_raw_canonical_direction_to_offset_loss_detach_canonical_direction_grad
        self.use_screw_motion_spectrum_entropy_loss = use_screw_motion_spectrum_entropy_loss
        self.screw_motion_spectrum_entropy_loss_weight = screw_motion_spectrum_entropy_loss_weight
        self.disable_canonical_location_for_translation = disable_canonical_location_for_translation
        self.use_voxel_discriminator = use_voxel_discriminator
        self.anchor_position_chain_assumption_loss_only_self = anchor_position_chain_assumption_loss_only_self
        self.anchor_position_chain_assumption_loss_threshold = anchor_position_chain_assumption_loss_threshold
        self.is_sample_transform_param_random_additive_to_original = is_sample_transform_param_random_additive_to_original
        if self.is_sample_transform_param_random_additive_to_original:
            assert self.model.bound_amount_to_pi_and_one

    def visualize(self, data):
        result_images = []
        if self.model_input_type == 'voxel':
            inputs = data['voxels'].to(self.device)
        elif self.model_input_type == 'surface_points':
            inputs = data['surface_points'].to(self.device)
        else:
            raise NotImplementedError

        if 'part_surface_points' in data:
            seg_points = data['part_surface_points']
            seg_labels = data['part_surface_labels']
        else:
            seg_points = data['points']
            seg_labels = data['values']
            # We assume in this case "values" is labels
            assert seg_labels.max() > 1

        is_motion_param_in_data = False
        if 'param_primitive_type' in data:
            is_motion_param_in_data = True
        """
        images = self.visualizer.visualize_pointcloud(seg_points.numpy(),
                                                      seg_labels.numpy())
        result_images.append({
            'type': 'image',
            'desc': 'part_seg_gt',
            'data': images
        })
        """

        inputs_gpu = inputs.to(self.device)
        seg_points_gpu = seg_points.to(self.device)
        with torch.no_grad():
            ret = self.model(inputs_gpu, seg_points_gpu)
        pred_occ = ret['occupancy']
        #pred_occ_max, pred_occ_argmax = pred_occ.max(axis=-1)

        if self.occupancy_reduction_loss_type in self.imnet_occupancy_reduction_loss_types:
            pred_occ_max, pred_occ_argmax = pred_occ.max(axis=-1)
        elif self.occupancy_reduction_loss_type == 'occnet':
            pred_occ_max, pred_occ_argmax = torch.sigmoid(pred_occ).max(
                axis=-1)
        else:
            raise NotImplementedError
        if 'part_surface_points' in data:
            pred_occ_label = pred_occ_argmax + 1
        else:
            pred_occ_label = torch.where(
                pred_occ_max >= self.visualize_isosurface_threshold,
                pred_occ_argmax + 1,
                torch.zeros_like(pred_occ_max).long())
        """
        # Original pred points
        images = self.visualizer.visualize_pointcloud(
            seg_points.numpy(),
            pred_occ_label.detach().cpu().numpy())

        result_images.append({
            'type': 'image',
            'desc': 'part_seg_pred',
            'data': images
        })
        """

        # Recon gt
        points = data['points']
        values = data['values']
        kwargs = {}
        if is_motion_param_in_data:
            kwargs['primitive_type'] = data['param_primitive_type'].numpy()
            kwargs['rotation_direction'] = data['param_direction'].numpy()
            kwargs['rotation_anchor_point'] = data['param_anchor_point'].numpy(
            )
        s = time.time()
        images = self.visualizer.visualize_pointcloud(points.numpy(),
                                                      values.numpy(), **kwargs)
        result_images.append({
            'type': 'image',
            'desc': 'recon_gt',
            'data': images
        })

        # Recon pred
        inputs_gpu = inputs.to(self.device)
        points_gpu = points.clone().to(self.device)
        s = time.time()
        with torch.no_grad():
            ret = self.model(inputs_gpu, points_gpu)
        pred_occ = ret['occupancy']
        pred_latent = ret['latent']
        pred_params = ret['param']

        pred_generator_kwargs = ret['generator_kwargs']

        if self.model.is_expand_rotation_for_euler_angle:
            points_gpu_expand = points_gpu.unsqueeze(1).expand(
                -1, self.model.expand_rotation_for_euler_angle_sample_num + 1,
                -1, -1).contiguous().view(
                    points_gpu.size(0) *
                    (self.model.expand_rotation_for_euler_angle_sample_num +
                     1), points_gpu.size(-2), points_gpu.size(-1))
        else:
            points_gpu_expand = points_gpu

        self.preprocess_transformation(ret, points, points_gpu_expand,
                                       pred_occ, pred_params, pred_latent,
                                       pred_generator_kwargs)

        if self.is_only_visualize_surface_points_prediction:
            kwargs = {}
            if is_motion_param_in_data:
                kwargs = self.get_kwargs_for_visualizer(pred_params)

            transformed_points = self.get_explicitly_transformed_points(
                ret['surface_points'], pred_params)
            labels = []
            for idx in range(transformed_points.size(-2)):
                labels.append(
                    torch.ones_like(transformed_points[..., idx, 0]) *
                    (idx + 1))
            labels = torch.cat(labels, -1)
            images = self.visualizer.visualize_pointcloud(
                ret['surface_points'].detach().cpu().view(
                    labels.size(0), -1, 3).numpy(),
                labels.detach().cpu().numpy(), **kwargs)
            result_images.append({
                'type': 'image',
                'desc': 'recon_surface_points',
                'data': images
            })

            images = self.visualizer.visualize_pointcloud(
                transformed_points.detach().cpu().view(labels.size(0), -1,
                                                       3).numpy(),
                labels.detach().cpu().numpy(), **kwargs)
            result_images.append({
                'type': 'image',
                'desc': 'recon_moved_surface_points',
                'data': images
            })

        else:
            #pred_occ_max, pred_occ_argmax = pred_occ.max(axis=-1)
            if self.model.is_expand_rotation_for_euler_angle:
                pred_occ_reduced = pred_occ.view(
                    pred_latent.size(0),
                    self.model.expand_rotation_for_euler_angle_sample_num + 1,
                    pred_occ.size(-2), pred_occ.size(-1))[:, 0, :, :]
            else:
                pred_occ_reduced = pred_occ

            if self.occupancy_reduction_loss_type in self.imnet_occupancy_reduction_loss_types:
                pred_occ_max, pred_occ_argmax = pred_occ_reduced.max(axis=-1)
            elif self.occupancy_reduction_loss_type == 'occnet':
                pred_occ_max, pred_occ_argmax = torch.sigmoid(
                    pred_occ_reduced).max(axis=-1)
            else:
                raise NotImplementedError
            pred_occ_label = torch.where(
                pred_occ_max >= self.visualize_isosurface_threshold,
                pred_occ_argmax + 1,
                torch.zeros_like(pred_occ_max).long())

            if self.is_move_points:
                images = self.visualizer.visualize_pointcloud(
                    points.numpy(),
                    pred_occ_label.detach().cpu().numpy())

                result_images.append({
                    'type': 'image',
                    'desc': 'recon_pred',
                    'data': images
                })

                moved_points = self.move_points(pred_occ, points_gpu,
                                                pred_params)
                images = self.visualizer.visualize_pointcloud(
                    moved_points.detach().cpu().numpy(),
                    pred_occ_label.detach().cpu().numpy())
                result_images.append({
                    'type': 'image',
                    'desc': 'recon_moved',
                    'data': images
                })

            else:
                with torch.no_grad():
                    moved_occ = self.get_moved_occupancy_value(
                        ret['latent'],
                        points_gpu_expand, {'occupancy': pred_occ},
                        pred_params,
                        generator_kwargs=pred_generator_kwargs)['occupancy']

                if self.model.is_expand_rotation_for_euler_angle:
                    moved_occ_reduced = moved_occ.view(
                        pred_latent.size(0),
                        self.model.expand_rotation_for_euler_angle_sample_num +
                        1, pred_occ.size(-2), pred_occ.size(-1))[:, 0, :, :]
                else:
                    moved_occ_reduced = moved_occ

                #pred_occ_max, pred_occ_argmax = pred_occ.max(axis=-1)
                if self.occupancy_reduction_loss_type in self.imnet_occupancy_reduction_loss_types:
                    moved_pred_occ_max, moved_pred_occ_argmax = moved_occ_reduced.max(
                        axis=-1)
                elif self.occupancy_reduction_loss_type == 'occnet':
                    moved_pred_occ_max, moved_pred_occ_argmax = torch.sigmoid(
                        moved_occ_reduced).max(axis=-1)
                else:
                    raise NotImplementedError
                moved_pred_occ_label = torch.where(
                    moved_pred_occ_max >= self.visualize_isosurface_threshold,
                    moved_pred_occ_argmax + 1,
                    torch.zeros_like(moved_pred_occ_max).long())

                kwargs = {}
                if is_motion_param_in_data:
                    kwargs = self.get_kwargs_for_visualizer(pred_params)
                images = self.visualizer.visualize_pointcloud(
                    points.numpy(),
                    moved_pred_occ_label.detach().cpu().numpy(), **kwargs)
                result_images.append({
                    'type': 'image',
                    'desc': 'recon_moved',
                    'data': images
                })

        return result_images

    def get_kwargs_for_visualizer(self, param):
        kwargs = {}
        kwargs['rotation_direction'] = param['rotation_direction'].detach(
        ).cpu().numpy()
        kwargs['rotation_anchor_point'] = param[
            'rotation_anchor_point'].detach().cpu().numpy()

        if 'translation_anchor_point' in param:
            translation_anchor_point = param[
                'translation_anchor_point'].detach().cpu().numpy()
            kwargs['rotation_anchor_point'] = np.concatenate(
                [kwargs['rotation_anchor_point'], translation_anchor_point],
                axis=1)
            translation_direction = param['translation_direction'].detach(
            ).cpu().numpy()
            kwargs['rotation_direction'] = np.concatenate(
                [kwargs['rotation_direction'], translation_direction], axis=1)

        primitive_type = np.ones(
            [param['rotation_direction'].shape[0], self.primitive_num])
        primitive_type[:, 0] = 0
        primitive_type[:, 1:self.model.rotation_primitive_num + 1] = 1
        primitive_type[:, self.model.rotation_primitive_num + 1:] = 2
        kwargs['primitive_type'] = primitive_type

        if self.model.is_expand_rotation_for_euler_angle:
            kwargs_tmp = {}
            for key, value in kwargs.items():
                batch_size = int(
                    kwargs[key].shape[0] /
                    (self.model.expand_rotation_for_euler_angle_sample_num +
                     1))
                kwargs_tmp[key] = kwargs[key].reshape([
                    batch_size,
                    self.model.expand_rotation_for_euler_angle_sample_num + 1,
                    *kwargs[key].shape[1:]
                ])[:, 0, ...]
            kwargs = kwargs_tmp

        return kwargs

    def train_implicit_losses(self,
                              data,
                              step,
                              skip_gp,
                              return_eval_loss=False):
        if self.model_input_type == 'voxel':
            inputs = data['voxels'].to(self.device)
        elif self.model_input_type == 'surface_points':
            inputs = data['surface_points'].to(self.device)
        else:
            raise NotImplementedError

        if self.use_voxel_discriminator:
            voxels = data['voxels'].to(self.device)
            voxel_grid = data['voxel_grid'].to(self.device)
        else:
            voxels = None
            voxel_grid = None

        points = data['points'].to(self.device)
        values = data['values'].to(self.device).float()
        target_params = {
            key: value
            for key, value in data.items() if key.startswith('param')
        }
        # loss

        if return_eval_loss:
            eval_losses = self.compute_eval_losses(target_params, values,
                                                   points, inputs)
            yield eval_losses

        if 'tsdf' in data:
            values = data['tsdf'].to(self.device).float()
        use_gp = self.use_gradient_penalty and (
            step % self.apply_gradient_penalty_every == 0) and not skip_gp
        if not self.disable_gan_training:
            if self.use_self_supervised_canonical_learning:
                disc_losses = self.train_implicit_canonical_learning_discriminator(
                    values,
                    points,
                    inputs,
                    use_gp=use_gp,
                    voxels=voxels,
                    voxel_grid=voxel_grid)
            else:
                disc_losses = self.train_implicit_discriminator(
                    values,
                    points,
                    inputs,
                    voxels=voxels,
                    voxel_grid=voxel_grid)
            yield disc_losses
        if self.use_alternate_train_generator_and_paramnet:
            if self.train_paramnet_every > self.train_generator_every:
                if step % self.train_paramnet_every == 0:
                    if self.is_freeze_recon_loss_during_paramnet_training:
                        ori_occupancy_loss_weight = self.occupancy_loss_weight
                        self.occupancy_loss_weight = 0
                    with train_util.freeze_models([self.model.generator]):
                        gen_losses = self.train_generator(
                            values,
                            points,
                            inputs,
                            voxels=voxels,
                            voxel_grid=voxel_grid)
                    if self.is_freeze_recon_loss_during_paramnet_training:
                        self.occupancy_loss_weight = ori_occupancy_loss_weight
                else:
                    with train_util.freeze_models([self.model.paramnet]):
                        gen_losses = self.train_generator(
                            values,
                            points,
                            inputs,
                            voxels=voxels,
                            voxel_grid=voxel_grid)
            else:
                if step % self.train_generator_every == 0:
                    with train_util.freeze_models([self.model.paramnet]):
                        gen_losses = self.train_generator(
                            values,
                            points,
                            inputs,
                            voxels=voxels,
                            voxel_grid=voxel_grid)
                else:
                    if self.is_freeze_recon_loss_during_paramnet_training:
                        ori_occupancy_loss_weight = self.occupancy_loss_weight
                        self.occupancy_loss_weight = 0
                    with train_util.freeze_models([self.model.generator]):
                        gen_losses = self.train_generator(
                            values,
                            points,
                            inputs,
                            voxels=voxels,
                            voxel_grid=voxel_grid)
                    if self.is_freeze_recon_loss_during_paramnet_training:
                        self.occupancy_loss_weight = ori_occupancy_loss_weight

        else:
            gen_losses = self.train_generator(values,
                                              points,
                                              inputs,
                                              voxels=voxels,
                                              voxel_grid=voxel_grid)

        yield gen_losses

    def train_generator(self, *args, **kwargs):
        if self.use_self_supervised_canonical_learning:
            gen_losses = self.train_implicit_canonical_learning_generator(
                *args, **kwargs)
        else:
            gen_losses = self.train_implicit_generator(*args)
        return gen_losses

    def train_implicit_generator(self, values, points, inputs, prefix=''):
        losses = {}
        total_G_loss = 0.
        # inference for generator
        ret = self.model(inputs, points)
        pred_values = ret['occupancy']

        ## recon loss

        reconstruction_losses, occ_loss_weighted = self.get_implicit_reconstruction_loss(
            pred_values, values)
        losses.update(reconstruction_losses)

        total_G_loss = total_G_loss + occ_loss_weighted

        if not self.disable_induction_by_moving:
            pred_params = ret['param']
            sampled_transform_param = self.sample_transform_param(pred_params)

            losses.update(self.get_transformation_stats(pred_params))
        else:
            sampled_transform_param = None

        if not self.disable_gan_training:
            discriminator_ret = self.get_discriminator_result(
                pred_values,
                points,
                sampled_transform_param,
                inputs,
                no_move=self.disable_induction_by_moving,
                detach_param=self.is_detach_param_in_generator,
                detach_func=self.dict_detach_func)

            ### Generator
            G_real = -discriminator_ret['D']
            if self.gan_type == 'lsgan':
                G_loss = F.mse_loss(G_real,
                                    torch.ones_like(G_real),
                                    reduction='mean')
            else:
                raise NotImplementedError

            losses['G_loss'] = G_loss
            G_loss_weighted = G_loss * self.G_loss_weight
            losses['G_loss_weighted'] = G_loss_weighted

            total_G_loss = total_G_loss + G_loss_weighted

            if self.use_part_shape_similarity_loss:
                part_shape_similarity_loss_ret, part_shape_similarity_loss_weighted = self.get_part_shape_similarity_loss(
                    discriminator_ret['points_value'], pred_values)
                part_shape_similarity_loss_ret = {
                    'G_' + name: value
                    for name, value in part_shape_similarity_loss_ret.items()
                }
                total_G_loss = total_G_loss + part_shape_similarity_loss_weighted
                losses.update(part_shape_similarity_loss_ret)
            if self.use_anchor_position_loss:
                anchor_position_loss_ret, anchor_position_loss_weighted = self.get_anchor_position_loss(
                    points, pred_values, pred_params['rotation_anchor_point'])

        if (self.learn_moving_self_slide_reg or self.learn_moving_self_seg_reg
            ) and not self.disable_induction_by_moving:
            ret = self.model(inputs, inputs, return_param=False)

            sampled_transform_param = self.sample_transform_param(pred_params)
            moved_inputs = self.move_points(ret['occupancy'], inputs,
                                            sampled_transform_param)
            ret2 = self.model(moved_inputs, moved_inputs)

            losses.update(
                self.get_transformation_stats(ret2['param'],
                                              prefix='self_supervised'))
            ## recon loss

            if self.learn_moving_self_seg_reg:
                self_supervised_seg_loss_ret, self_supervised_seg_loss_weighted = self.self_supervised_seg_loss(
                    ret2['occupancy'], ret['occupancy'])
                losses.update(self_supervised_seg_loss_ret)
                total_G_loss = total_G_loss + self_supervised_seg_loss_weighted

            if self.learn_moving_self_slide_reg:
                self_supervised_slide_loss = self.transformation_loss(
                    ret2['param'], sampled_transform_param, inputs,
                    ret['occupancy'])
                self_supervised_slide_loss_weighted = self_supervised_slide_loss * self.self_supervised_slide_loss_weight
                losses[
                    'self_supervised_param_loss'] = self_supervised_slide_loss
                losses[
                    'self_supervised_param_loss_weighted'] = self_supervised_slide_loss_weighted
                total_G_loss = total_G_loss + self_supervised_slide_loss_weighted

        if self.use_overlap_regularizer:
            overlap_regularizer_losses, total_overlap_regularizer_loss_weighted = self.get_regularizer_overlap(
                pred_values, pred_values)
            total_G_loss = total_G_loss + total_overlap_regularizer_loss_weighted
            losses.update(overlap_regularizer_losses)

        losses['total_G_loss'] = total_G_loss

        if self.is_check_gradient_scale:
            grad_ret = self.check_gradient_scale(losses,
                                                 ret,
                                                 ret_keys=['occupancy'],
                                                 prefix='G/')
            grad_ret2 = self.check_occupancy_gradient_scale(
                losses, dict(pred_values=pred_values), prefix='G/')
            losses.update(grad_ret)
            losses.update(grad_ret2)

        prefixed_losses = {(prefix + name): value
                           for name, value in losses.items()}
        return prefixed_losses

    def train_implicit_discriminator(self, values, points, inputs, prefix=''):
        disc_losses = {}
        total_D_loss = 0.  #torch.zeros_like(points_A.sum())
        with torch.no_grad():
            ret = self.model(inputs, points)

        pred_values = ret['occupancy']

        pred_params = ret['param']
        sampled_transform_param = self.sample_transform_param(pred_params)
        discriminator_ret = self.get_discriminator_result(
            pred_values,
            points,
            sampled_transform_param,
            inputs,
            no_move=self.disable_induction_by_moving,
            detach=True)

        if not self.disable_gan_training:
            discriminator_real_ret = self.get_real_discriminator_result(
                values, points)
            points_value_B = discriminator_real_ret['points_value']
            D_real = discriminator_real_ret['D']
            if self.gan_type == 'lsgan':
                D_real_loss = F.mse_loss(D_real,
                                         torch.ones_like(D_real),
                                         reduction='mean')
            else:
                raise NotImplementedError
            D_real_loss_weighted = D_real_loss * self.D_real_loss_weight * self.D_loss_weight

            disc_losses.update({
                'D_real_loss': D_real_loss,
                'D_real_loss_weighted': D_real_loss_weighted,
            })

            D_fake = discriminator_ret['D']

            if self.gan_type == 'lsgan':
                D_fake_loss = F.mse_loss(D_fake,
                                         torch.zeros_like(D_fake),
                                         reduction='mean')
            else:
                raise NotImplementedError
            D_fake_loss_weighted = D_fake_loss * self.D_fake_loss_weight * self.D_loss_weight

            total_D_loss = total_D_loss + D_real_loss_weighted + D_fake_loss_weighted

            disc_losses.update({
                'D_fake_loss': D_fake_loss,
                'D_fake_loss_weighted': D_fake_loss_weighted
            })

        if self.use_part_shape_similarity_loss:
            part_shape_similarity_loss_ret, part_shape_similarity_loss_weighted = self.get_part_shape_similarity_loss(
                discriminator_ret['points_value'], pred_values)
            total_D_loss = total_D_loss + part_shape_similarity_loss_weighted
            part_shape_similarity_loss_ret = {
                'D_' + name: value
                for name, value in part_shape_similarity_loss_ret.items()
            }
            disc_losses.update(part_shape_similarity_loss_ret)

        disc_losses['total_D_loss'] = total_D_loss
        prefixed_losses = {(prefix + name): value
                           for name, value in disc_losses.items()}
        return prefixed_losses

    def dict_detach_func(self, params):
        assert isinstance(params, dict)
        new_params = {}
        for key, value in params.items():
            new_params[key] = value.clone().detach()
        return new_params

    def compute_eval_losses(self, params, values, points, inputs):
        with torch.no_grad():
            ret = self.model(inputs, points)
        if self.disable_induction_by_moving:
            pred_params = None
        else:
            pred_params = ret['param']
        pred_latent = ret['latent']

        pred_values = ret['occupancy']
        pred_generator_kwargs = ret['generator_kwargs']
        if self.model.is_expand_rotation_for_euler_angle:
            points_expand = points.unsqueeze(1).expand(
                -1, self.model.expand_rotation_for_euler_angle_sample_num + 1,
                -1, -1).contiguous().view(
                    points.size(0) *
                    (self.model.expand_rotation_for_euler_angle_sample_num +
                     1), points.size(-2), points.size(-1))
        else:
            points_expand = points

        self.preprocess_transformation(ret, points, points_expand, pred_values,
                                       pred_params, pred_latent,
                                       pred_generator_kwargs)
        """
        if self.model.use_canonical_location_from_generator:
            pred_params['canonical_location'] = ret['canonical_location']
            if 'rotation_location_offset' in ret:
                canonical_location = pred_params['canonical_location']
                rotation_canonical_location = canonical_location[:,
                                                                 1:self.model.
                                                                 rotation_primitive_num
                                                                 + 1, :]
                canonical_location[:, 1:self.model.rotation_primitive_num +
                                   1, :] = rotation_canonical_location + ret[
                                       'rotation_location_offset']
                pred_params['canonical_location'] = canonical_location
            if 'translation_location_offset' in ret:
                translation_canonical_location = canonical_location[:,
                                                                    self.model.
                                                                    rotation_primitive_num
                                                                    + 1:, :]
                canonical_location[:, self.model.rotation_primitive_num +
                                   1:, :] = translation_canonical_location + ret[
                                       'translation_location_offset']
                pred_params['canonical_location'] = canonical_location

            if self.is_apply_tanh_to_merged_canonical_location:
                pred_params['canonical_location'] = torch.tanh(
                    pred_params['canonical_location']) * 0.5

        if self.is_canonical_location_as_anchor_point and 'canonical_location' in pred_params:
            pred_params['rotation_anchor_point'] = pred_params[
                'canonical_location'][:, 1:self.model.rotation_primitive_num +
                                      1, :]
        if self.model.use_canonical_direction_from_generator:
            if self.model.is_decode_canonical_direction_as_rot_matrix:
                if self.is_correct_continuous_rotation:
                    pred_params['rotation_matrix'] = torch.matmul(
                        ret['canonical_rotation_matrix']
                        [:, 1:self.model.rotation_primitive_num + 1, :, :],
                        pred_params['rotation_matrix'])
                else:
                    pred_params['rotation_matrix'] = torch.matmul(
                        pred_params['rotation_matrix'],
                        ret['canonical_rotation_matrix']
                        [:, 1:self.model.rotation_primitive_num + 1, :, :])
            pred_params['rotation_direction'] = ret[
                'canonical_direction'][:, 1:self.model.rotation_primitive_num +
                                       1, :]
            pred_params['translation_direction'] = ret[
                'canonical_direction'][:, self.model.rotation_primitive_num +
                                       1:, :]
        """

        if self.use_self_supervised_canonical_learning:
            with torch.no_grad():
                pred_values = self.get_moved_occupancy_value(
                    ret['latent'],
                    points_expand, {'occupancy': pred_values},
                    pred_params,
                    generator_kwargs=pred_generator_kwargs)['occupancy']

        if self.model.is_expand_rotation_for_euler_angle:
            if pred_values.size(0) != values.size(0):
                pred_values = pred_values.view(
                    values.size(0),
                    self.model.expand_rotation_for_euler_angle_sample_num + 1,
                    *pred_values.shape[1:])[:, 0, ...]
            new_pred_params = {}
            for key, value in pred_params.items():
                if value.size(0) != values.size(0):
                    new_pred_params[key] = value.view(
                        values.size(0),
                        self.model.expand_rotation_for_euler_angle_sample_num +
                        1, *value.shape[1:])[:, 0, ...]
            pred_params = new_pred_params

        eval_losses = self.get_eval_losses(pred_values, values, pred_values,
                                           pred_params, params)
        return eval_losses

    def get_param_loss_for_eval(self, pred_params, params, prefix=''):
        losses = {}
        if params and self.model.motion_decoding_type in [
                'one_joint_type_per_primitive',
                'one_joint_type_per_primitive_rot_pm_num_specified'
        ]:
            post_param_loss = {}
            post_param_loss['pred_rotation_direction'] = pred_params[
                'rotation_direction'].detach().cpu()
            post_param_loss['pred_rotation_anchor_point'] = pred_params[
                'rotation_anchor_point'].detach().cpu()
            if self.is_constant_motion_range:
                rotation_scale = pred_params['rotation_scale'].detach().cpu(
                ) * 0. + np.pi
            else:
                rotation_scale = pred_params['rotation_scale'].detach().cpu()
            post_param_loss['pred_rotation_deg'] = pred_params[
                'rotation_amount'].detach().cpu() * rotation_scale

            post_param_loss['pred_translation_direction'] = pred_params[
                'translation_direction'].detach().cpu()

            if 'translation_anchor_point' in pred_params:
                post_param_loss['pred_translation_anchor_point'] = pred_params[
                    'translation_anchor_point'].detach().cpu()
            if self.is_constant_motion_range:
                translation_scale = pred_params['translation_scale'].detach(
                ).cpu() * 0. + np.pi
            else:
                translation_scale = pred_params['translation_scale'].detach(
                ).cpu()
            post_param_loss['pred_translation_amount'] = pred_params[
                'translation_amount'].detach().cpu() * translation_scale

            pred_primitive_type = [0]
            pred_primitive_type.extend([1] * self.model.rotation_primitive_num)
            pred_primitive_type.extend([2] *
                                       self.model.translation_primitive_num)
            batch_size = post_param_loss['pred_rotation_deg'].shape[0]
            post_param_loss['pred_primitive_type'] = torch.tensor(
                pred_primitive_type).reshape(1, -1).expand(batch_size, -1)
            post_param_loss['pred_rotation_offset'] = torch.tensor(
                np.ones([batch_size], dtype=np.float32)).reshape(-1, 1)

            post_param_loss['target_direction'] = params['param_direction']
            post_param_loss['target_anchor_point'] = params[
                'param_anchor_point']
            post_param_loss['target_amount'] = params['param_amount']
            post_param_loss['target_primitive_type'] = params[
                'param_primitive_type']
            post_param_loss['target_primitive_type_offset'] = params[
                'param_primitive_type_offset']
            losses['post_param_loss'] = post_param_loss
            return losses
        else:
            return losses

    def move_points(self, values, points, transformation, force_move=False):
        moved_points_moving_points = points.clone()
        if self.is_move_points or force_move:
            # B, P, 1
            occ_argmax = values.argmax(-1, keepdims=True)
            if self.is_temporal_pad_slide_dim:
                raise NotImplementedError
            for idx, midx in enumerate(range(1, self.primitive_num)):

                if self.model.motion_decoding_type == 'default':
                    raise NotImplementedError("need reconsider")
                    primitive_wise_transformation = {
                        key: value[:, idx, ...]
                        for key, value in transformation.items()
                    }
                    tmp = self.apply_translation(
                        moved_points_moving_points,
                        primitive_wise_transformation)
                    tmp = self.apply_rotation(tmp,
                                              primitive_wise_transformation)
                elif self.model.motion_decoding_type in [
                        'one_joint_type_per_primitive',
                        'one_joint_type_per_primitive_rot_pm_num_specified'
                ]:
                    if idx < self.model.rotation_primitive_num:
                        primitive_wise_transformation = {
                            key:
                            value[:,
                                  midx if key.startswith('canonical') else idx,
                                  ...]
                            for key, value in transformation.items()
                            if key.startswith('rotation')
                            or key.startswith('canonical')
                        }
                        tmp = self.apply_rotation(
                            moved_points_moving_points,
                            primitive_wise_transformation)
                    else:
                        primitive_wise_transformation = {
                            key:
                            value[:, idx -
                                  (-1 if key.startswith('canonical') else self.
                                   model.rotation_primitive_num), ...]
                            for key, value in transformation.items()
                            if key.startswith('translation')
                            or key.startswith('canonical')
                        }
                        tmp = self.apply_translation(
                            moved_points_moving_points,
                            primitive_wise_transformation)
                primitive_mask = (occ_argmax == midx)
                # B, 3
                a = moved_points_moving_points * torch.logical_not(
                    primitive_mask)
                b = tmp * primitive_mask
                moved_points_moving_points = a + b
        else:
            raise NotImplementedError

        return moved_points_moving_points

    """
    def get_overlap_loss_for_eval(self,
                                  pred_occ_after_move,
                                  pred_occ_before_move,
                                  prefix=''):
        pred_occ_before_move_bool = pred_occ_before_move >= 0
        overlap_before_move_bool = torch.relu(
            pred_occ_before_move_bool.sum(-1) - 1).sum()
        losses = {'overlap_before_move_loss': overlap_before_move_bool}
        prefix = (prefix + '_') if prefix != '' else prefix
        losses = {prefix + name: value for name, value in losses.items()}
        return losses
    """

    def transformation_loss(self, pred_param, target_param, points, values):
        detached_param = {
            key: value.clone().detach()
            for key, value in target_param.items()
        }
        pred_moved_points = self.move_points(values, points, pred_param)
        target_moved_points = self.move_points(values, points, detached_param)

        return F.mse_loss(pred_moved_points, target_moved_points.detach())

    def apply_translation(self, coord, transformation, is_explicit=False):
        # B, dim
        if self.model.param_type == 'default':
            assert not is_explicit
            if self.disable_translation:
                return coord
            translation_direction = transformation['translation_direction']
            # B, 1
            translation_amount = transformation['translation_amount']
            if self.is_constant_motion_range:
                translation_scale = transformation['translation_scale'] * 0 + 1.
            else:
                translation_scale = transformation['translation_scale']
            translation = translation_direction * translation_amount * translation_scale
            if self.model.use_canonical_location_from_generator:
                translation = translation + transformation['canonical_location']
                return coord - translation.unsqueeze(-2)
            else:
                return coord + translation.unsqueeze(-2)
        elif self.model.param_type == 'motion_separate_affine_rotation_quaternion' and self.model.use_canonical_location_from_generator and self.is_canonical_location_as_anchor_point:
            assert not is_explicit
            translation = transformation[
                'translation_vector'] + transformation['canonical_location']
            return coord - translation.unsqueeze(-2)
        elif self.model.param_type.startswith('only_amount') and (
            (self.model.use_canonical_direction_from_generator and
             self.model.use_canonical_location_from_generator) or self.model.
                use_canonical_motion_from_paramnet_and_shape_from_generator
                or self.model.param_type.startswith(
                    'only_amount_as_matrix_loc_offset_canonical_motion_euler_angle'
                )) and self.is_canonical_location_as_anchor_point:
            translation_direction = transformation['translation_direction']
            # B, 1
            translation_amount = transformation['translation_amount']
            translation = translation_direction * translation_amount
            if not self.disable_canonical_location_for_translation:
                translation = translation + transformation['canonical_location']
            if is_explicit:
                return coord + translation.unsqueeze(-2)
            else:
                return coord - translation.unsqueeze(-2)
        else:
            raise NotImplementedError

    def apply_rotation(self, coord, transformation, is_explicit=False):
        if self.is_randomly_disable_rotation_per_part:
            assert not is_explicit
            transformation[
                'rotation_amount'] = transformation['rotation_amount'] * (
                    torch.rand_like(transformation['rotation_amount']) >= 0.5)
        if self.model.param_type == 'default':
            assert not is_explicit
            # B, dim
            rotation_anchor_point = transformation[
                'rotation_anchor_point'] * self.rotation_anchor_point_scale
            rotation_direction = transformation['rotation_direction']
            # B, 1
            rotation_amount = transformation['rotation_amount']
            if self.is_constant_motion_range:
                rotation_scale = transformation['rotation_scale'] * 0. + np.pi
            else:
                rotation_scale = transformation['rotation_scale']
            rotation_angle = rotation_amount * rotation_scale

            # B, 1, dim
            unsqueezed_rotation_anchor_point = rotation_anchor_point.unsqueeze(
                -2)
            # B, P, dim
            if self.model.use_canonical_location_from_generator:
                primitive_location_at_origin = transformation[
                    'canonical_location'].unsqueeze(
                        -2) - unsqueezed_rotation_anchor_point
                quat = geometry.get_quaternion(rotation_direction,
                                               rotation_angle)
                new_primitive_location = geometry.apply_3d_rotation(
                    primitive_location_at_origin,
                    quat) + unsqueezed_rotation_anchor_point

                rotated_coord = geometry.apply_3d_rotation(
                    coord - new_primitive_location, quat, inv=True)
            else:
                coord_at_origin = coord - unsqueezed_rotation_anchor_point

                quat = geometry.get_quaternion(rotation_direction,
                                               rotation_angle)

                rotated_coord = geometry.apply_3d_rotation(
                    coord_at_origin, quat) + unsqueezed_rotation_anchor_point
        elif self.model.param_type == 'motion_separate_affine_rotation_quaternion' and self.model.use_canonical_location_from_generator and self.is_canonical_location_as_anchor_point:
            assert not is_explicit
            primitive_location_at_origin = transformation[
                'canonical_location'].unsqueeze(-2)
            quat = transformation['rotation_vector']
            rotated_coord = geometry.apply_3d_rotation(
                coord - primitive_location_at_origin, quat, inv=True)

        elif self.model.param_type == 'only_amount' or (
                self.model.param_type.startswith('only_amount_as_matrix')
                and self.is_continous_axis_angle_rotation
        ) and self.model.use_canonical_direction_from_generator and self.model.use_canonical_location_from_generator and self.is_canonical_location_as_anchor_point:
            assert not is_explicit
            rotation_direction = transformation['rotation_direction']
            # B, 1
            rotation_amount = transformation['rotation_amount']
            print('rot amount mean', rotation_amount[0])

            rotation_angle = rotation_amount * np.pi

            quat = geometry.get_quaternion(rotation_direction, rotation_angle)
            print('rot dir norm', torch.norm(rotation_direction, dim=-1))

            primitive_location_at_origin = transformation[
                'canonical_location'].unsqueeze(-2)
            print('before rot coord',
                  (coord[0] - primitive_location_at_origin[0]),
                  rotation_angle[0])
            """
            rotated_coord = geometry.apply_3d_rotation(
                coord - primitive_location_at_origin, quat, inv=True)
            """
            rot_mat = geometry.compute_rotation_matrix_from_quaternion(quat)

            rotated_coord = torch.matmul(
                rot_mat.transpose(-1, -2).unsqueeze(1),
                (coord -
                 primitive_location_at_origin).unsqueeze(-1)).squeeze(-1)

            print('after rot coord', rotated_coord[0])
        elif self.model.param_type.startswith('only_amount_as_matrix') and (
            (self.model.use_canonical_direction_from_generator and
             self.model.use_canonical_location_from_generator) or self.model.
                use_canonical_motion_from_paramnet_and_shape_from_generator
                or self.model.param_type.startswith(
                    'only_amount_as_matrix_loc_offset_canonical_motion_euler_angle'
                )) and self.is_canonical_location_as_anchor_point:
            primitive_location_at_origin = transformation[
                'canonical_location'].unsqueeze(-2)
            if self.is_randomly_disable_rotation_per_part:
                transformation['rotation_matrix'] = torch.zeros_like(
                    transformation['rotation_matrix'])
                transformation['rotation_matrix'][..., 0, 0] = 1.
                transformation['rotation_matrix'][..., 1, 1] = 1.
                transformation['rotation_matrix'][..., 2, 2] = 1.
            if is_explicit:
                rotated_coord = torch.matmul(
                    transformation['rotation_matrix'].unsqueeze(1),
                    coord.unsqueeze(-1)).squeeze(
                        -1) + primitive_location_at_origin
            else:
                sgn = -1 if self.is_canonical_location_as_anchor_point_not_part_centric else 1
                if self.model.param_type.startswith(
                        'only_amount_as_matrix_loc_offset_canonical_motion_euler_angle_screw'
                ):
                    rotated_coord = torch.matmul(
                        transformation['rotation_matrix'].transpose(
                            -1, -2).unsqueeze(1),
                        (coord - sgn * primitive_location_at_origin -
                         (transformation['rotation_direction'] *
                          transformation['translation_amount']).unsqueeze(-2)
                         ).unsqueeze(-1)).squeeze(-1)
                else:
                    rotated_coord = torch.matmul(
                        transformation['rotation_matrix'].transpose(
                            -1, -2).unsqueeze(1),
                        (coord - sgn * primitive_location_at_origin
                         ).unsqueeze(-1)).squeeze(-1)
                if self.is_canonical_location_as_anchor_point_not_part_centric:
                    rotated_coord = rotated_coord + sgn * primitive_location_at_origin

        elif self.model.param_type == 'affine':
            raise NotImplementedError

        else:
            raise NotImplementedError

        return rotated_coord

    def sample_transform_param(self, params):
        new_params = defaultdict(lambda: [])
        for key, value in params.items():
            if self.model.param_type.startswith('only_amount_as_matrix') and (
                (self.model.use_canonical_direction_from_generator
                 and self.model.use_canonical_location_from_generator)
                    or self.model.
                    use_canonical_motion_from_paramnet_and_shape_from_generator
            ) and self.is_canonical_location_as_anchor_point:
                if key == 'translation_amount':

                    new_tr = torch.rand_like(
                        value) * self.sample_transform_param_translation_max
                    if self.is_sample_transform_param_random_additive_to_original:
                        new_theta = params['translation_amount'] + new_tr 
                        residual = torch.relu(
                            new_theta -
                            self.sample_transform_param_translation_max)
                        new_tr = torch.where(residual > 0, residual, new_theta)
                    new_params[key] = new_tr
                elif key == 'rotation_matrix':
                    batch_size = value.size(0)
                    theta = torch.rand_like(value)[
                        ..., 0, 0] * self.sample_transform_param_rotation_max

                    if self.is_sample_transform_param_random_additive_to_original:
                        new_theta = params['rotation_amount'].squeeze(-1) + theta
                        residual = torch.relu(
                            new_theta -
                            self.sample_transform_param_rotation_max)
                        theta = torch.where(residual > 0, residual, new_theta)

                    new_params['rotation_amount'] = theta

                    zero = torch.zeros_like(theta)
                    """
                    one = torch.zeros_like(theta)
                    new_rotation_matrix = torch.cat(
                        [
                            torch.cos(theta), -torch.sin(theta), zero,
                            torch.sin(theta),
                            torch.cos(theta), zero, zero, zero, one
                        ],
                        dim=-1).view(batch_size,
                                     self.model.rotation_primitive_num, 3, 3)
                    """
                    one = torch.ones_like(theta)
                    new_rotation_matrix = torch.stack(
                        [
                            torch.cos(theta), -torch.sin(theta), zero,
                            torch.sin(theta),
                            torch.cos(theta), zero, zero, zero, one
                        ],
                        dim=-1).view(batch_size,
                                     self.model.rotation_primitive_num, 3, 3)

                    if self.model.is_decode_canonical_direction_as_rot_matrix or self.model.use_canonical_motion_from_paramnet_and_shape_from_generator:
                        if self.is_correct_continuous_rotation:
                            new_params[key] = torch.matmul(
                                params['canonical_rotation_matrix']
                                [:, 1:self.model.rotation_primitive_num +
                                 1, :, :], new_rotation_matrix)
                        else:
                            new_params[key] = torch.matmul(
                                new_rotation_matrix,
                                params['canonical_rotation_matrix']
                                [:, 1:self.model.rotation_primitive_num +
                                 1, :, :])
                    else:
                        raise NotImplementedError
                else:
                    new_params[key] = value
            else:
                if key.endswith('_amount'):
                    if self.use_adaptive_max_motion_range_sampling:
                        if self.model.motion_decoding_type == 'default':
                            num = self.primitive_num
                        elif self.model.motion_decoding_type in [
                                'one_joint_type_per_primitive',
                                'one_joint_type_per_primitive_rot_pm_num_specified'
                        ]:
                            if key == 'rotation_amount':
                                num = self.model.rotation_primitive_num
                            elif key == 'translation_amount':
                                num = self.model.translation_primitive_num
                            else:
                                raise NotImplementedError
                        else:
                            raise NotImplementedError

                        for idx in range(num):
                            max_range = value[:, idx, :].max(1,
                                                             keepdim=True)[0]
                            new_value = torch.rand_like(
                                value[:, idx, :]) * max_range
                            new_params[key].append(new_value.unsqueeze(1))
                        new_params[key] = torch.cat(new_params[key], dim=1)
                    else:
                        new_params[key] = torch.rand_like(value)
                else:
                    new_params[key] = value
        return new_params

    def self_supervised_seg_loss(self, pred_values, target_values):
        assert pred_values.ndim == 3
        assert pred_values.shape[-1] == self.primitive_num
        assert target_values.ndim == 3
        assert target_values.shape[-1] == self.primitive_num
        losses = {}

        self_supervised_seg_loss = torch.mean(
            (pred_values - target_values.clone().detach())**2)
        self_supervised_seg_loss_weighted = self_supervised_seg_loss * self.self_supervised_seg_loss_weight
        losses['self_supervised_seg_loss'] = self_supervised_seg_loss
        losses[
            'self_supervised_seg_loss_weighted'] = self_supervised_seg_loss_weighted

        return losses, losses['self_supervised_seg_loss_weighted']

    def get_transformation_stats(self, transformation, prefix=''):
        stats = {}
        for key, value in transformation.items():
            if not isinstance(value, torch.Tensor):
                continue

            stats[key + '_mean'] = value.mean()
            stats[key + '_std'] = value.std()
            # stats[key + '_min'] = value.min()
            # stats[key + '_max'] = value.max()

        prefixed_losses = {(prefix + name): value
                           for name, value in stats.items()}
        return prefixed_losses

    def get_anchor_position_loss(self, points, values, anchors):
        if self.occupancy_reduction_loss_type in self.imnet_occupancy_reduction_loss_types:
            pred_occ_max, pred_occ_argmax = values.max(axis=-1)
        elif self.occupancy_reduction_loss_type == 'occnet':
            pred_occ_max, pred_occ_argmax = torch.sigmoid(values).max(axis=-1)
        else:
            raise NotImplementedError
        pred_occ_label = torch.where(
            pred_occ_max >= self.visualize_isosurface_threshold,
            pred_occ_argmax + 1,
            torch.zeros_like(pred_occ_max).long())

        cnt = 0
        total_rot_pm_dist = torch.zeros([1],
                                        device=points.device,
                                        dtype=points.dtype).mean()
        total_pm1_dist = torch.zeros([1],
                                     device=points.device,
                                     dtype=points.dtype).mean()
        for idx in range(self.model.rotation_primitive_num):
            source_points = anchors[:, idx, :].unsqueeze(-2)

            rot_pm_idx = idx + 2
            mask = (pred_occ_label == rot_pm_idx)
            if mask.sum() == 0:
                continue
            dists = (
                (source_points - points)**2).sum(-1) * mask + (~mask) * 100
            dist = torch.relu(
                dists.min(-1)[0] - self.anchor_position_loss_margin**2)
            dist_clipped = torch.where(dist > 50, torch.zeros_like(dist),
                                       dist).sum()
            num = (dist < 50).sum()
            if num < 1:
                dist_rot_pm = dist_clipped * 0.
            else:
                dist_rot_pm = dist_clipped / num

            total_rot_pm_dist = total_rot_pm_dist + dist_rot_pm / self.model.rotation_primitive_num

            mask = pred_occ_label == 1
            dists = (
                (source_points - points)**2).sum(-1)[0] * mask + (~mask) * 100
            dist = torch.relu(
                dists.min(-1)[0] - self.anchor_position_loss_margin**2)
            dist_clipped = torch.where(dist > 50, torch.zeros_like(dist),
                                       dist).sum()
            num = (dist < 50).sum()
            if num < 1:
                dist_pm1 = dist_clipped * 0.
            else:
                dist_pm1 = dist_clipped / num

            total_pm1_dist = total_pm1_dist + dist_pm1 / self.model.rotation_primitive_num

        anchor_position_loss = total_pm1_dist + total_rot_pm_dist
        anchor_position_loss_weighted = anchor_position_loss * self.anchor_position_loss_weight

        ret = {
            'anchor_position_loss': anchor_position_loss,
            'anchor_position_loss_weighted': anchor_position_loss_weighted,
            'total_pm1_dist': total_pm1_dist,
            'total_rot_pm_dist': total_rot_pm_dist
        }

        return ret, anchor_position_loss_weighted

    def train_implicit_canonical_learning_generator(self,
                                                    values,
                                                    points,
                                                    inputs,
                                                    voxels=None,
                                                    voxel_grid=None,
                                                    prefix=''):
        losses = {}
        total_G_loss = 0.
        """
        Inference for generator
        """
        ret = self.model(inputs, points)

        # occupancy value of "canonically posed" input shape
        pred_values = ret['occupancy']
        # Transform param to transform canoncial shape to original input shape
        pred_params = ret['param']
        pred_latent = ret['latent']
        pred_generator_kwargs = ret['generator_kwargs']

        if self.model.is_expand_rotation_for_euler_angle:
            points_expand = points.unsqueeze(1).expand(
                -1, self.model.expand_rotation_for_euler_angle_sample_num + 1,
                -1, -1).contiguous().view(
                    points.size(0) *
                    (self.model.expand_rotation_for_euler_angle_sample_num +
                     1), points.size(-2), points.size(-1))
        else:
            points_expand = points
        if self.model.is_expand_rotation_for_euler_angle:
            values_expand = values.unsqueeze(1).expand(
                -1, self.model.expand_rotation_for_euler_angle_sample_num + 1,
                -1).contiguous().view(
                    points.size(0) *
                    (self.model.expand_rotation_for_euler_angle_sample_num +
                     1), values.size(-1))
        else:
            values_expand = points

        if self.use_surface_quasi_sdf_loss:
            if self.surface_quasi_sdf_input_subsampling_num is not None:
                noised_inputs = common.subsample_points(
                    inputs,
                    self.surface_quasi_sdf_input_subsampling_num,
                    axis=1)
            else:
                noised_inputs = inputs
            noise = torch.randn(
                *noised_inputs.shape,
                self.surface_quasi_sdf_points_per_point,
                device=noised_inputs.device,
                dtype=noised_inputs.dtype) * self.surface_quasi_sdf_std
            noised_inputs = (noised_inputs.unsqueeze(-1) + noise).transpose(
                -2, -1).contiguous().view(noised_inputs.size(0), -1, 3)
            surface_ret = self.model(pred_latent,
                                     noised_inputs,
                                     return_param=False,
                                     direct_input_to_decoder=True,
                                     generator_kwargs=pred_generator_kwargs)

        is_latent_quantize_diff = False
        if 'latent_quantize_diff' in ret:
            pred_latent_quantize_diff = ret['latent_quantize_diff']
            is_latent_quantize_diff = True

        self.preprocess_transformation(ret, points, points_expand, pred_values,
                                       pred_params, pred_latent,
                                       pred_generator_kwargs)
        losses.update(self.get_transformation_stats(pred_params))
        if self.use_surface_quasi_sdf_loss and self.model.is_expand_rotation_for_euler_angle:
            pred_params_for_quasi_sdf_loss = {}
            for key, value in pred_params.items():
                batch_size = int(
                    pred_params[key].shape[0] /
                    (self.model.expand_rotation_for_euler_angle_sample_num +
                     1))
                pred_params_for_quasi_sdf_loss[key] = pred_params[key].reshape(
                    [
                        batch_size,
                        self.model.expand_rotation_for_euler_angle_sample_num +
                        1, *pred_params[key].shape[1:]
                    ])[:, 0, ...]
        else:
            pred_params_for_quasi_sdf_loss = pred_params

        if self.use_moved_pretrained_feature_canonical_loss or self.use_recon_canonical_shape_loss:
            moved_values_ret = self.get_moved_occupancy_value(
                pred_latent,
                points_expand, {
                    'occupancy': pred_values,
                    'canonical_occupancy': ret['canonical_occupancy']
                },
                pred_params,
                detach_transformation=self.
                is_detach_transformation_for_occupancy_reconstruction,
                generator_kwargs=pred_generator_kwargs)
            moved_values = moved_values_ret['occupancy']
            moved_canonical_values = moved_values_ret['canonical_occupancy']

            if self.use_surface_quasi_sdf_loss:
                surface_moved_values_ret = self.get_moved_occupancy_value(
                    pred_latent,
                    noised_inputs, {
                        'occupancy': surface_ret['occupancy'],
                        'canonical_occupancy':
                        surface_ret['canonical_occupancy']
                    },
                    pred_params_for_quasi_sdf_loss,
                    detach_transformation=self.
                    is_detach_transformation_for_occupancy_reconstruction,
                    generator_kwargs=pred_generator_kwargs)
                surface_moved_values = surface_moved_values_ret[
                    'occupancy'].view(noised_inputs.size(0), -1,
                                      self.surface_quasi_sdf_points_per_point,
                                      self.primitive_num).mean(-2)
                surface_moved_canonical_values = surface_moved_values_ret[
                    'canonical_occupancy'].view(
                        noised_inputs.size(0), -1,
                        self.surface_quasi_sdf_points_per_point,
                        self.primitive_num).mean(-2)

        else:
            moved_values = self.get_moved_occupancy_value(
                pred_latent,
                points_expand, {'occupancy': pred_values},
                pred_params,
                detach_transformation=self.
                is_detach_transformation_for_occupancy_reconstruction,
                generator_kwargs=pred_generator_kwargs)['occupancy']

            if self.use_surface_quasi_sdf_loss:
                surface_moved_values_ret = self.get_moved_occupancy_value(
                    pred_latent,
                    noised_inputs, {'occupancy': surface_ret['occupancy']},
                    pred_params_for_quasi_sdf_loss,
                    detach_transformation=self.
                    is_detach_transformation_for_occupancy_reconstruction,
                    generator_kwargs=pred_generator_kwargs)
                surface_moved_values = surface_moved_values_ret[
                    'occupancy'].view(noised_inputs.size(0), -1,
                                      self.surface_quasi_sdf_points_per_point,
                                      self.primitive_num).mean(-2)

        ## recon loss
        if self.use_weight_for_positive_values_for_implicit_reconstruction_loss:
            weight = (
                values >= self.
                weight_for_positive_values_for_implicit_reconstruction_loss_threshold
            ) * self.weight_for_positive_values_for_implicit_reconstruction_loss_scale
        else:
            weight = None
        if self.use_occ_recon_loss_for_occ_and_tsdf_for_param:
            if self.use_moved_pretrained_feature_canonical_loss:
                detached_moved_values_ret = self.get_moved_occupancy_value(
                    pred_latent,
                    points, {
                        'occupancy': pred_values,
                        'canonical_occupancy': ret['canonical_occupancy']
                    },
                    pred_params,
                    detach_transformation=True,
                    generator_kwargs=pred_generator_kwargs)
                detached_moved_values = detached_moved_values_ret['occupancy']
            else:
                detached_moved_values = self.get_moved_occupancy_value(
                    pred_latent,
                    points, {'occupancy': pred_values},
                    pred_params,
                    detach_transformation=True,
                    generator_kwargs=pred_generator_kwargs)['occupancy']

            occ_reconstruction_losses, occ_loss_weighted = self.get_implicit_reconstruction_loss(
                detached_moved_values *
                self.occ_recon_loss_for_occ_and_tsdf_for_param_occinize_scale,
                (values >= 0).float(),
                weight=weight)
            losses.update(occ_reconstruction_losses)
            total_G_loss = total_G_loss + occ_loss_weighted

            tsdf_reconstruction_losses, tsdf_loss_weighted = self.get_tsdf_reconstruction_loss(
                moved_values, values)

            losses.update(tsdf_reconstruction_losses)
            total_G_loss = total_G_loss + tsdf_loss_weighted
        else:
            if self.is_add_tie_breaking_noise_in_occ_recon:
                noise_added_values = moved_values + torch.randn_like(
                    moved_canonical_values
                ) * self.tie_breaking_noise_in_occ_recon_scale
            else:
                noise_added_values = moved_values
            if self.model.is_expand_rotation_for_euler_angle:
                values_expanded = values.unsqueeze(1).expand(
                    values.size(0),
                    (self.model.expand_rotation_for_euler_angle_sample_num +
                     1), -1)
                noise_added_values = noise_added_values.view(
                    values.size(0),
                    (self.model.expand_rotation_for_euler_angle_sample_num +
                     1), values.size(1), self.primitive_num)
            else:
                values_expanded = values
            if self.use_canonical_diff_prior_recon_loss:
                noise_added_values = torch.sigmoid(
                    moved_values) * torch.sigmoid(moved_canonical_values)
                without_bce_logits = True
            else:
                without_bce_logits = False
            reconstruction_losses, occ_loss_weighted = self.get_implicit_reconstruction_loss(
                noise_added_values,
                values_expanded,
                weight=weight,
                disable_logits_bce=without_bce_logits,
                is_reduction=(
                    not self.model.is_expand_rotation_for_euler_angle))
            if self.model.is_expand_rotation_for_euler_angle:
                occ_loss_weighted = occ_loss_weighted.min(1)[0]
                if not (self.use_imnet_style_occupancy_loss_with_logit
                        and self.use_tsdf_in_occupancy_loss):
                    occ_loss_weighted = occ_loss_weighted.sum(-1).mean()
                else:
                    occ_loss_weighted = occ_loss_weighted.mean()
                reconstruction_losses['occ_loss_weighted'] = occ_loss_weighted

                occ_loss = reconstruction_losses['occ_loss'].min(1)[0]
                if not (self.use_imnet_style_occupancy_loss_with_logit
                        and self.use_tsdf_in_occupancy_loss):
                    occ_loss = occ_loss.sum(-1).mean()
                else:
                    occ_loss = occ_loss.mean()
                reconstruction_losses['occ_loss'] = occ_loss

            losses.update(reconstruction_losses)
            total_G_loss = total_G_loss + occ_loss_weighted

        if self.use_recon_canonical_shape_loss:
            if self.is_add_tie_breaking_noise_in_occ_recon:
                noise_added_values = moved_canonical_values + torch.randn_like(
                    moved_canonical_values
                ) * self.tie_breaking_noise_in_occ_recon_scale
            else:
                noise_added_values = moved_canonical_values
            if self.model.is_expand_rotation_for_euler_angle:
                values_expanded = values.unsqueeze(1).expand(
                    values.size(0),
                    (self.model.expand_rotation_for_euler_angle_sample_num +
                     1), -1)
                noise_added_values = noise_added_values.view(
                    values.size(0),
                    (self.model.expand_rotation_for_euler_angle_sample_num +
                     1), values.size(1), self.primitive_num)
            else:
                values_expanded = values
            canonical_reconstruction_losses, canonical_occ_loss_weighted = self.get_implicit_reconstruction_loss(
                noise_added_values,
                values_expanded,
                loss_weight=self.recon_canonical_shape_loss_weight,
                weight=weight,
                is_reduction=(
                    not self.model.is_expand_rotation_for_euler_angle),
                prefix='canonical')

            if self.model.is_expand_rotation_for_euler_angle:
                canonical_occ_loss_weighted = canonical_occ_loss_weighted.min(
                    1)[0]
                if not (self.use_imnet_style_occupancy_loss_with_logit
                        and self.use_tsdf_in_occupancy_loss):
                    canonical_occ_loss_weighted = canonical_occ_loss_weighted.sum(
                        -1).mean()
                else:
                    canonical_occ_loss_weighted = canonical_occ_loss_weighted.mean(
                    )
                canonical_reconstruction_losses[
                    'canonical_occ_loss_weighted'] = canonical_occ_loss_weighted

                canonical_occ_loss = canonical_reconstruction_losses[
                    'canonical_occ_loss'].min(1)[0]
                if not (self.use_imnet_style_occupancy_loss_with_logit
                        and self.use_tsdf_in_occupancy_loss):
                    canonical_occ_loss = canonical_occ_loss.sum(-1).mean()
                else:
                    canonical_occ_loss = canonical_occ_loss.mean()
                canonical_reconstruction_losses[
                    'canonical_occ_loss'] = canonical_occ_loss

            losses.update(canonical_reconstruction_losses)
            total_G_loss = total_G_loss + canonical_occ_loss_weighted

        if self.use_surface_quasi_sdf_loss:
            surface_quasi_sdf_losses, surface_quasi_sdf_loss_weighted = self.get_surface_quasi_sdf_loss(
                surface_moved_values, 0.5)
            losses.update(surface_quasi_sdf_losses)
            total_G_loss = total_G_loss + surface_quasi_sdf_loss_weighted

        if is_latent_quantize_diff:
            latent_quantize_diff_loss = pred_latent_quantize_diff
            latent_quantize_diff_loss_weighted = latent_quantize_diff_loss * self.latent_quantize_diff_loss_weight
            losses.update(
                dict(latent_quantize_diff_loss=latent_quantize_diff_loss,
                     latent_quantize_diff_loss_weighted=
                     latent_quantize_diff_loss_weighted))
            total_G_loss = total_G_loss + latent_quantize_diff_loss_weighted

        if not self.disable_gan_training:
            g_loss_internal_weight = 0
            g_loss_internal_weight_inv = float(
                self.use_learn_only_generator_with_canonical_shape +
                (not self.skip_learn_generator_with_recon_shape) +
                self.use_learn_generator_with_randomly_moved_shape +
                self.use_canonical_shape_for_gan_training)
            if g_loss_internal_weight_inv > 0:
                g_loss_internal_weight = 1. / g_loss_internal_weight_inv

            if self.use_learn_only_generator_with_canonical_shape:
                canonical_discriminator_ret = self.get_discriminator_result(
                    pred_values,
                    points,
                    None,
                    inputs,
                    no_move=True,
                    detach_param=self.is_detach_param_in_generator,
                    detach_func=self.dict_detach_func)

                ### Generator
                canonical_G_real = -canonical_discriminator_ret['D']
                if self.gan_type == 'lsgan':
                    canonical_G_loss = F.mse_loss(
                        canonical_G_real,
                        torch.ones_like(canonical_G_real),
                        reduction='mean')
                elif self.gan_type == 'wgangp':
                    canonical_G_loss = canonical_G_real.mean()
                else:
                    raise NotImplementedError

                losses['G_loss_canonical'] = canonical_G_loss

                canonical_G_loss_weighted = canonical_G_loss * self.G_loss_weight * g_loss_internal_weight
                losses['G_loss_weighted_canonical'] = canonical_G_loss_weighted

                total_G_loss = total_G_loss + canonical_G_loss_weighted

            if self.use_canonical_shape_for_gan_training:
                canonical_shape_discriminator_ret = self.get_discriminator_result(
                    pred_values,
                    points,
                    None,
                    inputs,
                    no_move=True,
                    detach_param=self.is_detach_param_in_generator,
                    detach_func=self.dict_detach_func)

                ### Generator
                canonical_shape_G_real = -canonical_shape_discriminator_ret['D']
                if self.gan_type == 'lsgan':
                    canonical_shape_G_loss = F.mse_loss(
                        canonical_shape_G_real,
                        torch.ones_like(canonical_shape_G_real),
                        reduction='mean')
                elif self.gan_type == 'wgangp':
                    canonical_shape_G_loss = canonical_shape_G_real.mean()
                else:
                    raise NotImplementedError

                losses['G_loss_canonical_shape'] = canonical_shape_G_loss

                canonical_shape_G_loss_weighted = canonical_shape_G_loss * self.G_loss_weight * g_loss_internal_weight
                losses[
                    'G_loss_weighted_canonical_shape'] = canonical_shape_G_loss_weighted

                total_G_loss = total_G_loss + canonical_shape_G_loss_weighted

            # Align to canonical pose
            if not self.skip_learn_generator_with_recon_shape:
                discriminator_ret = self.get_discriminator_result(
                    moved_values,
                    points,
                    None,
                    inputs,
                    no_move=True,
                    detach_param=self.is_detach_param_in_generator,
                    detach_func=self.dict_detach_func)

                ### Generator
                G_real = -discriminator_ret['D']
                if self.gan_type == 'lsgan':
                    G_loss = F.mse_loss(G_real,
                                        torch.ones_like(G_real),
                                        reduction='mean')
                elif self.gan_type == 'wgangp':
                    G_loss = G_real.mean()
                else:
                    raise NotImplementedError

                losses['G_loss'] = G_loss

                G_loss_weighted = G_loss * self.G_loss_weight * g_loss_internal_weight
                losses['G_loss_weighted'] = G_loss_weighted

                total_G_loss = total_G_loss + G_loss_weighted

            if self.use_learn_generator_with_randomly_moved_shape:
                sampled_transform_param = self.sample_transform_param(
                    pred_params)

                if self.use_voxel_discriminator:
                    points_or_voxel_grid = voxel_grid
                else:
                    points_or_voxel_grid = points
                if self.use_motion_gan_loss:
                    if self.is_freeze_generator_in_motion_gan_training:
                        with train_util.freeze_models([self.model.generator]):
                            randomly_moved_values = self.get_moved_occupancy_value(
                                pred_latent.clone().detach(),
                                points_or_voxel_grid,
                                {'occupancy': pred_values.clone().detach()},
                                sampled_transform_param,
                                generator_kwargs=pred_generator_kwargs
                            )['occupancy']
                    else:
                        randomly_moved_values = self.get_moved_occupancy_value(
                            pred_latent.clone().detach(),
                            points_or_voxel_grid,
                            {'occupancy': pred_values.clone().detach()},
                            sampled_transform_param,
                            generator_kwargs=pred_generator_kwargs
                        )['occupancy']
                else:
                    randomly_moved_values = self.get_moved_occupancy_value(
                        pred_latent,
                        points_or_voxel_grid, {'occupancy': pred_values},
                        sampled_transform_param,
                        generator_kwargs=pred_generator_kwargs)['occupancy']

                # Align to canonical pose
                randomly_moved_discriminator_ret = self.get_discriminator_result(
                    randomly_moved_values,
                    points_or_voxel_grid,
                    None,
                    inputs,
                    no_move=True,
                    detach_param=self.is_detach_param_in_generator,
                    detach_func=self.dict_detach_func)

                ### Generator
                randomly_moved_G_real = -randomly_moved_discriminator_ret['D']
                if self.gan_type == 'lsgan':
                    randomly_moved_G_loss = F.mse_loss(
                        randomly_moved_G_real,
                        torch.ones_like(randomly_moved_G_real),
                        reduction='mean')
                elif self.gan_type == 'wgangp':
                    randomly_moved_G_loss = randomly_moved_G_real.mean()
                else:
                    raise NotImplementedError

                losses['G_loss_randomly_moved'] = randomly_moved_G_loss
                randomly_moved_G_loss_weighted = randomly_moved_G_loss * self.G_loss_weight * g_loss_internal_weight
                losses[
                    'G_loss_weighted_randomly_moved'] = randomly_moved_G_loss_weighted
                total_G_loss = total_G_loss + randomly_moved_G_loss_weighted

        if self.use_explicit_pointcloud_loss_for_motion:
            if self.model.is_generator_return_dict:
                pred_values_explicit_pointcloud_loss_for_motion = ret[
                    'canonical_occupancy']
            else:
                pred_values_explicit_pointcloud_loss_for_motion = pred_values
            moved_points = self.move_points(
                pred_values_explicit_pointcloud_loss_for_motion,
                points,
                pred_params,
                force_move=True)
            canonical_input_mask = pred_values_explicit_pointcloud_loss_for_motion.max(
                dim=-1, keepdim=True)[0] >= 0.5
            target_input_mask = (values.unsqueeze(-1) > 0)
            explicit_pointcloud_loss_for_motion = chamfer_loss.chamfer_loss(
                moved_points,
                points,
                source_mask=canonical_input_mask,
                target_mask=target_input_mask)
            explicit_pointcloud_loss_for_motion_weighted = explicit_pointcloud_loss_for_motion * self.explicit_pointcloud_loss_for_motion_weight
            losses.update(
                dict(explicit_pointcloud_loss_for_motion_weighted=
                     explicit_pointcloud_loss_for_motion_weighted,
                     explicit_pointcloud_loss_for_motion=
                     explicit_pointcloud_loss_for_motion))
            total_G_loss = total_G_loss + explicit_pointcloud_loss_for_motion_weighted

        if self.use_self_supervised_motion_learning:
            sampled_transform_param_for_motion_learning = self.sample_transform_param(
                pred_params)
            moved_inputs = self.move_points(
                ret['canonical_occupancy'],
                inputs,
                sampled_transform_param_for_motion_learning,
                force_move=True)
            self_supervised_param = self.model(moved_inputs,
                                               points,
                                               return_occupancy=False)['param']
            self_supervised_motion_learning_line_distance_loss = geometry.get_line_to_line_distance(
                self_supervised_param['rotation_anchor_point'],
                self_supervised_param['rotation_direction'],
                sampled_transform_param_for_motion_learning[
                    'rotation_anchor_point'].clone().detach(),
                sampled_transform_param_for_motion_learning[
                    'rotation_direction'].clone().detach()).mean()

            # 0 perfecto match, 2, opposite match
            self_supervised_motion_learning_direction_loss = 1 - geometry.get_direction_error(
                self_supervised_param['rotation_direction'],
                sampled_transform_param_for_motion_learning[
                    'rotation_direction'].clone().detach()).mean()

            assert self.is_constant_motion_range
            self_supervised_motion_learning_rotation_loss = F.mse_loss(
                geometry.get_quaternion(
                    self_supervised_param['rotation_direction'],
                    self_supervised_param['rotation_amount'] * np.pi),
                geometry.get_quaternion(
                    sampled_transform_param_for_motion_learning[
                        'rotation_direction'].clone().detach(),
                    sampled_transform_param_for_motion_learning[
                        'rotation_amount'].clone().detach() * np.pi))

            self_supervised_motion_learning_line_distance_loss_weighted = self_supervised_motion_learning_line_distance_loss * self.self_supervised_motion_learning_line_distance_loss_weight
            total_G_loss = total_G_loss + self_supervised_motion_learning_line_distance_loss_weighted
            losses.update(
                dict(
                    self_supervised_motion_learning_line_distance_loss_weighted=
                    self_supervised_motion_learning_line_distance_loss_weighted,
                    self_supervised_motion_learning_line_distance_loss=
                    self_supervised_motion_learning_line_distance_loss))
            self_supervised_motion_learning_direction_loss_weighted = self_supervised_motion_learning_direction_loss * self.self_supervised_motion_learning_line_direction_loss_weight
            total_G_loss = total_G_loss + self_supervised_motion_learning_direction_loss_weighted
            losses.update(
                dict(self_supervised_motion_learning_direction_loss_weighted=
                     self_supervised_motion_learning_direction_loss_weighted,
                     self_supervised_motion_learning_direction_loss=
                     self_supervised_motion_learning_direction_loss))
            self_supervised_motion_learning_rotation_loss_weighted = self_supervised_motion_learning_rotation_loss * self.self_supervised_motion_learning_line_rotation_loss_weight
            total_G_loss = total_G_loss + self_supervised_motion_learning_rotation_loss_weighted
            losses.update(
                dict(self_supervised_motion_learning_rotation_loss_weighted=
                     self_supervised_motion_learning_rotation_loss_weighted,
                     self_supervised_motion_learning_rotation_loss=
                     self_supervised_motion_learning_rotation_loss))

        if self.use_part_shape_similarity_loss:
            part_shape_similarity_loss_ret, part_shape_similarity_loss_weighted = self.get_part_shape_similarity_loss_no_moving_point(
                points, pred_values)
            part_shape_similarity_loss_ret = {
                'G_' + name: value
                for name, value in part_shape_similarity_loss_ret.items()
            }
            total_G_loss = total_G_loss + part_shape_similarity_loss_weighted
            losses.update(part_shape_similarity_loss_ret)

        if self.use_canonical_similarity_loss:
            cano_sim_part_shape_similarity_loss_ret, cano_sim_part_shape_similarity_loss_weighted = self.get_part_shape_canonical_similarity(
                points, pred_values, moved_values)
            cano_sim_part_shape_similarity_loss_ret = {
                'G_' + name: value
                for name, value in
                cano_sim_part_shape_similarity_loss_ret.items()
            }
            total_G_loss = total_G_loss + cano_sim_part_shape_similarity_loss_weighted
            losses.update(cano_sim_part_shape_similarity_loss_ret)

        if self.use_anchor_position_loss:
            anchor_position_loss_ret, anchor_position_loss_weighted = self.get_anchor_position_loss(
                points, pred_values, pred_params['rotation_anchor_point'])
            total_G_loss = total_G_loss + anchor_position_loss_weighted
            losses.update(anchor_position_loss_ret)

        if self.use_anchor_position_near_gt_shape_loss:
            rotation_anchor_point = pred_params['rotation_anchor_point']
            if self.model.is_expand_rotation_for_euler_angle:
                rotation_anchor_point = rotation_anchor_point.view(
                    points.size(0),
                    self.model.expand_rotation_for_euler_angle_sample_num + 1,
                    rotation_anchor_point.size(-2),
                    rotation_anchor_point.size(-1))[:, 0, :, :]
            anchor_position_near_gt_shape_loss_ret, anchor_position_near_gt_shape_loss_weighted = self.get_anchor_position_near_gt_shape_loss(
                points,
                values,
                rotation_anchor_point,
                pred_values=moved_values)
            total_G_loss = total_G_loss + anchor_position_near_gt_shape_loss_weighted
            losses.update(anchor_position_near_gt_shape_loss_ret)

        if self.use_raw_anchor_position_near_gt_shape_loss:
            rotation_anchor_point = pred_params['rotation_anchor_point']
            if self.model.is_expand_rotation_for_euler_angle:
                raise NotImplementedError
            raw_anchor_position_near_gt_shape_loss_ret_temp, raw_anchor_position_near_gt_shape_loss_weighted = self.get_anchor_position_near_gt_shape_loss(
                points,
                values,
                ret['raw_param']['canonical_location']
                [:, 1:self.model.rotation_primitive_num + 1, :],
                pred_values=moved_values)
            raw_anchor_position_near_gt_shape_loss_ret = {}
            raw_anchor_position_near_gt_shape_loss_ret[
                'raw_anchor_position_near_gt_shape_loss'] = raw_anchor_position_near_gt_shape_loss_ret_temp[
                    'anchor_position_near_gt_shape_loss']
            raw_anchor_position_near_gt_shape_loss_ret[
                'raw_anchor_position_near_gt_shape_loss_weighted'] = raw_anchor_position_near_gt_shape_loss_ret_temp[
                    'anchor_position_near_gt_shape_loss_weighted']
            total_G_loss = total_G_loss + raw_anchor_position_near_gt_shape_loss_weighted
            losses.update(raw_anchor_position_near_gt_shape_loss_ret)

        if self.use_minimize_raw_canonical_location_to_offset_loss:
            minimize_raw_canonical_location_to_offset_loss_ret, minimize_raw_canonical_location_to_offset_loss_weighted = self.get_minimize_raw_canonical_location_to_offset_loss(
                pred_params['canonical_location'],
                ret['raw_param']['canonical_location'])
            total_G_loss = total_G_loss + minimize_raw_canonical_location_to_offset_loss_weighted
            losses.update(minimize_raw_canonical_location_to_offset_loss_ret)

        if self.use_minimize_raw_canonical_direction_to_offset_loss:
            minimize_raw_canonical_direction_to_offset_loss_ret, minimize_raw_canonical_direction_to_offset_loss_weighted = self.get_minimize_raw_canonical_direction_to_offset_loss(
                pred_params['canonical_direction'],
                ret['raw_param']['canonical_direction'])
            total_G_loss = total_G_loss + minimize_raw_canonical_direction_to_offset_loss_weighted
            losses.update(minimize_raw_canonical_direction_to_offset_loss_ret)

        if self.use_screw_motion_spectrum_entropy_loss:
            screw_motion_spectrum_entropy_loss_ret, screw_motion_spectrum_entropy_loss_weighted = self.get_screw_motion_spectrum_entropy_loss(
                ret['raw_param']['motion_spectrum'])
            total_G_loss = total_G_loss + screw_motion_spectrum_entropy_loss_weighted
            losses.update(screw_motion_spectrum_entropy_loss_ret)

        if self.use_overlap_regularizer:
            overlap_regularizer_losses, total_overlap_regularizer_loss_weighted = self.get_regularizer_overlap(
                pred_values, moved_values)
            total_G_loss = total_G_loss + total_overlap_regularizer_loss_weighted
            losses.update(overlap_regularizer_losses)

        if self.use_overlap_regularizer_loss:
            overlap_regularizer_loss_ret, overlap_regularizer_loss_weighted = self.get_overlap_regularizer_loss(
                pred_values)
            total_G_loss = total_G_loss + overlap_regularizer_loss_weighted
            losses.update(overlap_regularizer_loss_ret)

        if self.use_moved_overlap_regularizer_loss:
            moved_overlap_regularizer_loss_ret, moved_overlap_regularizer_loss_weighted = self.get_overlap_regularizer_loss(
                moved_values, prefix='moved')
            total_G_loss = total_G_loss + moved_overlap_regularizer_loss_weighted
            losses.update(moved_overlap_regularizer_loss_ret)

        if self.use_randomly_moved_overlap_regularizer_loss:
            randomly_moved_overlap_regularizer_loss_ret, randomly_moved_overlap_regularizer_loss_weighted = self.get_overlap_regularizer_loss(
                randomly_moved_values, prefix='randomly_moved')
            total_G_loss = total_G_loss + randomly_moved_overlap_regularizer_loss_weighted
            losses.update(randomly_moved_overlap_regularizer_loss_ret)

        if self.use_volume_preserving_loss:
            volume_preserving_loss_ret, volume_preserving_loss_weighted = self.get_volume_preserving_loss(
                pred_values, moved_values)
            total_G_loss = total_G_loss + volume_preserving_loss_weighted
            losses.update(volume_preserving_loss_ret)

        if self.use_soft_volume_preserving_loss:
            soft_volume_preserving_loss_ret, soft_volume_preserving_loss_weighted = self.get_soft_volume_preserving_loss(
                pred_values, moved_values)
            total_G_loss = total_G_loss + soft_volume_preserving_loss_weighted
            losses.update(soft_volume_preserving_loss_ret)

        if self.use_primitive_number_loss:
            primitive_number_loss_ret, weighted_primitive_number_loss = self.get_primitive_number_loss(
                pred_values)
            losses.update(primitive_number_loss_ret)
            total_G_loss = total_G_loss + weighted_primitive_number_loss

            moved_primitive_number_loss_ret, moved_weighted_primitive_number_loss = self.get_primitive_number_loss(
                moved_values, '_moved')
            losses.update(moved_primitive_number_loss_ret)
            total_G_loss = total_G_loss + moved_weighted_primitive_number_loss

        if self.use_entropy_reduction_loss:
            entropy_reduction_loss_ret, entropy_reduction_loss_weighted = self.get_shape_entropy_loss(
                moved_values,
                points,
                gt_occ=(
                    values if
                    self.use_gt_values_as_whole_shape_in_entropy_reduction_loss
                    else None))
            total_G_loss = total_G_loss + entropy_reduction_loss_weighted
            losses.update(entropy_reduction_loss_ret)

        if self.use_pretrained_feature_canonical_loss:
            pretrained_feature_canonical_loss_ret, pretrained_feature_canonical_loss_weighted = self.get_pretrained_feature_canonical_loss(
                pred_values, ret['canonical_occupancy'], points)
            total_G_loss = total_G_loss + pretrained_feature_canonical_loss_weighted
            losses.update(pretrained_feature_canonical_loss_ret)

        if self.use_moved_pretrained_feature_canonical_loss:
            moved_pretrained_feature_canonical_loss_ret, moved_pretrained_feature_canonical_loss_weighted = self.get_pretrained_feature_canonical_loss(
                moved_values, moved_canonical_values, points)
            total_G_loss = total_G_loss + moved_pretrained_feature_canonical_loss_weighted
            losses.update(moved_pretrained_feature_canonical_loss_ret)

        if self.use_rotation_anchor_point_similarity_loss:
            rotation_anchor_point = pred_params['rotation_anchor_point']
            if self.model.is_expand_rotation_for_euler_angle:
                rotation_anchor_point = rotation_anchor_point.view(
                    points.size(0),
                    self.model.expand_rotation_for_euler_angle_sample_num + 1,
                    rotation_anchor_point.size(-2),
                    rotation_anchor_point.size(-1))[:, 0, :, :]
            rotation_anchor_point_similarity_loss_ret, rotation_anchor_point_similarity_loss_weighted = self.get_rotation_anchor_point_similarity_loss(
                rotation_anchor_point)
            total_G_loss = total_G_loss + rotation_anchor_point_similarity_loss_weighted
            losses.update(rotation_anchor_point_similarity_loss_ret)

        if self.use_rotation_anchor_point_occupancy_relation_loss:
            rotation_anchor_point_value = self.model(
                pred_latent,
                pred_params['rotation_anchor_point'],
                direct_input_to_decoder=True,
                return_param=False)['occupancy']
            rotation_anchor_point_occupancy_relation_loss_ret, rotation_anchor_point_occupancy_relation_loss_weighted = self.get_rotation_anchor_point_occupancy_relation_loss(
                rotation_anchor_point_value)
            total_G_loss = total_G_loss + rotation_anchor_point_occupancy_relation_loss_weighted
            losses.update(rotation_anchor_point_occupancy_relation_loss_ret)

        if self.use_bone_occupancy_loss:
            bone_occupancy_loss_ret, bone_occupancy_loss_weighted, bone_occupancy_values = self.get_bone_occupancy_loss(
                pred_latent, pred_params)
            total_G_loss = total_G_loss + bone_occupancy_loss_weighted
            losses.update(bone_occupancy_loss_ret)

        if self.use_canonical_location_near_gt_loss:
            canonical_location_near_gt_loss_ret, canonical_location_near_gt_loss_weighted = self.get_canonical_location_near_gt_loss(
                points, values, pred_params)
            total_G_loss = total_G_loss + canonical_location_near_gt_loss_weighted
            losses.update(canonical_location_near_gt_loss_ret)

        if self.use_explicit_pointcloud_loss_for_motion_with_surface_points:
            explicit_pointcloud_loss_for_motion_with_surface_points_ret, explicit_pointcloud_loss_for_motion_with_surface_points_weighted = self.get_explicit_pointcloud_loss_for_motion_with_surface_points(
                ret['surface_points'], inputs, pred_params)
            total_G_loss = total_G_loss + explicit_pointcloud_loss_for_motion_with_surface_points_weighted
            losses.update(
                explicit_pointcloud_loss_for_motion_with_surface_points_ret)

        if self.use_motion_amount_inverse_std_loss:
            motion_amount_inverse_std_loss_ret, motion_amount_inverse_std_loss = self.get_motion_amount_inverse_std_loss(
                pred_params)
            total_G_loss = total_G_loss + motion_amount_inverse_std_loss
            losses.update(motion_amount_inverse_std_loss_ret)

        if self.use_location_offset_regulariztion_loss:
            location_offset_regulariztion_loss_ret, location_offset_regulariztion_loss_weighted = self.get_location_offset_regulariztion_loss(
                ret['raw_param']['canonical_location_offset'])
            total_G_loss = total_G_loss + location_offset_regulariztion_loss_weighted
            losses.update(location_offset_regulariztion_loss_ret)

        if self.use_anchor_position_chain_assumption_loss:
            anchor_position_chain_assumption_loss_ret, anchor_position_chain_assumption_loss_weighted = self.get_anchor_position_chain_assumption_loss(
                points, moved_values, pred_params['rotation_anchor_point'])
            total_G_loss = total_G_loss + anchor_position_chain_assumption_loss_weighted
            losses.update(anchor_position_chain_assumption_loss_ret)

        nan_losses = []
        for losskey, lossvalue in losses.items():
            if 'weighted' in losskey and torch.any(torch.isnan(lossvalue)):
                nan_losses.append(losskey)

        if nan_losses:
            print('########## NaN loss occurred!!! ###########')
            for key in nan_losses:
                print(key)

        losses['total_G_loss'] = total_G_loss

        if self.is_check_gradient_scale and total_G_loss.requires_grad:
            """
            ret['total_values'] = pred_values + moved_values
            ret['rotation_anchor_point'] = ret['param'][
                'rotation_anchor_point']
            ret['rotation_direction'] = ret['param']['rotation_direction']
            ret['rotation_amount'] = ret['param']['rotation_amount']
            grad_ret = self.check_gradient_scale(
                losses,
                ret,
                ret_keys=[
                    #'total_values',
                    #'raw_param',
                    'rotation_anchor_point',
                    'rotation_direction',
                    'rotation_amount'
                ],
                prefix='G/')

            losses.update(grad_ret)
            """
            values_ret = dict(pred_values=pred_values,
                              moved_values=moved_values,
                              canonical_values=ret['canonical_occupancy'])
            if self.use_learn_generator_with_randomly_moved_shape and not self.disable_gan_training:
                values_ret['randomly_moved_values'] = randomly_moved_values
            if self.use_rotation_anchor_point_occupancy_relation_loss:
                values_ret[
                    'rotation_anchor_point_value'] = rotation_anchor_point_value
            if self.use_bone_occupancy_loss:
                values_ret['bone_occupancy_values'] = bone_occupancy_values
            if self.use_moved_pretrained_feature_canonical_loss or self.use_recon_canonical_shape_loss:
                values_ret['moved_canonical_values'] = moved_canonical_values
            """
            values_ret['canonical_location'] = ret['raw_param'][
                'canonical_location']
            values_ret['canonical_location_offset'] = ret['raw_param'][
                'canonical_location_offset']
            """
            values_ret['rotation_amount'] = ret['raw_param']['rotation_amount']
            grad_ret2 = self.check_gradient_scale(losses,
                                                  values_ret,
                                                  prefix='G/')
            losses.update(grad_ret2)

            values_ret = {}
            if self.use_explicit_pointcloud_loss_for_motion_with_surface_points:
                values_ret['surface_points'] = ret['surface_points']
                values_ret['canonical_rotation_matrix'] = pred_params[
                    'canonical_rotation_matrix']
                grad_ret3 = self.check_gradient_scale(losses,
                                                      values_ret,
                                                      prefix='G/')
                losses.update(grad_ret3)

        prefixed_losses = {(prefix + name): value
                           for name, value in losses.items()}
        return prefixed_losses

    def train_implicit_canonical_learning_discriminator(
            self,
            values,
            points,
            inputs,
            use_gp=False,
            prefix='',
            voxels=None,
            voxel_grid=None):
        disc_losses = {}
        total_D_loss = 0.  #torch.zeros_like(points_A.sum())
        with torch.no_grad():
            ret = self.model(inputs, points)

        pred_values = ret['occupancy']
        pred_params = ret['param']
        pred_latent = ret['latent']
        pred_generator_kwargs = ret['generator_kwargs']

        if self.model.is_expand_rotation_for_euler_angle:
            points_expand = points.unsqueeze(1).expand(
                -1, self.model.expand_rotation_for_euler_angle_sample_num + 1,
                -1, -1).contiguous().view(
                    points.size(0) *
                    (self.model.expand_rotation_for_euler_angle_sample_num +
                     1), points.size(-2), points.size(-1))
        else:
            points_expand = points

        self.preprocess_transformation(ret, points, points_expand, pred_values,
                                       pred_params, pred_latent,
                                       pred_generator_kwargs)
        d_loss_internal_weight = 0
        d_loss_internal_weight_inv = float(
            self.use_learn_only_generator_with_canonical_shape +
            (not self.skip_learn_generator_with_recon_shape) +
            self.use_learn_generator_with_randomly_moved_shape +
            self.use_canonical_shape_for_gan_training)
        if d_loss_internal_weight_inv > 0:
            d_loss_internal_weight = 1. / d_loss_internal_weight_inv

        if not self.skip_learn_generator_with_recon_shape or self.use_learn_only_generator_with_canonical_shape:
            moved_values = self.get_moved_occupancy_value(
                pred_latent,
                points_expand, {'occupancy': pred_values},
                pred_params,
                generator_kwargs=pred_generator_kwargs)['occupancy']

        if self.use_learn_only_generator_with_canonical_shape:

            canonical_discriminator_ret = self.get_discriminator_result(
                moved_values, points, None, inputs, no_move=True, detach=True)

            canonical_D_fake = canonical_discriminator_ret['D']

            if self.gan_type == 'lsgan':
                canonical_D_fake_loss = F.mse_loss(
                    canonical_D_fake,
                    torch.zeros_like(canonical_D_fake),
                    reduction='mean')
            elif self.gan_type == 'wgangp':
                canonical_D_fake_loss = canonical_D_fake.mean()
            else:
                raise NotImplementedError
            canonical_D_fake_loss_weighted = canonical_D_fake_loss * self.D_fake_loss_weight * self.D_loss_weight * d_loss_internal_weight

            total_D_loss = total_D_loss + canonical_D_fake_loss_weighted

            disc_losses.update({
                'canonical_D_fake_loss':
                canonical_D_fake_loss,
                'canonical_D_fake_loss_weighted':
                canonical_D_fake_loss_weighted
            })

        if self.use_canonical_shape_for_gan_training:
            canonical_shape_discriminator_ret = self.get_discriminator_result(
                pred_values, points, None, inputs, no_move=True, detach=True)

            canonical_shape_D_fake = canonical_shape_discriminator_ret['D']

            if self.gan_type == 'lsgan':
                canonical_shape_D_fake_loss = F.mse_loss(
                    canonical_shape_D_fake,
                    torch.zeros_like(canonical_shape_D_fake),
                    reduction='mean')
            elif self.gan_type == 'wgangp':
                canonical_shape_D_fake_loss = canonical_shape_D_fake.mean()
            else:
                raise NotImplementedError
            canonical_shape_D_fake_loss_weighted = canonical_shape_D_fake_loss * self.D_fake_loss_weight * self.D_loss_weight * d_loss_internal_weight

            total_D_loss = total_D_loss + canonical_shape_D_fake_loss_weighted

            disc_losses.update({
                'canonical_shape_D_fake_loss':
                canonical_shape_D_fake_loss,
                'canonical_shape_D_fake_loss_weighted':
                canonical_shape_D_fake_loss_weighted
            })

        if not self.skip_learn_generator_with_recon_shape:
            discriminator_ret = self.get_discriminator_result(moved_values,
                                                              points,
                                                              None,
                                                              inputs,
                                                              no_move=True,
                                                              detach=True)

            D_fake = discriminator_ret['D']

            if self.gan_type == 'lsgan':
                D_fake_loss = F.mse_loss(D_fake,
                                         torch.zeros_like(D_fake),
                                         reduction='mean')
            elif self.gan_type == 'wgangp':
                D_fake_loss = D_fake.mean()
            else:
                raise NotImplementedError
            D_fake_loss_weighted = D_fake_loss * self.D_fake_loss_weight * self.D_loss_weight * d_loss_internal_weight

            total_D_loss = total_D_loss + D_fake_loss_weighted

            disc_losses.update({
                'D_fake_loss': D_fake_loss,
                'D_fake_loss_weighted': D_fake_loss_weighted
            })

        if self.use_learn_generator_with_randomly_moved_shape:
            sampled_transform_param = self.sample_transform_param(pred_params)

            if self.use_voxel_discriminator:
                points_or_voxel_grid = voxel_grid
            else:
                points_or_voxel_grid = points
            randomly_moved_values = self.get_moved_occupancy_value(
                pred_latent,
                points_or_voxel_grid, {'occupancy': pred_values},
                sampled_transform_param,
                generator_kwargs=pred_generator_kwargs)['occupancy']

            randomly_moved_discriminator_ret = self.get_discriminator_result(
                randomly_moved_values,
                points_or_voxel_grid,
                None,
                inputs,
                no_move=True,
                detach=True)

            randomly_moved_D_fake = randomly_moved_discriminator_ret['D']

            if self.gan_type == 'lsgan':
                randomly_moved_D_fake_loss = F.mse_loss(
                    randomly_moved_D_fake,
                    torch.zeros_like(randomly_moved_D_fake),
                    reduction='mean')
            elif self.gan_type == 'wgangp':
                randomly_moved_D_fake_loss = randomly_moved_D_fake.mean()
            else:
                raise NotImplementedError
            randomly_moved_D_fake_loss_weighted = randomly_moved_D_fake_loss * self.D_fake_loss_weight * self.D_loss_weight * d_loss_internal_weight
            total_D_loss = total_D_loss + randomly_moved_D_fake_loss_weighted

            disc_losses.update({
                'D_fake_loss_randomly_moved':
                randomly_moved_D_fake_loss,
                'D_fake_loss_weighted_randomly_moved':
                randomly_moved_D_fake_loss_weighted
            })

        if self.gradient_penalty_type == 'real_input':
            points_values_requires_grad = True
        else:
            points_values_requires_grad = False

        if self.use_voxel_discriminator:
            points_or_voxel_grid = voxel_grid
            values_or_voxels = voxels
        else:
            points_or_voxel_grid = points
            values_or_voxels = values

        discriminator_real_ret = self.get_real_discriminator_result(
            values_or_voxels,
            points_or_voxel_grid,
            points_values_requires_grad=points_values_requires_grad)
        D_real = discriminator_real_ret['D']
        if self.gan_type == 'lsgan':
            D_real_loss = F.mse_loss(D_real,
                                     torch.ones_like(D_real),
                                     reduction='mean')
        elif self.gan_type == 'wgangp':
            D_real_loss = -D_real.mean()
        else:
            raise NotImplementedError
        D_real_loss_weighted = D_real_loss * self.D_real_loss_weight * self.D_loss_weight

        total_D_loss = total_D_loss + D_real_loss_weighted

        disc_losses.update({
            'D_real_loss': D_real_loss,
            'D_real_loss_weighted': D_real_loss_weighted,
        })

        if self.use_gradient_penalty and use_gp:
            if self.use_learn_generator_with_randomly_moved_shape:
                d_ret = randomly_moved_discriminator_ret
            elif not self.skip_learn_generator_with_recon_shape:
                d_ret = discriminator_ret
            elif self.use_canonical_shape_for_gan_training:
                d_ret = canonical_shape_discriminator_ret

            gradient_penalty_ret, gradient_penalty_weighted = self.get_gradient_penalty(
                d_ret, discriminator_real_ret)
            if self.is_apply_gradient_penalty_as_independent_step:
                if 'independent_D_loss' in disc_losses:
                    disc_losses['independent_D_loss'] = disc_losses[
                        'independent_D_loss'] + gradient_penalty_weighted
                else:
                    disc_losses[
                        'independent_D_loss'] = gradient_penalty_weighted
            else:
                total_D_loss = total_D_loss + gradient_penalty_weighted
            disc_losses.update(gradient_penalty_ret)

        if self.use_part_shape_similarity_loss and not self.disable_part_similarity_loss_in_discriminator:
            part_shape_similarity_loss_ret, part_shape_similarity_loss_weighted = self.get_part_shape_similarity_loss_no_moving_point(
                points, pred_values)
            total_D_loss = total_D_loss + part_shape_similarity_loss_weighted
            part_shape_similarity_loss_ret = {
                'D_' + name: value
                for name, value in part_shape_similarity_loss_ret.items()
            }
            disc_losses.update(part_shape_similarity_loss_ret)

        disc_losses['total_D_loss'] = total_D_loss

        if self.is_check_gradient_scale and False:
            values_ret = dict(pred_values=pred_values)
            if self.use_learn_generator_with_randomly_moved_shape and not self.disable_gan_training:
                values_ret['randomly_moved_values'] = randomly_moved_values
            grad_ret2 = self.check_occupancy_gradient_scale(disc_losses,
                                                            values_ret,
                                                            prefix='D/')
            disc_losses.update(grad_ret2)

        prefixed_losses = {(prefix + name): value
                           for name, value in disc_losses.items()}
        return prefixed_losses

    def get_moved_occupancy_value(self,
                                  latent,
                                  points,
                                  values_ret,
                                  transformation,
                                  detach_transformation=False,
                                  generator_kwargs={}):
        assert not self.is_move_points

        if detach_transformation:
            transformation = {
                key: param.clone().detach()
                for key, param in transformation.items()
            }

        moved_points_moving_points = points.clone()
        ret_values_ret = {}
        for key, values in values_ret.items():
            ret_values = values.clone()
            ret_values_ret[key] = ret_values
        for idx, midx in enumerate(range(1, self.primitive_num)):
            if self.model.motion_decoding_type == 'default':
                raise NotImplementedError("need reconsider")
                primitive_wise_transformation = {
                    key: value[:, idx, ...]
                    for key, value in transformation.items()
                }
                tmp = self.apply_translation(moved_points_moving_points,
                                             primitive_wise_transformation)
                tmp = self.apply_rotation(tmp, primitive_wise_transformation)
            elif self.model.motion_decoding_type in [
                    'one_joint_type_per_primitive',
                    'one_joint_type_per_primitive_rot_pm_num_specified'
            ]:
                if idx < self.model.rotation_primitive_num:
                    primitive_wise_transformation = {
                        key:
                        value[:, midx if key.startswith('canonical') else idx,
                              ...]
                        for key, value in transformation.items()
                        if key.startswith('rotation')
                        or key.startswith('canonical') or
                        (key.startswith('translation_amount')
                         and self.model.param_type ==
                         'only_amount_as_matrix_loc_offset_canonical_motion_euler_angle_screw'
                         )
                    }
                    tmp = self.apply_rotation(moved_points_moving_points,
                                              primitive_wise_transformation)
                else:
                    primitive_wise_transformation = {
                        key:
                        value[:, idx -
                              (-1 if key.startswith('canonical') else self.
                               model.rotation_primitive_num), ...]
                        for key, value in transformation.items()
                        if key.startswith('translation')
                        or key.startswith('canonical')
                    }
                    tmp = self.apply_translation(
                        moved_points_moving_points,
                        primitive_wise_transformation)
            ret = self.model(latent,
                             tmp,
                             return_param=False,
                             direct_input_to_decoder=True,
                             generator_kwargs=generator_kwargs)
            for key, ret_values in ret_values_ret.items():
                ret_values[..., midx] = ret[key][..., midx]

        if self.is_input_pm0_to_motion_net_too:
            ret = self.model(latent,
                             points,
                             return_param=False,
                             direct_input_to_decoder=True,
                             generator_kwargs=generator_kwargs)
            for key, ret_values in ret_values_ret.items():
                ret_values[:, :, 0] = ret[key][:, :, 0]
        if self.is_scale_pm0_values_after_motion:
            for key, ret_values in ret_values_ret.items():
                ret_values[:, :,
                           0] = ret_values[:, :,
                                           0] * self.scale_pm0_values_after_motion

        return ret_values_ret

    def get_part_shape_similarity_loss_no_moving_point(
            self,
            points,
            values,
            disable_hard_negative_sampling=False,
            suffix=''):
        labels = []
        embeddings = []
        occ_primitives = []
        rg = (
            self.primitive_num + 1
        ) if self.use_entire_shape_as_a_part_in_similarity_loss else self.primitive_num
        values = values.clone()
        for idx in range(rg):
            if idx == self.primitive_num:
                occ_primitive, _ = values.max(-1)
            else:
                occ_primitive = values[:, :, idx]
            pred_mask = self.get_discriminator_input_mask(occ_primitive)
            is_background = ((occ_primitive >= 0).sum(axis=1) == 0)
            if is_background.sum() == is_background.size(0):
                continue
            label = torch.ones_like(is_background.long()) * (idx + 1)

            non_background_index = torch.where(~is_background)[0]
            background_index = torch.where(is_background)[0]
            if non_background_index.size(0) != is_background.size(0):
                continue

                sampled_index = non_background_index[torch.randint(
                    len(non_background_index), background_index.shape)]
                occ_primitive[is_background,
                              ...] = occ_primitive[sampled_index, ...]
            labels.append(label)
            occ_primitives.append(occ_primitive)

            pred_occ_max = self.get_indicator_value_for_discriminator(
                occ_primitive.unsqueeze(-1))

            points_pred_label_for_discriminator = pred_occ_max.unsqueeze(-1)

            points_value = torch.cat(
                [points, points_pred_label_for_discriminator], axis=-1)

            ret = self.model(points_value,
                             mask=pred_mask,
                             mode='occupancy_points_encoder')
            embeddings.append(ret['latent_normalized'])

        if len(labels) == 0:
            loss = torch.zeros([1], device=points.device,
                               dtype=points.dtype).mean()
            loss_weighted = loss
        else:
            labels = torch.cat(labels, axis=0).to(self.device)
            embeddings = torch.cat(embeddings, axis=0).to(self.device)

            if self.use_part_shape_similarity_loss_hard_miner and not disable_hard_negative_sampling:
                hard_pairs = self.part_shape_similarity_miner_func(
                    embeddings, labels)  # in your training loop
                loss = self.part_shape_similarity_loss_func(
                    embeddings, labels, hard_pairs)
            else:
                loss = self.part_shape_similarity_loss_func(embeddings, labels)

            loss_weighted = loss * self.part_shape_similarity_loss_weight

        ret = {
            'part_shape_similarity_loss': loss,
            'part_shape_similarity_loss_weighted': loss_weighted
        }
        prefixed_losses = {(name + suffix): value
                           for name, value in ret.items()}
        return prefixed_losses, loss_weighted

    def get_volume_preserving_loss(self, values, moved_values):
        loss = torch.zeros([1], dtype=values.dtype, device=values.device).sum()
        moved_values_sum = torch.relu(moved_values).sum()
        if moved_values_sum > 0:
            moved_values = moved_values.clone().detach().max(-1)[0]
            loss = loss + ((torch.relu(values.max(-1)[0]).sum() -
                            moved_values_sum)**2) / moved_values_sum
        neg_moved_values_sum = torch.relu(-moved_values).sum()
        if neg_moved_values_sum > 0:
            loss_neg = ((torch.relu(-values.max(-1)[0]).sum() -
                         neg_moved_values_sum)**2) / neg_moved_values_sum
            loss = loss * 0.5 + loss_neg * 0.5

        loss_weighted = loss * self.volume_preserving_loss_weight

        ret = {
            'volume_preserving_loss': loss,
            'volume_preserving_loss_weighted': loss_weighted,
        }

        return ret, loss_weighted

    def get_primitive_number_loss(self, values, suffix=''):
        values_front = values.max(-1)[0].reshape(-1) >= 0
        pm0_value = values.reshape(-1, values.size(-1))[:, 0] * values_front
        other_pms_value = values.reshape(
            -1, values.size(-1))[:, 1:].max(-1)[0] * values_front
        loss = (other_pms_value -
                pm0_value).mean().clamp(min=-self.primitive_number_loss_margin)
        # * ((
        #    (pm0_value >= 0).sum() / values_front.sum()) < 0.2)

        pred_occ_max_vis, pred_occ_argmax_vis = torch.sigmoid(values).max(
            axis=-1)
        pred_occ_label = torch.where(
            pred_occ_max_vis >= self.visualize_isosurface_threshold,
            pred_occ_argmax_vis + 1,
            torch.zeros_like(pred_occ_max_vis).long())
        is_pm0_in = np.any(
            np.unique(pred_occ_label.detach().cpu().numpy()) == 1).astype(
                np.float32)

        weighted_loss = loss * self.primitive_number_loss_weight
        ret = {
            'primitive_number_loss': loss,
            'primitive_number_loss_weighted': weighted_loss,
        }

        prefixed_losses = {(name + suffix): value
                           for name, value in ret.items()}
        return prefixed_losses, weighted_loss

    def get_part_shape_canonical_similarity(
            self,
            points,
            values1,
            values2,
            disable_hard_negative_sampling=False,
            suffix=''):
        embeddings = []

        label1 = torch.zeros([points.size(0)], device=points.device).long()
        label2 = torch.ones([points.size(0)], device=points.device).long()
        labels = torch.cat([label1, label2])

        for value in [values1, values2]:
            pred_occ_max = self.get_indicator_value_for_discriminator(value)

            points_pred_label_for_discriminator = pred_occ_max.unsqueeze(-1)

            points_value = torch.cat(
                [points, points_pred_label_for_discriminator], axis=-1)

            pred_mask = self.get_discriminator_input_mask(value.max(-1)[0])
            ret = self.model(points_value,
                             mask=pred_mask,
                             mode='occupancy_points_encoder')
            embeddings.append(ret['latent_normalized'])

        embeddings = torch.cat(embeddings, axis=0).to(self.device)

        loss = self.part_shape_canonical_similarity_loss_func(
            embeddings, labels)
        loss_weighted = loss * self.canonical_similarity_loss_weight

        ret = {
            'part_shape_canonical_similarity_loss': loss,
            'part_shape_canonical_similarity__loss_weighted': loss_weighted
        }
        prefixed_losses = {(name + suffix): value
                           for name, value in ret.items()}
        return prefixed_losses, loss_weighted

    def check_gradient_scale(self,
                             losses,
                             ret,
                             losses_keys=None,
                             ret_keys=None,
                             prefix=''):
        reret = {}
        for retkey, retvalue in ret.items():
            if ret_keys is not None and retkey not in ret_keys:
                continue
            for losskey, lossvalue in losses.items():
                if losses_keys is not None and losskey not in losses_keys:
                    continue
                if 'weighted' in losskey and lossvalue.requires_grad:
                    grad, = torch.autograd.grad(outputs=lossvalue,
                                                inputs=retvalue,
                                                allow_unused=True,
                                                create_graph=True)
                    if grad is None:
                        continue
                    grad_scale = grad.abs().mean()
                    reret['{prefix}grad_scale_{loss}_{output}'.format(
                        prefix=prefix, loss=losskey,
                        output=retkey)] = grad_scale
        assert len(reret) > 0
        return reret

    def check_occupancy_gradient_scale(self,
                                       losses,
                                       ret,
                                       losses_keys=None,
                                       prefix=''):
        reret = {}
        for retkey, retvalue in ret.items():
            for losskey, lossvalue in losses.items():
                if losses_keys is not None and losskey not in losses_keys:
                    continue
                if (
                        'weighted' in losskey and lossvalue.requires_grad
                ) or 'total_G_loss' == losskey or 'total_D_loss' == losskey:
                    grad, = torch.autograd.grad(outputs=lossvalue,
                                                inputs=retvalue,
                                                allow_unused=True,
                                                create_graph=True)
                    if grad is None:
                        continue
                    for idx in range(self.primitive_num):
                        grad_scale = grad.abs()[:, :, idx].mean()
                        reret[
                            '{prefix}grad_scale_{loss}_{output}_{idx}'.format(
                                prefix=prefix,
                                loss=losskey,
                                output=retkey,
                                idx=idx)] = grad_scale
        assert len(reret) > 0
        return reret

    def check_gradient_scale(self, losses, ret, losses_keys=None, prefix=''):
        reret = {}
        done = []
        try:
            for retkey, retvalue in ret.items():
                for losskey, lossvalue in losses.items():
                    if losses_keys is not None and losskey not in losses_keys:
                        continue
                    if (
                            'weighted' in losskey and lossvalue.requires_grad
                    ) or 'total_G_loss' == losskey or 'total_D_loss' == losskey:
                        if (retkey, losskey) in done:
                            continue
                        grad, = torch.autograd.grad(outputs=lossvalue,
                                                    inputs=retvalue,
                                                    allow_unused=True,
                                                    create_graph=True)
                        done.append((retkey, losskey))
                        if grad is None:
                            continue
                        grad_scale = grad.abs().mean()
                        reret['{prefix}grad_scale_{loss}_{output}'.format(
                            prefix=prefix, loss=losskey,
                            output=retkey)] = grad_scale
        except BaseException as e:
            print('error', retkey, losskey)
            print(done)
            print(retvalue.requires_grad, lossvalue.requires_grad)
            raise e
        assert len(reret) > 0
        return reret

    def get_gradient_penalty(self, discriminator_ret, discriminator_real_ret):
        real = discriminator_real_ret['points_value']
        batch_size = real.size(0)

        if self.gan_type == 'wgangp' and self.gradient_penalty_type == 'wgangp':
            fake = discriminator_ret['points_value']
            if self.use_voxel_discriminator:
                alpha = torch.rand(batch_size, 1, 1, 1, 1,
                                   requires_grad=True).to(self.device)
            else:
                alpha = torch.rand(batch_size, 1, 1,
                                   requires_grad=True).to(self.device)
            # randomly mix real and fake data
            interpolates = real + alpha * (fake - real)
            # compute output of D for interpolated input

            disc_interpolates = self.model(interpolates,
                                           mode='discriminator')['D']
            # compute gradients w.r.t the interpolated outputs

            gradients = torch.autograd.grad(
                outputs=disc_interpolates,
                inputs=interpolates,
                grad_outputs=torch.ones(disc_interpolates.size()).to(
                    self.device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0].contiguous().view(batch_size, -1)

            gradient_penalty = (((gradients.norm(2, dim=1) - self.gp_gamma) /
                                 self.gp_gamma)**2).mean() * self.gp_lambda

        elif self.gradient_penalty_type == 'real_input':
            grad_real, = torch.autograd.grad(
                outputs=discriminator_real_ret['D'].sum(),
                inputs=real,
                allow_unused=True,
                create_graph=True)
            gradient_penalty = grad_real.pow(2).reshape(
                grad_real.shape[0],
                -1).sum(1).mean() * self.apply_gradient_penalty_every
        else:
            raise NotImplementedError

        gradient_penalty_weighted = gradient_penalty * self.gradient_pelnalty_weight * self.D_loss_weight
        ret = {
            'gradient_penalty': gradient_penalty,
            'gradient_penalty_weighted': gradient_penalty_weighted
        }
        return ret, gradient_penalty_weighted

    def get_pretrained_feature_canonical_loss(self, values, canonical_values,
                                              points):

        inputs = torch.cat(
            [points, torch.sigmoid(values.max(-1, keepdims=True)[0])], dim=-1)
        canonical_inputs = torch.cat([
            points,
            torch.sigmoid(canonical_values.max(-1, keepdims=True)[0])
        ],
                                     dim=-1)

        pretrained_model = self.pretrained_models['occae']
        feat = pretrained_model(inputs)[
            self.which_pretrained_feature_canonical_loss_feat]
        canonical_feat = pretrained_model(canonical_inputs)[
            self.which_pretrained_feature_canonical_loss_feat].detach()

        loss = F.mse_loss(feat, canonical_feat)
        loss_weighted = loss * self.pretrained_feature_canonical_loss_weight

        ret = dict(pretrained_feature_canonical_loss_weighted=loss_weighted,
                   pretrained_feature_canonical_loss=loss)
        return ret, loss_weighted

    def get_soft_volume_preserving_loss(self, values, moved_values):
        if self.use_occ_recon_loss_for_occ_and_tsdf_for_param:
            values = torch.sigmoid(
                values *
                self.occ_recon_loss_for_occ_and_tsdf_for_param_occinize_scale)
            moved_values = torch.sigmoid(
                moved_values *
                self.occ_recon_loss_for_occ_and_tsdf_for_param_occinize_scale)

        if self.soft_volume_preserving_loss_type == 'default':
            loss = F.mse_loss(
                torch.relu(values.max(-1)[0]).mean(),
                torch.relu(moved_values.max(-1)[0]).mean())
        elif self.soft_volume_preserving_loss_type == 'primitive_wise_mean':
            loss = F.mse_loss(
                values.mean(0).mean(1),
                moved_values.mean(0).mean(1))
        loss_weighted = loss * self.soft_volume_preserving_loss_weight

        ret = {
            'soft_volume_preserving_loss': loss,
            'soft_volume_preserving_loss_weighted': loss_weighted,
        }

        return ret, loss_weighted

    def get_anchor_position_near_gt_shape_loss(self,
                                               points,
                                               values,
                                               rotation_anchor_point,
                                               pred_values=None,
                                               pm_num=None):
        total_rot_pm_dist = torch.zeros([1],
                                        device=points.device,
                                        dtype=points.dtype).mean()

        if pm_num is None:
            pm_num = self.model.rotation_primitive_num
        for idx in range(pm_num):
            mask = values > 0
            if self.use_anchor_position_near_gt_shape_loss_near_primitive:
                pred_primitive_flag = (pred_values.argmax(-1) == (idx + 1))
                all_zero_flag = (
                    pred_primitive_flag.sum(-1).unsqueeze(-1) == 0)
                pred_mask = mask & pred_primitive_flag
                mask = (mask & all_zero_flag) | (pred_mask & (~all_zero_flag))
            elif self.use_anchor_position_near_gt_shape_loss_union_to_static_primitive:
                pred_primitive_flag = (torch.sigmoid(pred_values[
                    ..., idx + 1]) >= self.visualize_isosurface_threshold) & (
                        torch.sigmoid(pred_values[..., 0]) >=
                        self.visualize_isosurface_threshold)
                all_zero_flag = (
                    pred_primitive_flag.sum(-1).unsqueeze(-1) == 0)
                pred_mask = mask & pred_primitive_flag
                mask = (mask & all_zero_flag) | (pred_mask & (~all_zero_flag))
            source_points = rotation_anchor_point[:, idx, :].unsqueeze(-2)

            dists = (
                (source_points - points)**2).sum(-1) * mask + (~mask) * 100
            dist = dists.min(-1)[0]
            dist_clipped = torch.where(dist > 50, torch.zeros_like(dist),
                                       dist).sum()
            num = (dist < 50).sum()
            if num < 1:
                dist_rot_pm = dist_clipped * 0.
            else:
                dist_rot_pm = dist_clipped / num

            total_rot_pm_dist = total_rot_pm_dist + dist_rot_pm / self.model.rotation_primitive_num

        anchor_position_near_gt_shape_loss = total_rot_pm_dist
        anchor_position_near_gt_shape_loss_weighted = anchor_position_near_gt_shape_loss * self.anchor_position_near_gt_shape_loss_weight

        ret = {
            'anchor_position_near_gt_shape_loss':
            anchor_position_near_gt_shape_loss,
            'anchor_position_near_gt_shape_loss_weighted':
            anchor_position_near_gt_shape_loss_weighted,
            'total_rot_pm_dist': total_rot_pm_dist
        }

        return ret, anchor_position_near_gt_shape_loss_weighted
        """
        """

    def sample_rotation_anchor_point_by_occupancy(self, points, values):
        static_primitive_value = values[..., 0]
        rotation_anchor_points = []
        for idx in range(1, self.model.rotation_primitive_num + 1):
            rotation_primitive_value = values[..., idx]
            prob = F.softplus(static_primitive_value) * F.softplus(
                rotation_primitive_value)
            if self.sample_rotation_anchor_point_by_occupancy_type == 'gumbel':
                sampled_weight = gumbel_softmax.gumbel_softmax(
                    prob, dim=-1).unsqueeze(-1)
            elif self.sample_rotation_anchor_point_by_occupancy_type == 'softmax':
                sampled_weight = F.softmax(prob, dim=-1).unsqueeze(-1)
            else:
                raise NotImplementedError
            rotation_anchor_point = (sampled_weight * points).sum(dim=1,
                                                                  keepdim=True)
            rotation_anchor_points.append(rotation_anchor_point)
        return torch.cat(rotation_anchor_points, dim=1)

    def get_rotation_anchor_point_similarity_loss(self, rotation_anchor_point):
        scale = 0.1
        var = scale**2
        if self.rotation_anchor_point_similarity_loss_type == 'gaussian_logprob':
            log_scale = np.log(scale)
            loss = -((rotation_anchor_point.unsqueeze(1) -
                      rotation_anchor_point.unsqueeze(2))**2).sum(-1) / (
                          2 * var) - log_scale - np.log(np.sqrt(2 * np.pi))
            loss = loss.triu().sum() / ((rotation_anchor_point.shape[1]**2 -
                                         rotation_anchor_point.shape[1]) / 2)
        elif self.rotation_anchor_point_similarity_loss_type == 'gaussian_prob':
            loss = torch.exp(
                -((rotation_anchor_point.unsqueeze(1) -
                   rotation_anchor_point.unsqueeze(2))**2).sum(-1) / (2 * var))
            loss = loss.triu().sum() / ((rotation_anchor_point.shape[1]**2 -
                                         rotation_anchor_point.shape[1]) / 2)
        else:
            raise NotImplementedError
        loss_weighted = loss * self.rotation_anchor_point_similarity_loss_weight

        ret = dict(
            rotation_anchor_point_similarity_loss=loss,
            rotation_anchor_point_similarity_loss_weighted=loss_weighted)
        return ret, loss_weighted

    def get_rotation_anchor_point_occupancy_relation_loss(
            self, rotation_anchor_point_value):
        loss = 0.
        for idx in range(1, self.model.rotation_primitive_num + 1):
            static_primitive_value = rotation_anchor_point_value[:, idx - 1, 0]
            self_primitive_value = rotation_anchor_point_value[:, idx - 1, idx]
            occ_loss_static = F.binary_cross_entropy_with_logits(
                static_primitive_value,
                torch.ones_like(static_primitive_value),
                reduction='none').sum(-1).mean()
            occ_loss_self = F.binary_cross_entropy_with_logits(
                self_primitive_value,
                torch.ones_like(self_primitive_value),
                reduction='none').sum(-1).mean()
            loss = loss + occ_loss_self + occ_loss_static
        loss_weighted = loss * self.rotation_anchor_point_occupancy_relation_loss_weight
        ret = dict(
            rotation_anchor_point_occupancy_relation_loss=loss,
            rotation_anchor_point_occupancy_relation_loss_weighted=loss_weighted
        )
        return ret, loss_weighted

    def transform_by_canonical_location(self,
                                        latent,
                                        points,
                                        values_ret,
                                        canonical_location,
                                        generator_kwargs={}):
        assert not self.is_move_points

        moved_points_moving_points = points.clone()
        ret_values_ret = {}
        for key, values in values_ret.items():
            ret_values = values.clone()
            ret_values_ret[key] = ret_values
        for idx, midx in enumerate(range(1, self.primitive_num)):
            if self.model.motion_decoding_type == 'default':
                raise NotImplementedError("need reconsider")
            elif self.model.motion_decoding_type in [
                    'one_joint_type_per_primitive',
                    'one_joint_type_per_primitive_rot_pm_num_specified'
            ]:
                primitive_wise_transformation = canonical_location[:, midx,
                                                                   ...]
                tmp = points - primitive_wise_transformation.unsqueeze(-2)
            else:
                raise NotImplementedError

            ret = self.model(latent,
                             tmp,
                             return_param=False,
                             direct_input_to_decoder=True,
                             generator_kwargs=generator_kwargs)
            for key, ret_values in ret_values_ret.items():
                ret_values[:, :, midx] = ret[key][:, :, midx]
        if self.is_input_pm0_to_motion_net_too:
            ret = self.model(latent,
                             points,
                             return_param=False,
                             direct_input_to_decoder=True,
                             generator_kwargs=generator_kwargs)
            for key, ret_values in ret_values_ret.items():
                ret_values[:, :, 0] = ret[key][:, :, 0]
        if self.is_scale_pm0_values_after_motion:
            for key, ret_values in ret_values_ret.items():
                ret_values[:, :,
                           0] = ret_values[:, :,
                                           0] * self.scale_pm0_values_after_motion

        return ret_values_ret

    def get_bone_occupancy_loss(self, latent, transformation):
        canonical_location = transformation['canonical_location']
        batch_size = canonical_location.size(0)
        sampled_point = torch.rand(batch_size,
                                   self.bone_occupancy_loss_sampling_num,
                                   1,
                                   dtype=canonical_location.dtype,
                                   device=canonical_location.device)
        loss = 0.
        pred_values_list = []
        for idx in range(self.model.rotation_primitive_num):
            rotation_canonical_location = transformation[
                'canonical_location'][:, 1 + idx, :]
            rotation_anchor_point = transformation[
                'rotation_anchor_point'][:, idx, :]
            rotation_direction = transformation['rotation_direction'][:,
                                                                      idx, :]
            c_a = (rotation_canonical_location - rotation_anchor_point)
            proj = ((c_a * rotation_direction).sum(dim=-1, keepdim=True) *
                    rotation_direction) + rotation_anchor_point

            points = proj.unsqueeze(
                -2
            ) + (rotation_canonical_location - proj).unsqueeze(
                -2) * sampled_point - rotation_canonical_location.unsqueeze(-2)
            pred_values = self.model(latent,
                                     points,
                                     return_param=False,
                                     direct_input_to_decoder=True)['occupancy']
            pred_values_list.append(pred_values[:, :, idx + 1])

            occ_loss = F.binary_cross_entropy_with_logits(
                pred_values[:, :, idx + 1],
                torch.ones_like(pred_values[:, :, 0]),
                reduction='none').sum(-1).mean()
            loss = loss + occ_loss
        loss_weighted = loss * self.bone_occupancy_loss_weight
        ret = dict(bone_occupancy_loss=loss,
                   bone_occupancy_loss_weighted=loss_weighted)

        ret_values = []
        for idx in range(self.primitive_num):
            if idx == 0:
                ret_values.append(torch.ones_like(pred_values[:, :, idx]))
            elif idx < self.model.rotation_primitive_num + 1:
                ret_values.append(pred_values_list[idx - 1])
            else:
                ret_values.append(torch.ones_like(pred_values[:, :, idx]))
        pred_values = torch.stack(ret_values, dim=-1)

        return ret, loss_weighted, pred_values

    def get_canonical_location_near_gt_loss(self, points, values,
                                            transformation):
        loss = torch.zeros([1], device=points.device,
                           dtype=points.dtype).mean()
        for idx, midx in enumerate(range(1, self.primitive_num)):
            if self.model.motion_decoding_type == 'default':
                raise NotImplementedError("need reconsider")

            elif self.model.motion_decoding_type in [
                    'one_joint_type_per_primitive',
                    'one_joint_type_per_primitive_rot_pm_num_specified'
            ]:
                if idx < self.model.rotation_primitive_num:
                    primitive_wise_transformation = {
                        key:
                        value[:, midx if key.startswith('canonical') else idx,
                              ...]
                        for key, value in transformation.items()
                        if key.startswith('rotation')
                        or key.startswith('canonical')
                    }
                    rotation_anchor_point = primitive_wise_transformation[
                        'rotation_anchor_point'] * self.rotation_anchor_point_scale
                    rotation_direction = primitive_wise_transformation[
                        'rotation_direction']
                    # B, 1
                    rotation_amount = primitive_wise_transformation[
                        'rotation_amount']
                    if self.is_constant_motion_range:
                        rotation_scale = primitive_wise_transformation[
                            'rotation_scale'] * 0. + np.pi
                    else:
                        rotation_scale = primitive_wise_transformation[
                            'rotation_scale']
                    rotation_angle = rotation_amount * rotation_scale

                    canonical_location = primitive_wise_transformation[
                        'canonical_location']

                    tmp = canonical_location.unsqueeze(
                        -2) - rotation_anchor_point.unsqueeze(-2)
                    quat = geometry.get_quaternion(
                        primitive_wise_transformation['rotation_direction'],
                        rotation_angle)
                    tmp = geometry.apply_3d_rotation(
                        tmp, quat) + rotation_anchor_point.unsqueeze(-2)
                else:
                    primitive_wise_transformation = {
                        key:
                        value[:, idx -
                              (-1 if key.startswith('canonical') else self.
                               model.rotation_primitive_num), ...]
                        for key, value in transformation.items()
                        if key.startswith('translation')
                        or key.startswith('canonical')
                    }
                    translation_direction = primitive_wise_transformation[
                        'translation_direction']
                    # B, 1
                    translation_amount = primitive_wise_transformation[
                        'translation_amount']
                    if self.is_constant_motion_range:
                        translation_scale = primitive_wise_transformation[
                            'translation_scale'] * 0 + 1.
                    else:
                        translation_scale = primitive_wise_transformation[
                            'translation_scale']
                    translation = translation_direction * translation_amount * translation_scale
                    tmp = (
                        primitive_wise_transformation['canonical_location'] +
                        translation).unsqueeze(-2)

            source_points = tmp
            mask = values > 0
            dists = (
                (source_points - points)**2).sum(-1) * mask + (~mask) * 100
            dist = dists.min(-1)[0]
            dist_clipped = torch.where(dist > 50, torch.zeros_like(dist),
                                       dist).sum()
            num = (dist < 50).sum()
            if num < 1:
                dist_rot_pm = dist_clipped * 0.
            else:
                dist_rot_pm = dist_clipped / num

            loss = loss + dist_rot_pm / (self.model.primitive_num - 1)

        loss_weighted = loss * self.canonical_location_near_gt_loss_weight
        ret = dict(canonical_location_near_gt_loss=loss,
                   canonical_location_near_gt_loss_weighted=loss_weighted)
        return ret, loss_weighted

    def get_tsdf_reconstruction_loss(self, pred_values, values):
        pred_values = pred_values.max(-1)[0]
        loss = F.l1_loss(pred_values, values)
        loss_weighted = loss * self.tsdf_loss_weight
        ret = dict(tsdf_reconstruction_loss=loss,
                   tsdf_reconstruction_loss_weighted=loss_weighted)
        return ret, loss_weighted

    def get_surface_quasi_sdf_loss(self,
                                   pred_occ,
                                   level_set,
                                   loss_weight=None,
                                   prefix=''):
        pred_occ_max_logit = self.get_indicator_value(pred_occ)

        losses = {}
        occ_loss = (
            (torch.sigmoid(pred_occ_max_logit) -
             torch.ones_like(pred_occ_max_logit) * level_set)**2).mean()

        if loss_weight is not None:
            occ_loss_weighted = occ_loss * loss_weight
        else:
            occ_loss_weighted = occ_loss * self.surface_quasi_sdf_loss_weight
        losses['surface_quasi_sdf_loss'] = occ_loss
        losses['surface_quasi_sdf_loss_weighted'] = occ_loss_weighted
        if prefix:
            losses = {
                prefix + '_' + key: value
                for key, value in losses.items()
            }

        return losses, occ_loss_weighted

    def get_explicit_pointcloud_loss_for_motion_with_surface_points(
            self, points, target_points, params):
        transformed_points = self.get_explicitly_transformed_points(
            points, params)
        loss = chamfer_loss.chamfer_loss(
            transformed_points.view(points.size(0), -1, points.size(-1)),
            target_points)
        loss_weighted = loss * self.explicit_pointcloud_loss_for_motion_with_surface_points_weight

        return dict(
            explicit_pointcloud_loss_for_motion_with_surface_points=loss,
            explicit_pointcloud_loss_for_motion_with_surface_points_weighted=
            loss_weighted), loss_weighted

    def get_explicitly_transformed_points(self,
                                          points,
                                          transformation,
                                          detach_transformation=False):
        assert not self.is_move_points

        if detach_transformation:
            transformation = {
                key: param.clone().detach()
                for key, param in transformation.items()
            }

        moved_points_moving_points = points.clone()

        transformed_points = [
            moved_points_moving_points[:, :, 0, :].unsqueeze(-2)
        ]
        for idx, midx in enumerate(range(1, self.primitive_num)):
            tmp = moved_points_moving_points[:, :, midx, :]
            if self.model.motion_decoding_type == 'default':
                raise NotImplementedError("need reconsider")
            elif self.model.motion_decoding_type in [
                    'one_joint_type_per_primitive',
                    'one_joint_type_per_primitive_rot_pm_num_specified'
            ]:
                if idx < self.model.rotation_primitive_num:
                    primitive_wise_transformation = {
                        key:
                        value[:, midx if key.startswith('canonical') else idx,
                              ...]
                        for key, value in transformation.items()
                        if key.startswith('rotation')
                        or key.startswith('canonical')
                    }
                    tmp = self.apply_rotation(tmp,
                                              primitive_wise_transformation,
                                              is_explicit=True)
                else:
                    primitive_wise_transformation = {
                        key:
                        value[:, idx -
                              (-1 if key.startswith('canonical') else self.
                               model.rotation_primitive_num), ...]
                        for key, value in transformation.items()
                        if key.startswith('translation')
                        or key.startswith('canonical')
                    }
                    tmp = self.apply_translation(tmp,
                                                 primitive_wise_transformation,
                                                 is_explicit=True)
            transformed_points.append(tmp.unsqueeze(-2))

        transformed_points = torch.cat(transformed_points, dim=-2)
        return transformed_points

    def preprocess_transformation(self, ret, points, points_expand,
                                  pred_values, pred_params, pred_latent,
                                  pred_generator_kwargs):
        if self.model.use_canonical_location_from_generator:
            pred_params['canonical_location'] = ret['canonical_location']
            if 'rotation_location_offset' in ret:
                canonical_location = pred_params['canonical_location']
                rotation_canonical_location = canonical_location[:,
                                                                 1:self.model.
                                                                 rotation_primitive_num
                                                                 + 1, :]
                canonical_location[:, 1:self.model.rotation_primitive_num +
                                   1, :] = rotation_canonical_location + ret[
                                       'rotation_location_offset']
                pred_params['canonical_location'] = canonical_location
            if 'translation_location_offset' in ret:
                canonical_location = pred_params['canonical_location']
                translation_canonical_location = canonical_location[:,
                                                                    self.model.
                                                                    rotation_primitive_num
                                                                    + 1:, :]
                canonical_location[:, self.model.rotation_primitive_num +
                                   1:, :] = translation_canonical_location + ret[
                                       'translation_location_offset']
                pred_params['canonical_location'] = canonical_location

            if self.is_apply_tanh_to_merged_canonical_location:
                pred_params['canonical_location'] = torch.tanh(
                    pred_params['canonical_location']) * 0.5

            if self.disable_detach_canonical_location_as_pred_values_in_generator:
                canonical_location = ret['canonical_location']
            else:
                canonical_location = ret['canonical_location'].clone().detach()
            values_ret = self.transform_by_canonical_location(
                pred_latent,
                points_expand,
                dict(occupancy=pred_values,
                     canonical_occupancy=ret['canonical_occupancy']),
                canonical_location,
                generator_kwargs=pred_generator_kwargs)

            pred_values = values_ret['occupancy']
            ret['canonical_occupancy'] = values_ret['canonical_occupancy']

        if self.is_canonical_location_as_anchor_point and 'canonical_location' in pred_params:
            pred_params['rotation_anchor_point'] = pred_params[
                'canonical_location'][:, 1:self.model.rotation_primitive_num +
                                      1, :]
            pred_params['translation_anchor_point'] = pred_params[
                'canonical_location'][:, self.model.rotation_primitive_num +
                                      1:, :]
        if self.model.use_canonical_direction_from_generator:
            if self.model.is_decode_canonical_direction_as_rot_matrix:
                pred_params['canonical_rotation_matrix'] = ret[
                    'canonical_rotation_matrix']
                if self.is_correct_continuous_rotation:
                    pred_params['rotation_matrix'] = torch.matmul(
                        ret['canonical_rotation_matrix']
                        [:, 1:self.model.rotation_primitive_num + 1, :, :],
                        pred_params['rotation_matrix'])
                else:
                    pred_params['rotation_matrix'] = torch.matmul(
                        pred_params['rotation_matrix'],
                        ret['canonical_rotation_matrix']
                        [:, 1:self.model.rotation_primitive_num + 1, :, :])
            pred_params['rotation_direction'] = ret[
                'canonical_direction'][:, 1:self.model.rotation_primitive_num +
                                       1, :]
            pred_params['translation_direction'] = ret[
                'canonical_direction'][:, self.model.rotation_primitive_num +
                                       1:, :]

        if self.is_sample_rotation_anchor_point_by_occupancy:
            pred_params[
                'rotation_anchor_point'] = self.sample_rotation_anchor_point_by_occupancy(
                    points, pred_values)

    def get_motion_amount_inverse_std_loss(self, transform):
        if self.motion_amount_inverse_std_loss_aggregation_type == 'mean':
            rotation_std = torch.rsqrt(
                torch.var(transform['rotation_amount'], dim=0) + EPS).mean()
            translation_std = torch.rsqrt(
                torch.var(transform['translation_amount'], dim=0) +
                EPS).mean()
        elif self.motion_amount_inverse_std_loss_aggregation_type == 'min':
            rotation_std = torch.rsqrt(
                torch.var(transform['rotation_amount'], dim=0) + EPS).min()
            translation_std = torch.rsqrt(
                torch.var(transform['translation_amount'], dim=0) + EPS).min()
        elif self.motion_amount_inverse_std_loss_aggregation_type == 'max':
            rotation_std = torch.rsqrt(
                torch.var(transform['rotation_amount'], dim=0) + EPS).max()
            translation_std = torch.rsqrt(
                torch.var(transform['translation_amount'], dim=0) + EPS).max()
        elif self.motion_amount_inverse_std_loss_aggregation_type in ['exact']:
            rotation_std = torch.rsqrt(
                torch.var(transform['rotation_amount'], dim=0) + EPS)
            translation_std = torch.rsqrt(
                torch.var(transform['translation_amount'], dim=0) + EPS)
        else:
            raise NotImplementedError

        if self.motion_amount_inverse_std_loss_aggregation_type in [
                'min', 'max', 'mean'
        ]:
            loss = torch.relu(
                rotation_std -
                self.motion_amount_inverse_std_loss_rotation_threshold
            ) + torch.relu(
                translation_std -
                self.motion_amount_inverse_std_loss_translation_threshold)
        elif self.motion_amount_inverse_std_loss_aggregation_type in ['exact']:
            loss = (
                (rotation_std -
                 self.motion_amount_inverse_std_loss_rotation_threshold)**
                2).mean() + (
                    (translation_std -
                     self.motion_amount_inverse_std_loss_translation_threshold)
                    **2).mean()
        else:
            raise NotImplementedError
        loss = loss * 0.5
        loss_weighted = loss * self.motion_amount_inverse_std_loss_weight
        ret = dict(motion_amount_inverse_std_loss=loss,
                   motion_amount_inverse_std_loss_weighted=loss_weighted)
        return ret, loss_weighted

    def get_location_offset_regulariztion_loss(self, offset, prefix=''):
        if self.location_offset_regulariztion_loss_type == 'gaussian':
            std = self.location_offset_regulariztion_loss_gausian_scale
            var = std**2
            A = 1. / (std * (2 * np.pi)**(0.5))
            loss = 1 - A * torch.exp(-((offset)**2).sum(-1) / (2 * var))
            loss = loss.mean()
        elif self.location_offset_regulariztion_loss_type == 'mse':
            loss = (offset**2).sum(-1).mean()
        loss_weighted = loss * self.location_offset_regulariztion_loss_weight

        ret = dict(location_offset_regulariztion_loss=loss,
                   location_offset_regulariztion_loss_weighted=loss_weighted)
        return ret, loss_weighted

    def get_minimize_raw_canonical_location_to_offset_loss(
            self, canonical_location, raw_canonical_location):
        if self.minimize_raw_canonical_location_to_offset_loss_detach_canonical_location_grad:
            loss = F.mse_loss(canonical_location.clone().detach(),
                              raw_canonical_location)
        else:
            loss = F.mse_loss(canonical_location, raw_canonical_location)
        loss_weighted = loss * self.minimize_raw_canonical_location_to_offset_loss_weight
        ret = dict(minimize_raw_canonical_location_to_offset_loss_weighted=
                   loss_weighted,
                   minimize_raw_canonical_location_to_offset_loss=loss)
        return ret, loss_weighted

    def get_minimize_raw_canonical_direction_to_offset_loss(
            self, canonical_direction, raw_canonical_direction):
        if self.minimize_raw_canonical_direction_to_offset_loss_detach_canonical_direction_grad:
            loss = F.mse_loss(canonical_direction.clone().detach(),
                              raw_canonical_direction)
        else:
            loss = F.mse_loss(canonical_direction, raw_canonical_direction)
        loss_weighted = loss * self.minimize_raw_canonical_direction_to_offset_loss_weight
        ret = dict(minimize_raw_canonical_direction_to_offset_loss_weighted=
                   loss_weighted,
                   minimize_raw_canonical_direction_to_offset_loss=loss)
        return ret, loss_weighted

    def get_screw_motion_spectrum_entropy_loss(self, motion_spectrum):
        p = torch.sigmoid(motion_spectrum)
        loss = ((-p * p.log()) + (-(1 - p) * (1 - p).log())).mean()
        loss_weighted = loss * self.screw_motion_spectrum_entropy_loss_weight
        ret = dict(screw_motion_spectrum_entropy_loss_weighted=loss_weighted,
                   screw_motion_spectrum_entropy_loss=loss)
        return ret, loss_weighted

    def get_anchor_position_chain_assumption_loss(self, points, pred_values,
                                                  rotation_anchor_point):
        total_rot_pm_dist = torch.zeros([1],
                                        device=points.device,
                                        dtype=points.dtype).mean()

        th = self.anchor_position_chain_assumption_loss_threshold
        if th is None:
            th = self.visualize_isosurface_threshold
        pm_num = self.model.rotation_primitive_num
        for idx in range(pm_num):

            mask = (pred_values.argmax(-1) == (idx + 1)) & (torch.sigmoid(
                pred_values[..., idx + 1]) >= th)

            source_points = rotation_anchor_point[:, idx, :].unsqueeze(-2)

            dists = (
                (source_points - points)**2).sum(-1) * mask + (~mask) * 100
            dist = dists.min(-1)[0]
            dist_clipped = torch.where(dist > 50, torch.zeros_like(dist),
                                       dist).sum()
            num = (dist < 50).sum()
            if num < 1:
                dist_rot_pm = dist_clipped * 0.
            else:
                dist_rot_pm = dist_clipped / num

            total_rot_pm_dist = total_rot_pm_dist + dist_rot_pm / self.model.rotation_primitive_num / 2

            if self.anchor_position_chain_assumption_loss_only_self:
                continue
            mask = (torch.argmax(pred_values, dim=-1) !=
                    (idx + 1)) & (torch.any(torch.sigmoid(pred_values) >= th,
                                            dim=-1))

            source_points = rotation_anchor_point[:, idx, :].unsqueeze(-2)

            dists = (
                (source_points - points)**2).sum(-1) * mask + (~mask) * 100
            dist = dists.min(-1)[0]
            dist_clipped = torch.where(dist > 50, torch.zeros_like(dist),
                                       dist).sum()
            num = (dist < 50).sum()
            if num < 1:
                dist_rot_pm = dist_clipped * 0.
            else:
                dist_rot_pm = dist_clipped / num

            total_rot_pm_dist = total_rot_pm_dist + dist_rot_pm / self.model.rotation_primitive_num / 2

        anchor_position_chain_assumption_loss = total_rot_pm_dist
        anchor_position_chain_assumption_loss_weighted = anchor_position_chain_assumption_loss * self.anchor_position_chain_assumption_loss_weight

        ret = {
            'anchor_position_chain_assumption_loss':
            anchor_position_chain_assumption_loss,
            'anchor_position_chain_assumption_loss_weighted':
            anchor_position_chain_assumption_loss_weighted,
            'anchor_position_chain_assumption_total_rot_pm_dist':
            total_rot_pm_dist
        }

        return ret, anchor_position_chain_assumption_loss_weighted
