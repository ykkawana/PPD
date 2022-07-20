import torch
from collections import defaultdict, OrderedDict
import torch.nn.functional as F
from PIL import Image
import numpy as np

import plotly.graph_objects as go
import seaborn as sns
import io
from loss import chamfer_loss
from utils import visualizer_util
from utils import eval_utils
from utils import geometry
import random
from pytorch_metric_learning import losses as metric_losses
from pytorch_metric_learning import miners as metric_miners
from concurrent.futures import process
from scipy import optimize

EPS = 1e-7


class Trainer:
    def __init__(
            self,
            model,
            optimizers,
            device,
            shape_recon_chamfer_loss_weight=1.,
            occupancy_loss_weight=1.,
            D_real_loss_weight=1.,
            D_fake_loss_weight=1.,
            D_loss_weight=1.,
            G_loss_weight=1.,
            generator_loss_weight=1.,
            gradient_pelnalty_weight=1.,
            gp_gamma=1.,
            gp_lambda=10.,
            train_explicit=False,
            train_implicit=False,
            dataset_boxsize=50,
            dataset_worldsize=255,
            use_canonical_self_supervised_slide_loss=False,
            self_supervised_slide_loss_weight=1.,
            visualizer_marker_size=3,
            visualize_isosurface_threshold=0.5,
            evaluate_isosurface_threshold=0.5,
            use_overlap_regularizer=False,
            overlap_threshold=1.,
            overlap_regularizer_loss_weight=1.,
            implicit_discriminator_activation_type='sigmoid',
            occupancy_reduction_type='max',
            points_label_for_discriminator_scale=1.,
            use_l1_occupancy_regularizer=False,
            l1_occupancy_regularizer_loss_weight=1.,
            use_canonical_recon_loss=False,
            use_canonical_occ_loss=False,
            is_movenet_classification_head=False,
            multilabel_classification_criterion_type='crossentropy',
            is_movenet_decode_sdf=False,
            overlap_regularizer_type='default',
            disable_moved_overlap_regularizer_loss=False,
            occupancy_reduction_type_for_discriminator='max',
            sdf_loss_weight=1.,
            is_clamp_sdf_value=False,
            clamp_sdf_value=0.1,
            use_discriminator_input_mask=False,
            is_move_points=False,
            is_temporal_pad_slide_dim=False,
            discriminator_input_mask_type='default',
            use_acgan=False,
            gan_type='wgangp',
            learn_D_slide_loss=False,
            D_slide_loss_weight=1.,
            G_slide_loss_weight=1.,
            learn_D_canonical_loss=False,
            D_canonical_loss_weight=1.,
            G_canonical_loss_weight=1.,
            learn_moving_self_slide_reg=False,
            is_detach_param_in_generator=False,
            is_move_points_slide_bug_fixed=False,
            use_wgangp_latent_code_interpolation=False,
            part_iou_vs_slide_envelope_loss_minimum_slide_loss_threshold=0.0135,
            use_part_shape_similarity_loss=False,
            part_shape_similarity_loss_type='TripletMarginLoss',
            triplet_margin_loss_margin=0.05,
            use_part_shape_similarity_loss_hard_miner=False,
            part_shape_similarity_miner_type='MultiSimilarityMiner',
            multi_similarity_miner_epsilon=0.1,
            part_shape_similarity_loss_weight=1.,
            disable_gan_training=False,
            disable_induction_by_moving=False,
            use_entire_shape_as_a_part_in_similarity_loss=False,
            use_entropy_reduction_loss=False,
            entropy_reduction_loss_type='aabb',
            entropy_reduction_loss_primitive_worker_num=5,
            each_entropy_reduction_loss_margin=0.,
            whole_entropy_reduction_loss_margin=0.,
            entropy_reduction_loss_weight=1.,
            use_gt_values_as_whole_shape_in_entropy_reduction_loss=False,
            disable_part_seg_loss_iou_for_eval=False,
            use_unsupervised_part_iou_for_eval=False,
            ignore_background_in_part_iou_loss=False,
            occupancy_reduction_loss_type='occnet',
            non_top_primitives_overlap_loss_margin=0.01,
            use_imnet_style_occupancy_loss_with_logit=False,
            use_tsdf_in_occupancy_loss=False,
            loss_optimizer_pairs=OrderedDict({
                'total_D_loss': {
                    'discriminator': {
                        'every': 1
                    }
                },
                'total_G_loss': {
                    'generator': {
                        'every': 1
                    }
                }
            }),
            is_occupancy_reduction_softmax_decay=False,
            occupancy_reduction_softmax_mix_ratio=0.5,
            occupancy_reduction_softmax_mix_ratio_max=0.5,
            occupancy_reduction_softmax_grow_ratio=0.000095,
            occupancy_reduction_softmax_decay_max_steps=100000,
            occupancy_reduction_decay_type='exponential',
            use_mse_in_occupancy_loss=False,
            is_occupancy_reduction_logsumexp_decay=False,
            occupancy_reduction_logsumexp_mix_ratio=1.0,
            use_bce_without_logits=False,
            use_overlap_regularizer_loss=False):
        self.pretrained_models = {}
        self.model = model
        self.optimizers = optimizers
        self.primitive_num = self.model.primitive_num
        self.device = device
        self.shape_recon_chamfer_loss_weight = shape_recon_chamfer_loss_weight
        self.occupancy_loss_weight = occupancy_loss_weight
        self.D_real_loss_weight = D_real_loss_weight
        self.D_fake_loss_weight = D_fake_loss_weight
        self.G_loss_weight = G_loss_weight
        self.D_loss_weight = D_loss_weight
        self.gp_gamma = gp_gamma
        self.gp_lambda = gp_lambda
        self.gradient_pelnalty_weight = gradient_pelnalty_weight
        self.generator_loss_weight = generator_loss_weight
        self.train_explicit = train_explicit
        self.train_implicit = train_implicit
        self.dataset_boxsize = dataset_boxsize
        self.dataset_worldsize = dataset_worldsize
        self.visualize_isosurface_threshold = visualize_isosurface_threshold
        self.evaluate_isosurface_threshold = evaluate_isosurface_threshold
        self.use_overlap_regularizer = use_overlap_regularizer
        self.overlap_threshold = overlap_threshold
        self.overlap_regularizer_loss_weight = overlap_regularizer_loss_weight

        self.use_l1_occupancy_regularizer = use_l1_occupancy_regularizer
        self.l1_occupancy_regularizer_loss_weight = l1_occupancy_regularizer_loss_weight

        self.use_canonical_self_supervised_slide_loss = use_canonical_self_supervised_slide_loss
        self.use_canonical_recon_loss = use_canonical_recon_loss
        self.use_canonical_occ_loss = use_canonical_occ_loss
        self.train_canonical = self.use_canonical_recon_loss or self.use_canonical_self_supervised_slide_loss or self.use_canonical_occ_loss
        self.self_supervised_slide_loss_weight = self_supervised_slide_loss_weight
        self.implicit_discriminator_activation_type = implicit_discriminator_activation_type
        if self.implicit_discriminator_activation_type == 'sigmoid':
            self.implicit_discriminator_activation = torch.sigmoid
        elif self.implicit_discriminator_activation_type == 'tanhshrink':
            self.implicit_discriminator_activation = torch.nn.Tanhshrink()
        elif self.implicit_discriminator_activation_type == 'none':
            self.implicit_discriminator_activation = torch.nn.Identity()
        else:
            raise NotImplementedError
        self.occupancy_reduction_type = occupancy_reduction_type
        assert self.occupancy_reduction_type in [
            'max', 'relu_sum', 'sigmoid_sum', 'relu_sum_direct',
            'sigmoid_sum_direct', 'min', 'negative_logsumexp', 'logsumexp',
            'sign_wise_sum', 'logsumexp_sigmoid', 'softmax', 'softmax_sigmoid',
            'sigmoid_softmax'
        ]
        self.occupancy_reduction_type_for_discriminator = occupancy_reduction_type_for_discriminator
        assert self.occupancy_reduction_type_for_discriminator in [
            'max', 'relu_sum', 'sigmoid_sum', 'relu_sum_direct',
            'sigmoid_sum_direct', 'min_direct', 'negative_logsumexp',
            'logsumexp', 'sign_wise_sum', 'max_sigmoid', 'logsumexp_sigmoid',
            'softmax', 'softmax_sigmoid', 'sigmoid_softmax'
        ]
        self.points_label_for_discriminator_scale = points_label_for_discriminator_scale
        self.is_movenet_classification_head = is_movenet_classification_head
        self.multilabel_classification_criterion_type = multilabel_classification_criterion_type
        assert self.multilabel_classification_criterion_type in [
            'softmargin', 'crossentropy'
        ]
        if self.multilabel_classification_criterion_type == 'crossentropy':
            self.multilabel_classification_criterion = torch.nn.CrossEntropyLoss(
                reduction='none')
        else:
            raise NotImplementedError

        assert self.train_explicit != self.train_implicit
        self.visualizer = visualizer_util.Visualizer(
            self.primitive_num, marker_size=visualizer_marker_size)

        self.is_movenet_decode_sdf = is_movenet_decode_sdf
        if self.is_movenet_decode_sdf:
            assert self.occupancy_reduction_type in [
                'min', 'negative_logsumexp'
            ]
            assert self.occupancy_reduction_type_for_discriminator in [
                'min_direct', 'negative_logsumexp'
            ]
        self.overlap_regularizer_type = overlap_regularizer_type

        assert self.overlap_regularizer_type in [
            'default', 'only_positive', 'raw_value', 'nsd_style',
            'non_top_primitives'
        ]
        if self.overlap_regularizer_type == 'default':
            assert self.implicit_discriminator_activation_type == 'sigmoid'

        self.disable_moved_overlap_regularizer_loss = disable_moved_overlap_regularizer_loss
        self.sdf_loss_weight = sdf_loss_weight
        self.is_clamp_sdf_value = is_clamp_sdf_value
        self.clamp_sdf_value = clamp_sdf_value
        self.use_discriminator_input_mask = use_discriminator_input_mask
        if self.use_discriminator_input_mask:
            assert not self.is_movenet_decode_sdf
            assert not self.is_movenet_classification_head
        self.is_move_points = is_move_points

        self.is_temporal_pad_slide_dim = is_temporal_pad_slide_dim
        if self.is_temporal_pad_slide_dim:
            assert self.model.param_dim == 2
            assert self.is_move_points
            assert not is_movenet_classification_head
            assert not is_movenet_decode_sdf
        self.discriminator_input_mask_type = discriminator_input_mask_type
        assert discriminator_input_mask_type in [
            'default', 'softmax_attention', 'sigmoid_normalize', 'raw',
            'raw_relu'
        ]
        self.gan_type = gan_type
        assert self.gan_type in ['lsgan', 'wgangp']
        self.use_acgan = use_acgan
        if self.use_acgan:
            assert self.gan_type in ['lsgan']

        self.learn_D_slide_loss = learn_D_slide_loss
        self.D_slide_loss_weight = D_slide_loss_weight
        self.G_slide_loss_weight = G_slide_loss_weight
        self.learn_D_canonical_loss = learn_D_canonical_loss
        self.D_canonical_loss_weight = D_canonical_loss_weight
        self.G_canonical_loss_weight = G_canonical_loss_weight
        self.learn_moving_self_slide_reg = learn_moving_self_slide_reg
        self.is_detach_param_in_generator = is_detach_param_in_generator
        self.is_move_points_slide_bug_fixed = is_move_points_slide_bug_fixed
        self.use_wgangp_latent_code_interpolation = use_wgangp_latent_code_interpolation
        self.part_iou_vs_slide_envelope_loss_minimum_slide_loss_threshold = part_iou_vs_slide_envelope_loss_minimum_slide_loss_threshold
        self.use_part_shape_similarity_loss = use_part_shape_similarity_loss
        if self.use_part_shape_similarity_loss:
            assert not is_movenet_classification_head
            assert not is_movenet_decode_sdf
        self.part_shape_similarity_loss_type = part_shape_similarity_loss_type
        if self.part_shape_similarity_loss_type == 'TripletMarginLoss':
            self.part_shape_similarity_loss_func = metric_losses.TripletMarginLoss(
                margin=triplet_margin_loss_margin)
        else:
            raise NotImplementedError
        self.use_part_shape_similarity_loss_hard_miner = use_part_shape_similarity_loss_hard_miner

        if self.use_part_shape_similarity_loss_hard_miner:
            self.part_shape_similarity_miner_type = part_shape_similarity_miner_type
            if self.part_shape_similarity_miner_type == 'MultiSimilarityMiner':
                self.part_shape_similarity_miner_func = metric_miners.MultiSimilarityMiner(
                    multi_similarity_miner_epsilon)
            else:
                raise NotImplementedError
        self.part_shape_similarity_loss_weight = part_shape_similarity_loss_weight
        self.disable_gan_training = disable_gan_training
        if self.disable_gan_training:
            assert not self.use_acgan

        self.disable_induction_by_moving = disable_induction_by_moving
        if self.disable_induction_by_moving:
            assert not self.use_canonical_self_supervised_slide_loss
        self.use_entire_shape_as_a_part_in_similarity_loss = use_entire_shape_as_a_part_in_similarity_loss
        if self.use_entire_shape_as_a_part_in_similarity_loss:
            assert self.use_part_shape_similarity_loss

        self.use_entropy_reduction_loss = use_entropy_reduction_loss
        if self.use_entropy_reduction_loss:
            assert not is_movenet_classification_head
            assert not is_movenet_decode_sdf
        self.each_entropy_reduction_loss_margin = each_entropy_reduction_loss_margin
        self.whole_entropy_reduction_loss_margin = whole_entropy_reduction_loss_margin
        self.entropy_reduction_loss_weight = entropy_reduction_loss_weight
        self.disable_part_seg_loss_iou_for_eval = disable_part_seg_loss_iou_for_eval
        self.use_unsupervised_part_iou_for_eval = use_unsupervised_part_iou_for_eval
        self.entropy_reduction_loss_type = entropy_reduction_loss_type
        assert self.entropy_reduction_loss_type in ['aabb', 'obb', 'convex']
        if self.entropy_reduction_loss_type in ['obb', 'convex']:
            self.entropy_reduction_loss_primitive_executor = process.ProcessPoolExecutor(
                max_workers=entropy_reduction_loss_primitive_worker_num)
        self.use_gt_values_as_whole_shape_in_entropy_reduction_loss = use_gt_values_as_whole_shape_in_entropy_reduction_loss
        self.ignore_background_in_part_iou_loss = ignore_background_in_part_iou_loss
        self.occupancy_reduction_loss_type = occupancy_reduction_loss_type
        self.imnet_occupancy_reduction_loss_types = [
            'imnet', 'imnet_ignore_gt_one'
        ]
        assert self.occupancy_reduction_loss_type in [
            'occnet', *self.imnet_occupancy_reduction_loss_types
        ]
        self.non_top_primitives_overlap_loss_margin = non_top_primitives_overlap_loss_margin
        self.use_imnet_style_occupancy_loss_with_logit = use_imnet_style_occupancy_loss_with_logit
        if use_imnet_style_occupancy_loss_with_logit:
            assert self.occupancy_reduction_loss_type == 'occnet'
        self.use_tsdf_in_occupancy_loss = use_tsdf_in_occupancy_loss
        self.loss_optimizer_pairs = loss_optimizer_pairs
        self.use_overlap_regularizer_loss = use_overlap_regularizer_loss
        assert not (self.use_overlap_regularizer_loss
                    and self.use_overlap_regularizer)
        self.is_occupancy_reduction_softmax_decay = is_occupancy_reduction_softmax_decay
        self.occupancy_reduction_softmax_mix_ratio = occupancy_reduction_softmax_mix_ratio
        self.occupancy_reduction_softmax_mix_ratio_max = occupancy_reduction_softmax_mix_ratio_max
        self.occupancy_reduction_softmax_decay_max_steps = occupancy_reduction_softmax_decay_max_steps
        self.occupancy_reduction_softmax_grow_ratio = occupancy_reduction_softmax_grow_ratio
        self.use_mse_in_occupancy_loss = use_mse_in_occupancy_loss
        self.occupancy_reduction_decay_type = occupancy_reduction_decay_type
        assert self.occupancy_reduction_decay_type in ['exponential', 'linear']
        self.occupancy_reduction_logsumexp_mix_ratio = occupancy_reduction_logsumexp_mix_ratio
        self.is_occupancy_reduction_logsumexp_decay = is_occupancy_reduction_logsumexp_decay
        self.use_bce_without_logits = use_bce_without_logits
        self.current_train_step = 0.

    def eval_step(self, data, step):
        #self.model.eval()
        all_losses = {}

        with torch.no_grad():
            for losses in self.compute_loss(data,
                                            step,
                                            return_eval_loss=True,
                                            skip_gp=True):
                all_losses.update(losses)
        return all_losses

    def visualize(self, data):
        #self.model.eval()
        pointcloud_moving_points = data['pointcloud_moving_points'].to(
            self.device)
        pointcloud_moving_label = data['pointcloud_moving_label'].long()
        if not self.disable_induction_by_moving:
            slide = data['slide'].unsqueeze(-1)
            if self.is_temporal_pad_slide_dim:
                slide = torch.cat([torch.zeros_like(slide), slide], axis=-1)
        # recon result
        # recon moved by pred slide
        # recon moved by gt slide
        # input (gt) moved by gt slide
        # input (gt) moved by pred slide

        result_images = []

        if self.train_explicit:
            images = self.visualizer.visualize_pointcloud(
                pointcloud_moving_points.clone().detach().cpu().numpy(),
                pointcloud_moving_label.numpy())
            result_images.append({
                'type': 'image',
                'desc': 'original_moving_points',
                'data': images
            })

            with torch.no_grad():
                ret = self.model(pointcloud_moving_points)
            pred_points = ret['point_set']
            pred_slide = ret['param'].detach().cpu()

            # Original pred points
            moved_pred_points_by_pred_list = [
                points.detach().cpu().clone() for points in pred_points
            ]
            pred_label = torch.cat([
                torch.zeros(points.shape[:2]) + (idx + 1)
                for idx, points in enumerate(moved_pred_points_by_pred_list)
            ],
                                   axis=1).numpy()

            moved_pred_points_by_pred = torch.cat(
                moved_pred_points_by_pred_list, axis=1).numpy()

            images = self.visualizer.visualize_pointcloud(
                moved_pred_points_by_pred, pred_label)

            result_images.append({
                'type': 'image',
                'desc': 'original_pred_points',
                'data': images
            })
            # assuming the 1 th primitive is moving
            # recon moved by pred slide
            moved_pred_points_by_pred_list = [
                points.detach().cpu().clone() for points in pred_points
            ]
            pred_label = torch.cat([
                torch.zeros(points.shape[:2]) + (idx + 1)
                for idx, points in enumerate(moved_pred_points_by_pred_list)
            ],
                                   axis=1).numpy()

            moved_pred_points_by_pred_list[1][:, :, 1] -= pred_slide

            moved_pred_points_by_pred = torch.cat(
                moved_pred_points_by_pred_list, axis=1).numpy()

            images = self.visualizer.visualize_pointcloud(
                moved_pred_points_by_pred, pred_label)

            result_images.append({
                'type': 'image',
                'desc': 'moved_pred_points_by_pred',
                'data': images
            })

            # recon moved by gt slide
            moved_pred_points_by_gt_list = [
                points.detach().cpu().clone() for points in pred_points
            ]
            pred_label = torch.cat([
                torch.zeros(points.shape[:2]) + (idx + 1)
                for idx, points in enumerate(moved_pred_points_by_gt_list)
            ],
                                   axis=1).numpy()

            moved_pred_points_by_gt_list[1][:, :, 1] -= slide

            moved_pred_points_by_gt = torch.cat(moved_pred_points_by_gt_list,
                                                axis=1).numpy()

            images = self.visualizer.visualize_pointcloud(
                moved_pred_points_by_gt, pred_label)

            result_images.append({
                'type': 'image',
                'desc': 'moved_pred_points_by_gt',
                'data': images
            })

            # gt moved by pred slide
            # The label index starts from 1, so the 1st primitive (0 origin) is 2.
            moved_gt_points_by_pred_tensor = pointcloud_moving_points.detach(
            ).cpu().clone()
            moved_gt_points_by_pred_batchwise_list = []
            new_label_batchwise_list = []
            for batch_idx in range(moved_gt_points_by_pred_tensor.shape[0]):
                moved_gt_points_by_pred_list = []
                new_label_list = []
                for idx in range(1, self.model.primitive_num + 1):
                    t = moved_gt_points_by_pred_tensor[
                        batch_idx,
                        pointcloud_moving_label[batch_idx, :] == idx, :]
                    if idx == 2:
                        t[:, 1] -= pred_slide[batch_idx]
                    moved_gt_points_by_pred_list.append(t)
                    new_label_list.append(torch.zeros(t.shape[0]) + idx)

                moved_gt_points_by_pred_batchwise_list.append(
                    torch.cat(moved_gt_points_by_pred_list, axis=0))
                new_label_batchwise_list.append(
                    torch.cat(new_label_list, axis=0))

            moved_gt_points_by_pred = torch.stack(
                moved_gt_points_by_pred_batchwise_list, axis=0).numpy()
            new_label = torch.stack(new_label_batchwise_list, axis=0)

            images = self.visualizer.visualize_pointcloud(
                moved_gt_points_by_pred, new_label.numpy())
            result_images.append({
                'type': 'image',
                'desc': 'moved_gt_points_by_pred',
                'data': images
            })

            # gt moved by gt slide
            # The label index starts from 1, so the 1st primitive (0 origin) is 2.
            moved_gt_points_by_gt_tensor = pointcloud_moving_points.detach(
            ).cpu().clone()
            moved_gt_points_by_gt_batchwise_list = []
            new_label_batchwise_list = []
            for batch_idx in range(moved_gt_points_by_gt_tensor.shape[0]):
                moved_gt_points_by_gt_list = []
                new_label_list = []
                for idx in range(1, self.model.primitive_num + 1):
                    t = moved_gt_points_by_gt_tensor[
                        batch_idx,
                        pointcloud_moving_label[batch_idx, :] == idx, :]
                    if idx == 2:
                        t[:, 1] -= slide[batch_idx]
                    moved_gt_points_by_gt_list.append(t)
                    new_label_list.append(torch.zeros(t.shape[0]) + idx)

                moved_gt_points_by_gt_batchwise_list.append(
                    torch.cat(moved_gt_points_by_gt_list, axis=0))
                new_label_batchwise_list.append(
                    torch.cat(new_label_list, axis=0))

            moved_gt_points_by_gt = torch.stack(
                moved_gt_points_by_gt_batchwise_list, axis=0).numpy()
            new_label = torch.stack(new_label_batchwise_list, axis=0)

            images = self.visualizer.visualize_pointcloud(
                moved_gt_points_by_gt, new_label.numpy())

            result_images.append({
                'type': 'image',
                'desc': 'moved_gt_points_by_gt',
                'data': images
            })
        elif self.train_implicit:
            if self.is_movenet_decode_sdf:
                points_moving_points = data['sdf_points_moving_points'].to(
                    self.device)
                points_moving_label = data['sdf_points_moving_label']
            else:
                points_moving_points = data['points_moving_points'].to(
                    self.device)
                points_moving_label = data['points_moving_label'].long()
            # recon result
            # recon moved by pred slide
            # recon moved by gt slide
            # input (gt) moved by gt slide
            # input (gt) moved by pred slide

            points_moving_points_numpy = points_moving_points.clone().detach(
            ).cpu().numpy()

            if self.is_movenet_decode_sdf:
                pred_occ_max, pred_occ_argmax = points_moving_label.min(
                    axis=-1)
                pred_occ_label = torch.where(
                    pred_occ_max <= 0, pred_occ_argmax + 1,
                    torch.zeros_like(pred_occ_max).long())

                images = self.visualizer.visualize_pointcloud(
                    points_moving_points_numpy, pred_occ_label.numpy())
            else:
                images = self.visualizer.visualize_pointcloud(
                    points_moving_points_numpy, points_moving_label.numpy())
            result_images.append({
                'type': 'image',
                'desc': 'original_moving_points',
                'data': images
            })
            with torch.no_grad():
                ret = self.model(pointcloud_moving_points,
                                 points_moving_points)
            pred_occ = ret['occupancy']

            if self.is_movenet_classification_head:
                pred_occ_label = torch.argmax(pred_occ, axis=-1)
            elif self.is_movenet_decode_sdf:
                pred_occ_max, pred_occ_argmax = pred_occ.min(axis=-1)
                pred_occ_label = torch.where(
                    pred_occ_max <= 0, pred_occ_argmax + 1,
                    torch.zeros_like(pred_occ_max).long())
            else:
                if self.occupancy_reduction_loss_type in self.imnet_occupancy_reduction_loss_types:
                    pred_occ_max, pred_occ_argmax = pred_occ.max(axis=-1)
                elif self.occupancy_reduction_loss_type == 'occnet':
                    pred_occ_max, pred_occ_argmax = torch.sigmoid(
                        pred_occ).max(axis=-1)
                else:
                    raise NotImplementedError

                pred_occ_label = torch.where(
                    pred_occ_max >= self.visualize_isosurface_threshold,
                    pred_occ_argmax + 1,
                    torch.zeros_like(pred_occ_max).long())

            # Original pred points
            images = self.visualizer.visualize_pointcloud(
                points_moving_points_numpy,
                pred_occ_label.detach().cpu().numpy())

            result_images.append({
                'type': 'image',
                'desc': 'original_pred_points',
                'data': images
            })

            # B, 1
            pred_slide = ret['param']
            # assuming the 1 th primitive is moving
            # recon moved by pred slide (batch, 1)
            # TODO: Loop by number of primitives

            # moved_pred_points_by_pred
            if not self.disable_induction_by_moving:
                moved_points_moving_points_by_pred = self.move_points(
                    pred_occ, points_moving_points, pred_slide)

                moving_idx = self.get_moving_idx()
                if not self.is_move_points:
                    with torch.no_grad():
                        moved_ret = self.model(
                            pointcloud_moving_points,
                            moved_points_moving_points_by_pred)
                    moved_pred_occ = moved_ret['occupancy']
                    pred_occ[:, :,
                             moving_idx] = moved_pred_occ[:, :,
                                                          moving_idx]  # * primitive_mask_pass

                if self.is_movenet_classification_head:
                    pred_occ_label = torch.argmax(pred_occ, axis=-1)
                elif self.is_movenet_decode_sdf:
                    pred_occ_max, pred_occ_argmax = pred_occ.min(axis=-1)
                    pred_occ_label = torch.where(
                        pred_occ_max <= 0, pred_occ_argmax + 1,
                        torch.zeros_like(pred_occ_max).long())
                else:
                    if self.occupancy_reduction_loss_type in self.imnet_occupancy_reduction_loss_types:
                        pred_occ_max, pred_occ_argmax = pred_occ.max(axis=-1)
                    elif self.occupancy_reduction_loss_type == 'occnet':
                        pred_occ_max, pred_occ_argmax = torch.sigmoid(
                            pred_occ).max(axis=-1)
                    else:
                        raise NotImplementedError

                    pred_occ_label = torch.where(
                        pred_occ_max >= self.visualize_isosurface_threshold,
                        pred_occ_argmax + 1,
                        torch.zeros_like(pred_occ_max).long())

                if self.is_move_points:
                    images = self.visualizer.visualize_pointcloud(
                        moved_points_moving_points_by_pred.detach().cpu().
                        numpy(),
                        pred_occ_label.detach().cpu().numpy())
                else:
                    images = self.visualizer.visualize_pointcloud(
                        points_moving_points_numpy,
                        pred_occ_label.detach().cpu().numpy())

                result_images.append({
                    'type': 'image',
                    'desc': 'moved_pred_points_by_pred',
                    'data': images
                })

                # assuming the 1 th primitive is moving
                # recon moved by pred slide (batch, 1)
                # TODO: Loop by number of primitives
                moved_points_moving_points_by_gt = points_moving_points.clone()

                slide = slide.to(self.device)
                moved_points_moving_points_by_gt = self.move_points(
                    pred_occ, points_moving_points, slide)

                if not self.is_move_points:
                    with torch.no_grad():
                        moved_ret = self.model(
                            pointcloud_moving_points,
                            moved_points_moving_points_by_gt)
                    moved_pred_occ = moved_ret['occupancy']
                    pred_occ[:, :,
                             moving_idx] = moved_pred_occ[:, :,
                                                          moving_idx]  # * primitive_mask_pass

                if self.is_movenet_classification_head:
                    pred_occ_label = torch.argmax(pred_occ, axis=-1)
                elif self.is_movenet_decode_sdf:
                    pred_occ_max, pred_occ_argmax = pred_occ.min(axis=-1)
                    pred_occ_label = torch.where(
                        pred_occ_max <= 0, pred_occ_argmax + 1,
                        torch.zeros_like(pred_occ_max).long())
                else:
                    if self.occupancy_reduction_loss_type in self.imnet_occupancy_reduction_loss_types:
                        pred_occ_max, pred_occ_argmax = pred_occ.max(axis=-1)
                    elif self.occupancy_reduction_loss_type == 'occnet':
                        pred_occ_max, pred_occ_argmax = torch.sigmoid(
                            pred_occ).max(axis=-1)
                    else:
                        raise NotImplementedError

                    pred_occ_label = torch.where(
                        pred_occ_max >= self.visualize_isosurface_threshold,
                        pred_occ_argmax + 1,
                        torch.zeros_like(pred_occ_max).long())

                if self.is_move_points:
                    images = self.visualizer.visualize_pointcloud(
                        moved_points_moving_points_by_gt.detach().cpu().numpy(
                        ),
                        pred_occ_label.detach().cpu().numpy())
                else:
                    images = self.visualizer.visualize_pointcloud(
                        points_moving_points_numpy,
                        pred_occ_label.detach().cpu().numpy())

                result_images.append({
                    'type': 'image',
                    'desc': 'moved_pred_points_by_gt',
                    'data': images
                })

            pred_slide = pred_slide.detach().cpu()
        else:
            raise NotImplementedError

        result_images.append({
            'desc': 'pred_slide',
            'type': 'array',
            'data': pred_slide.numpy()
        })
        return result_images

    def train_step(self, data, step):
        self.current_training_step = step
        self.model.train()
        all_losses = {}
        for losses in self.compute_loss(data, step):
            for loss_name, optimizer_dict in self.loss_optimizer_pairs.items():
                if not loss_name in losses:
                    continue
                for optimizer_name, optimize_config in optimizer_dict.items():
                    every = optimize_config.get('every', 1)
                    if not step % every == 0:
                        continue
                    self.optimizers[optimizer_name].zero_grad()
                    if 'independent_D_loss' in losses and optimizer_name == 'discriminator':
                        self.optimizers['discriminator'].zero_grad()

                losses[loss_name].backward()
                if 'independent_D_loss' in losses and optimizer_name == 'discriminator':
                    losses['independent_D_loss'].backward()

                for optimizer_name, optimize_config in optimizer_dict.items():
                    every = optimize_config.get('every', 1)
                    if not step % every == 0:
                        continue
                    assert not torch.any(torch.isnan(losses[loss_name]))
                    self.optimizers[optimizer_name].step()

                    if 'independent_D_loss' in losses and optimizer_name == 'discriminator':
                        assert not torch.any(
                            torch.isnan(losses['independent_D_loss']))
                        self.optimizers['discriminator'].step()
            all_losses.update(losses)
        """
        for losses in self.compute_loss(data, step):
            if 'total_D_loss' in losses:
                self.optimizers['discriminator'].zero_grad()
                assert not torch.any(torch.isnan(losses['total_D_loss']))
                losses['total_D_loss'].backward()
                self.optimizers['discriminator'].step()
                if 'independent_D_loss' in losses:
                    self.optimizers['discriminator'].zero_grad()
                    assert not torch.any(
                        torch.isnan(losses['independent_D_loss']))
                    losses['independent_D_loss'].backward()
                    self.optimizers['discriminator'].step()
            elif 'total_G_loss' in losses:
                self.optimizers['generator'].zero_grad()
                assert not torch.any(torch.isnan(losses['total_G_loss']))
                losses['total_G_loss'].backward()
                self.optimizers['generator'].step()
            all_losses.update(losses)
        """

        return all_losses

    def compute_loss(self, data, step, skip_gp=False, return_eval_loss=False):
        if self.train_explicit:
            for loss in self.train_explicit_losses(
                    data, step, skip_gp, return_eval_loss=return_eval_loss):
                yield loss
        elif self.train_implicit:
            for loss in self.train_implicit_losses(
                    data, step, skip_gp, return_eval_loss=return_eval_loss):
                yield loss
        else:
            raise NotImplementedError

    def get_indicator_value_for_discriminator(self, pred_occ):
        if self.occupancy_reduction_type_for_discriminator == 'relu_sum':
            pred_occ_max = self.implicit_discriminator_activation(
                torch.relu(pred_occ).sum(axis=-1))
        elif self.occupancy_reduction_type_for_discriminator == 'sigmoid_sum':
            pred_occ_max = self.implicit_discriminator_activation(
                torch.sigmoid(pred_occ).sum(axis=-1))
        elif self.occupancy_reduction_type_for_discriminator == 'relu_sum_direct':
            pred_occ_max = torch.relu(pred_occ).sum(axis=-1)
        elif self.occupancy_reduction_type_for_discriminator == 'sigmoid_sum_direct':
            pred_occ_max = torch.sigmoid(pred_occ).sum(axis=-1)
        elif self.occupancy_reduction_type_for_discriminator == 'max':
            pred_occ_max, _ = self.implicit_discriminator_activation(
                pred_occ).max(axis=-1)
        elif self.occupancy_reduction_type_for_discriminator == 'min_direct':
            pred_occ_max, _ = pred_occ.min(axis=-1)
        elif self.occupancy_reduction_type_for_discriminator == 'negative_logsumexp':
            pred_occ_max = -torch.logsumexp(-pred_occ, -1)
        elif self.occupancy_reduction_type_for_discriminator == 'logsumexp':
            if self.is_occupancy_reduction_logsumexp_decay:
                alpha = self.get_smooth_max_decay_alpha()
            else:
                alpha = self.occupancy_reduction_logsumexp_mix_ratio
            pred_occ_max = (1. / alpha) * torch.logsumexp(pred_occ * alpha, -1)
        elif self.occupancy_reduction_type_for_discriminator == 'sign_wise_sum':
            pos_sum = torch.relu(pred_occ).sum(-1)
            neg_sum = -torch.relu(-pred_occ).sum(-1)
            pred_occ_max = pos_sum + neg_sum
        elif self.occupancy_reduction_type_for_discriminator == 'max_sigmoid':
            pred_occ_max, _ = pred_occ.max(axis=-1)
            pred_occ_max = torch.sigmoid(pred_occ_max)
        elif self.occupancy_reduction_type_for_discriminator == 'logsumexp_sigmoid':
            if self.is_occupancy_reduction_logsumexp_decay:
                alpha = self.get_smooth_max_decay_alpha()
            else:
                alpha = self.occupancy_reduction_logsumexp_mix_ratio
            pred_occ_max = (1. / alpha) * torch.logsumexp(pred_occ * alpha, -1)
            pred_occ_max = torch.sigmoid(pred_occ_max)
        elif self.occupancy_reduction_type_for_discriminator in [
                'softmax', 'softmax_sigmoid', 'sigmoid_softmax'
        ]:
            if self.occupancy_reduction_type == 'sigmoid_softmax':
                pred_occ = torch.sigmoid(pred_occ)
            if self.is_occupancy_reduction_softmax_decay:
                alpha = self.get_smooth_max_decay_alpha()
            else:
                alpha = self.occupancy_reduction_softmax_mix_ratio

            pred_occ_max = (
                pred_occ *
                torch.nn.functional.softmax(alpha * pred_occ, -1)).sum(-1)
            if self.occupancy_reduction_type_for_discriminator == 'softmax_sigmoid':
                pred_occ_max = torch.sigmoid(pred_occ_max)
        else:
            raise NotImplementedError

        return pred_occ_max

    def get_moving_idx(self):
        if self.is_movenet_classification_head:
            moving_idx = 1 + 1
        else:
            moving_idx = 1
        return moving_idx

    def move_points(self, occ, points, slide):
        moving_idx = self.get_moving_idx()
        moved_points_moving_points = points.clone()
        if self.is_move_points:
            # B, P, 1
            occ_argmax = occ.argmax(-1, keepdims=True)
            if self.is_temporal_pad_slide_dim:
                rg = range(self.primitive_num)
            else:
                rg = [moving_idx]
            for idx, midx in enumerate(rg):
                primitive_mask = (occ_argmax == midx)
                # B, 1, 2
                slide_sliced = slide[:, idx].unsqueeze(-1)
                slide_expanded = torch.cat(
                    [torch.zeros_like(slide_sliced), slide_sliced],
                    axis=-1).unsqueeze(1)
                a = moved_points_moving_points * torch.logical_not(
                    primitive_mask)
                if self.is_move_points_slide_bug_fixed:
                    b = moved_points_moving_points * primitive_mask - slide_expanded * primitive_mask
                else:
                    b = (moved_points_moving_points -
                         slide_expanded) * primitive_mask
                moved_points_moving_points = a + b
        else:
            moved_points_moving_points[:, :,
                                       1] = moved_points_moving_points[:, :,
                                                                       1] + slide

        return moved_points_moving_points

    def get_discriminator_input_mask(self, label):
        assert label.ndim == 2
        if self.is_movenet_classification_head:
            raise NotImplementedError
        elif self.is_movenet_decode_sdf:
            raise NotImplementedError
        else:
            if self.discriminator_input_mask_type == 'default':
                zeros = torch.zeros_like(label)
                mask = torch.where(label >= 0, zeros, zeros - float('inf'))
                mask = mask.detach()
                mask = (mask >= 0)
            elif self.discriminator_input_mask_type == 'softmax_attention':
                mask = F.softmax(label, dim=-1)
            elif self.discriminator_input_mask_type == 'sigmoid_normalize':
                mask = torch.sigmoid(label)
            elif self.discriminator_input_mask_type == 'raw':
                mask = label
            elif self.discriminator_input_mask_type == 'raw_relu':
                mask = torch.relu(label)
            else:
                raise NotImplementedError

        return mask

    def get_indicator_value(self, pred_occ):
        if self.occupancy_reduction_type == 'max':
            pred_occ_max_logit, _ = pred_occ.max(-1)
        elif self.occupancy_reduction_type == 'min':
            pred_occ_max_logit, _ = pred_occ.min(-1)
        elif self.occupancy_reduction_type in ['relu_sum', 'relu_sum_direct']:
            pred_occ_max_logit = torch.relu(pred_occ).sum(-1)
        elif self.occupancy_reduction_type in [
                'sigmoid_sum', 'sigmoid_sum_direct'
        ]:
            pred_occ_max_logit = torch.sigmoid(pred_occ).sum(-1)
        elif self.occupancy_reduction_type == 'negative_logsumexp':
            if self.is_occupancy_reduction_logsumexp_decay:
                alpha = self.get_smooth_max_decay_alpha()
            else:
                alpha = self.occupancy_reduction_logsumexp_mix_ratio
            pred_occ_max_logit = -(1. / alpha) * torch.logsumexp(
                -pred_occ * alpha, -1)
        elif self.occupancy_reduction_type == 'logsumexp':
            if self.is_occupancy_reduction_logsumexp_decay:
                alpha = self.get_smooth_max_decay_alpha()
            else:
                alpha = self.occupancy_reduction_logsumexp_mix_ratio
            pred_occ_max_logit = (1. / alpha) * torch.logsumexp(
                pred_occ * alpha, -1)
        elif self.occupancy_reduction_type == 'sign_wise_sum':
            pos_sum = torch.relu(pred_occ).sum(-1)
            neg_sum = -torch.relu(-pred_occ).sum(-1)
            pred_occ_max_logit = pos_sum + neg_sum
        elif self.occupancy_reduction_type == 'logsumexp_sigmoid':
            if self.is_occupancy_reduction_logsumexp_decay:
                alpha = self.get_smooth_max_decay_alpha()
            else:
                alpha = self.occupancy_reduction_logsumexp_mix_ratio
            pred_occ_max = (1. / alpha) * torch.logsumexp(pred_occ * alpha, -1)
            pred_occ_max_logit = torch.sigmoid(pred_occ_max)
        elif self.occupancy_reduction_type in [
                'softmax', 'softmax_sigmoid', 'sigmoid_softmax'
        ]:
            if self.occupancy_reduction_type == 'sigmoid_softmax':
                pred_occ = torch.sigmoid(pred_occ)
            if self.is_occupancy_reduction_softmax_decay:
                alpha = self.get_smooth_max_decay_alpha()
            else:
                alpha = self.occupancy_reduction_softmax_mix_ratio

            pred_occ_max_logit = (
                pred_occ *
                torch.nn.functional.softmax(alpha * pred_occ, -1)).sum(-1)
            if self.occupancy_reduction_type == 'softmax_sigmoid':
                pred_occ_max_logit = torch.sigmoid(pred_occ_max_logit)
        else:
            raise NotImplementedError(
                self.occupancy_reduction_type_for_discriminator)

        return pred_occ_max_logit

    def get_smooth_max_decay_alpha(self):
        if (self.current_training_step <
                self.occupancy_reduction_softmax_decay_max_steps):
            if self.occupancy_reduction_decay_type == 'exponential':
                alpha = self.occupancy_reduction_softmax_mix_ratio * (
                    1 + self.occupancy_reduction_softmax_grow_ratio
                )**self.current_training_step
                alpha = min(alpha,
                            self.occupancy_reduction_softmax_mix_ratio_max)
            elif self.occupancy_reduction_decay_type == 'linear':
                alpha = self.occupancy_reduction_softmax_mix_ratio + self.occupancy_reduction_softmax_grow_ratio * self.current_training_step
                alpha = min(alpha,
                            self.occupancy_reduction_softmax_mix_ratio_max)
        else:
            alpha = self.occupancy_reduction_softmax_mix_ratio_max
        return alpha

    def get_implicit_reconstruction_loss(
        self,
        pred_occ,
        label,
        weight=None,
        loss_weight=None,
        prefix='',
        is_reduction=True,
        disable_logits_bce=False,
    ):
        pred_occ_max_logit = self.get_indicator_value(pred_occ)

        losses = {}
        ## recon loss
        if self.is_movenet_classification_head:
            points_pred_label_for_discriminator = F.softmax(pred_occ, dim=-1)
            points_pred_label_for_discriminator = torch.stack([
                points_pred_label_for_discriminator[:, :, 0],
                points_pred_label_for_discriminator[:, :, 1:].sum(-1)
            ],
                                                              axis=-1)
            if self.multilabel_classification_criterion_type == 'crossentropy':
                occ_loss = self.multilabel_classification_criterion(
                    points_pred_label_for_discriminator.transpose(
                        1, 2).contiguous(),
                    label.clamp(max=1).long()).sum(-1).mean()
            else:
                raise NotImplementedError

            if loss_weight is not None:
                occ_loss_weighted = occ_loss * loss_weight
            else:
                occ_loss_weighted = occ_loss * self.occupancy_loss_weight
            losses['occ_loss'] = occ_loss
            losses['occ_loss_weighted'] = occ_loss_weighted

        elif self.is_movenet_decode_sdf:
            label, _ = label.min(-1)

            if self.is_clamp_sdf_value:
                occ_loss = F.l1_loss(pred_occ_max_logit.clamp(
                    -self.clamp_sdf_value, self.clamp_sdf_value),
                                     label.clamp(-self.clamp_sdf_value,
                                                 self.clamp_sdf_value),
                                     reduction='sum')
            else:
                occ_loss = F.l1_loss(pred_occ_max_logit,
                                     label,
                                     reduction='sum')
            if loss_weight is not None:
                occ_loss_weighted = occ_loss * loss_weight
            else:
                occ_loss_weighted = occ_loss * self.sdf_loss_weight
            losses['sdf_loss'] = occ_loss
            losses['sdf_loss_weighted'] = occ_loss_weighted
        else:
            if self.use_tsdf_in_occupancy_loss:
                label = label
            else:
                label = label.clamp(max=1)

            occ_loss = self.get_occupancy_loss(
                pred_occ_max_logit,
                label,
                weight=weight,
                is_reduction=is_reduction,
                disable_logits_bce=disable_logits_bce)
            if loss_weight is not None:
                occ_loss_weighted = occ_loss * loss_weight
            else:
                occ_loss_weighted = occ_loss * self.occupancy_loss_weight
            losses['occ_loss'] = occ_loss
            losses['occ_loss_weighted'] = occ_loss_weighted
        if prefix:
            losses = {
                prefix + '_' + key: value
                for key, value in losses.items()
            }

        return losses, occ_loss_weighted

    def get_param_loss_for_eval(self, pred_slide, slide, prefix=''):
        loss = F.mse_loss(pred_slide, slide)
        prefix = (prefix + '_') if prefix != '' else prefix
        losses = {prefix + 'slide_loss': loss}
        return losses

    def get_overlap_loss_for_eval(self,
                                  pred_occ_after_move,
                                  pred_occ_before_move,
                                  prefix=''):
        if self.is_movenet_decode_sdf:
            pred_occ_after_move_bool = pred_occ_after_move <= 0
            overlap_after_move_bool = torch.relu(
                pred_occ_after_move_bool.sum(-1) - 1).sum()
            pred_occ_before_move_bool = pred_occ_before_move <= 0
            overlap_before_move_bool = torch.relu(
                pred_occ_before_move_bool.sum(-1) - 1).sum()
            losses = {
                'overlap_after_move_loss': overlap_after_move_bool,
                'overlap_before_move_loss': overlap_before_move_bool
            }
        else:
            pred_occ_after_move_bool = pred_occ_after_move >= 0
            overlap_after_move_bool = torch.relu(
                pred_occ_after_move_bool.sum(-1) - 1).sum()
            pred_occ_before_move_bool = pred_occ_before_move >= 0
            overlap_before_move_bool = torch.relu(
                pred_occ_before_move_bool.sum(-1) - 1).sum()
            losses = {
                'overlap_after_move_loss': overlap_after_move_bool,
                'overlap_before_move_loss': overlap_before_move_bool
            }
        prefix = (prefix + '_') if prefix != '' else prefix
        losses = {prefix + name: value for name, value in losses.items()}
        return losses

    def get_iou_loss_for_eval(self, pred_occ, label, prefix=''):
        if self.ignore_background_in_part_iou_loss:
            if self.is_movenet_classification_head:
                pred_occ_label = torch.argmax(pred_occ[..., 1:], dim=-1) + 1
            elif self.is_movenet_decode_sdf:
                pred_occ_max_vis, pred_occ_argmax_vis = pred_occ.min(axis=-1)
                pred_occ_label = pred_occ_argmax_vis + 1
            else:
                if self.occupancy_reduction_loss_type in self.imnet_occupancy_reduction_loss_types:
                    pred_occ_max_vis, pred_occ_argmax_vis = pred_occ.max(
                        axis=-1)
                elif self.occupancy_reduction_loss_type == 'occnet':
                    pred_occ_max_vis, pred_occ_argmax_vis = torch.sigmoid(
                        pred_occ).max(axis=-1)
                else:
                    raise NotImplementedError
                pred_occ_label = pred_occ_argmax_vis + 1
            if self.is_movenet_decode_sdf:
                occ2min, occ2argmin = label.min(-1)
                occ2 = occ2argmin + 1
            else:
                occ2 = label

        else:
            if self.is_movenet_classification_head:
                pred_occ_label = torch.argmax(pred_occ, dim=-1)
            elif self.is_movenet_decode_sdf:
                pred_occ_max_vis, pred_occ_argmax_vis = pred_occ.min(axis=-1)
                pred_occ_label = torch.where(
                    pred_occ_max_vis <= 0, pred_occ_argmax_vis + 1,
                    torch.zeros_like(pred_occ_max_vis).long())
            else:
                if self.occupancy_reduction_loss_type in self.imnet_occupancy_reduction_loss_types:
                    pred_occ_max_vis, pred_occ_argmax_vis = pred_occ.max(
                        axis=-1)
                elif self.occupancy_reduction_loss_type == 'occnet':
                    pred_occ_max_vis, pred_occ_argmax_vis = torch.sigmoid(
                        pred_occ).max(axis=-1)
                else:
                    raise NotImplementedError
                pred_occ_label = torch.where(
                    pred_occ_max_vis >= self.visualize_isosurface_threshold,
                    pred_occ_argmax_vis + 1,
                    torch.zeros_like(pred_occ_max_vis).long())

            if self.is_movenet_decode_sdf:
                occ2min, occ2argmin = label.min(-1)
                occ2 = torch.where(occ2min <= 0, occ2argmin + 1,
                                   torch.zeros_like(occ2min).long())
            else:
                occ2 = label

        losses = defaultdict(lambda: 0)

        if self.use_unsupervised_part_iou_for_eval:
            pred_labels = pred_occ_label.clone()
            gt_labels = occ2.clone().detach()
            losses['post_part_loss_iou'] = {
                'pred_labels': pred_labels,
                'gt_labels': gt_labels
            }
        else:
            for idx in range(1, self.primitive_num + 1):
                occ1 = pred_occ_label
                # Convert to boolean values
                occ1 = (occ1 >= self.evaluate_isosurface_threshold) & (occ1
                                                                       == idx)
                occ1 = occ1.squeeze(-1)
                occ2 = (occ2 >= self.evaluate_isosurface_threshold) & (occ2
                                                                       == idx)
                occ2 = occ2.squeeze(-1)

                # Compute IOU
                area_union = (occ1 | occ2).float().sum(axis=-1)
                area_intersect = (occ1 & occ2).sum(axis=-1)

                loss = (area_intersect / area_union).mean()

                losses['part_loss_iou_pm{}'.format(idx)] = loss
            total_loss = 0.
            for key, value in losses.items():
                total_loss += value / self.primitive_num

            losses['part_loss_iou'] = total_loss
            prefix = (prefix + '_') if prefix != '' else prefix
            losses = {prefix + name: value for name, value in losses.items()}
        return losses

    def get_part_iou_vs_slide_envelope_loss_for_eval(self, part_iou_loss,
                                                     slide_loss):
        loss = (part_iou_loss**2 + (1. - (slide_loss.clamp(
            max=self.
            part_iou_vs_slide_envelope_loss_minimum_slide_loss_threshold
        ) / self.part_iou_vs_slide_envelope_loss_minimum_slide_loss_threshold))
                **2).sqrt()
        ret = {'part_iou_slide_envelope_distance': loss}
        return ret

    def get_real_discriminator_result(self,
                                      label,
                                      points,
                                      points_values_requires_grad=False):
        if self.is_movenet_classification_head:
            points_canonical_label_for_discriminator = F.one_hot(
                label.clamp(max=1.).long(), num_classes=2).float()
        elif self.is_movenet_decode_sdf:
            points_canonical_label_for_discriminator, _ = label.min(
                -1, keepdim=True)

        else:
            points_canonical_label_for_discriminator = label.clamp(
                max=1.).unsqueeze(-1)
        if self.implicit_discriminator_activation_type in [
                'tanhshrink', 'none'
        ]:
            assert False
            points_canonical_label_for_discriminator = (
                points_canonical_label_for_discriminator -
                0.5) * 2 * self.points_label_for_discriminator_scale
        if self.use_voxel_discriminator:
            B = points.size(0)
            canonical_points_value = points_canonical_label_for_discriminator.view(B, 1, 16, 16, 16)
        else:
            canonical_points_value = torch.cat(
                [points, points_canonical_label_for_discriminator], axis=-1)
        canonical_mask = None
        if self.use_discriminator_input_mask:
            canonical_mask = self.get_discriminator_input_mask(label)
        if points_values_requires_grad:
            canonical_points_value.requires_grad = True
        ret = self.model(canonical_points_value,
                         mask=canonical_mask,
                         mode='discriminator')
        result = {
            'points_value': canonical_points_value,
            'mask': canonical_mask,
        }
        result.update(ret)
        return result

    def get_discriminator_result(self,
                                 pred_occ,
                                 points_moving_points,
                                 pred_slide,
                                 pointcloud_moving_points,
                                 no_move=False,
                                 detach=False,
                                 detach_param=False,
                                 detach_func=None):
        # TODO: Loop by number of primitives

        if no_move:
            pred_occ_after_move = pred_occ
            moved_points_moving_points_by_pred = points_moving_points
        else:
            if detach_param:
                if isinstance(pred_slide, torch.Tensor):
                    pred_slide = pred_slide.clone().detach()
                else:
                    pred_slide = detach_func(pred_slide)
            moved_points_moving_points_by_pred = self.move_points(
                pred_occ, points_moving_points, pred_slide)

            if self.is_move_points:
                pred_occ_after_move = pred_occ
            else:
                pred_occ_after_move = pred_occ.clone()
                with torch.no_grad():
                    moved_ret = self.model(pointcloud_moving_points,
                                           moved_points_moving_points_by_pred)
                moved_pred_occ = moved_ret['occupancy']
                pred_occ_after_move[:, :,
                                    1] = moved_pred_occ[:, :,
                                                        1]  # * primitive_mask_pass

        pred_occ_max = self.get_indicator_value_for_discriminator(
            pred_occ_after_move)

        #### moving (fake)
        if self.is_movenet_classification_head:
            points_pred_label_for_discriminator = F.softmax(
                pred_occ_after_move, dim=-1)
            points_pred_label_for_discriminator = torch.stack([
                points_pred_label_for_discriminator[:, :, 0],
                points_pred_label_for_discriminator[:, :, 1:].sum(-1)
            ],
                                                              axis=-1)
        else:
            points_pred_label_for_discriminator = pred_occ_max.unsqueeze(-1)

        if self.is_move_points:
            points_for_discriminator = moved_points_moving_points_by_pred
        else:
            points_for_discriminator = points_moving_points

        if self.use_voxel_discriminator:
            B = pred_occ_max.size(0)
            moved_pred_points_value = points_pred_label_for_discriminator.view(
                B, 1, 16, 16, 16)
        else:
            moved_pred_points_value = torch.cat([
                points_for_discriminator, points_pred_label_for_discriminator
            ],
                                                axis=-1)

        pred_mask = None
        if self.use_discriminator_input_mask:
            pred_occ_max, _ = pred_occ.max(-1)
            pred_mask = self.get_discriminator_input_mask(pred_occ_max)

        if detach:
            value_for_D = moved_pred_points_value.detach()
        else:
            value_for_D = moved_pred_points_value
        ret = self.model(value_for_D, mask=pred_mask, mode='discriminator')

        result = {
            'points_value': moved_pred_points_value,
            'mask': pred_mask,
            'pred_occ_after_move': pred_occ_after_move,
            'points_for_discriminator': points_for_discriminator
        }
        result.update(ret)

        return result

    def get_non_top_primitives_overlap_loss(self, values):
        values_argmax = values.argmax(-1, keepdim=True)
        pm_idx = torch.arange(values.size(-1),
                              dtype=values.dtype,
                              device=values.device).reshape(
                                  1, 1, -1).expand_as(values)

        non_topmask = values_argmax != pm_idx
        overlap_loss = torch.relu(
            (values + self.non_top_primitives_overlap_loss_margin) *
            non_topmask).sum(-1).mean()
        return overlap_loss

    def get_regularizer_overlap(self, pred_occ, primitive_moved_pred_occ):
        if self.overlap_regularizer_type == 'default':
            overlap_regularizer_loss = ((torch.relu(
                self.implicit_discriminator_activation(pred_occ).sum(-1) -
                self.overlap_threshold))**2).mean()

            moved_overlap_regularizer_loss = ((torch.relu(
                self.implicit_discriminator_activation(
                    primitive_moved_pred_occ).sum(-1) -
                self.overlap_threshold))**2).mean()
        elif self.overlap_regularizer_type == 'non_top_primitives':
            overlap_regularizer_loss = self.get_non_top_primitives_overlap_loss(
                pred_occ)
            moved_overlap_regularizer_loss = self.get_non_top_primitives_overlap_loss(
                primitive_moved_pred_occ)
        elif self.overlap_regularizer_type == 'nsd_style':
            logits = torch.sigmoid(pred_occ).sum(-1)
            overlap_regularizer_loss = torch.relu(
                logits - self.overlap_threshold).mean()
            logits = torch.sigmoid(primitive_moved_pred_occ).sum(-1)
            moved_overlap_regularizer_loss = torch.relu(
                logits - self.overlap_threshold).mean()
        elif self.overlap_regularizer_type == 'only_positive':
            overlap_regularizer_loss = (((torch.relu(
                self.implicit_discriminator_activation(pred_occ) -
                0.5).sum(-1) - self.overlap_threshold))**2).mean()

            moved_overlap_regularizer_loss = ((
                (torch.relu(
                    self.implicit_discriminator_activation(
                        primitive_moved_pred_occ) - 0.5).sum(-1) -
                 self.overlap_threshold))**2).mean()
        elif self.overlap_regularizer_type == 'raw_value':
            overlap_regularizer_loss = ((torch.relu(
                self.implicit_discriminator_activation(pred_occ).sum(-1) -
                self.overlap_threshold))**2).mean()

            moved_overlap_regularizer_loss = ((torch.relu(
                self.implicit_discriminator_activation(
                    primitive_moved_pred_occ).sum(-1) -
                self.overlap_threshold))**2).mean()
        else:
            raise NotImplementedError

        total_overlap_regularizer_loss = overlap_regularizer_loss

        if not self.disable_moved_overlap_regularizer_loss:
            total_overlap_regularizer_loss = total_overlap_regularizer_loss + moved_overlap_regularizer_loss

        total_overlap_regularizer_loss_weighted = total_overlap_regularizer_loss * self.overlap_regularizer_loss_weight

        losses = {}
        losses['overlap_regularizer_loss'] = overlap_regularizer_loss
        if not self.disable_moved_overlap_regularizer_loss:
            losses[
                'moved_overlap_regularizer_loss'] = moved_overlap_regularizer_loss
        losses[
            'total_overlap_regularizer_loss'] = total_overlap_regularizer_loss
        losses[
            'total_overlap_regularizer_loss_weighted'] = total_overlap_regularizer_loss_weighted

        return losses, total_overlap_regularizer_loss_weighted

    def get_regularizer_l1(self, pred_occ, primitive_moved_pred_occ):
        l1_occupancy_regularizer_loss = torch.sum(
            torch.relu(pred_occ)) + torch.sum(
                torch.relu(primitive_moved_pred_occ))

        losses = {}
        losses['l1_occupancy_regularizer_loss'] = l1_occupancy_regularizer_loss
        l1_occupancy_regularizer_loss_weighted = l1_occupancy_regularizer_loss * self.l1_occupancy_regularizer_loss_weight
        losses[
            'l1_occupancy_regularizer_loss_weighted'] = l1_occupancy_regularizer_loss_weighted

        return losses, l1_occupancy_regularizer_loss_weighted

    def train_explicit_losses(self,
                              step,
                              data,
                              skip_gp,
                              return_eval_loss=False):
        pointcloud_moving_points = data['pointcloud_moving_points'].to(
            self.device)
        slide = data['slide'].to(self.device).unsqueeze(-1)
        if self.is_temporal_pad_slide_dim:
            slide = torch.cat([torch.zeros_like(slide), slide], axis=-1)

        pointcloud_canonical_points = data['pointcloud_canonical_points'].to(
            self.device)

        pointcloud_moving_label = data['pointcloud_moving_label'].to(
            self.device).long()

        # Infererence for discrminator
        ret = self.model(pointcloud_moving_points)
        pred_points = ret['point_set']
        pred_slide = ret['param']
        # assuming primitive 1 is moving
        moved_pred_points = [points.clone() for points in pred_points]
        moved_pred_points[1][:, :, 1] -= pred_slide
        moved_pred_points_cat = torch.cat(moved_pred_points, axis=1)

        if return_eval_loss:
            losses = defaultdict(lambda: 0)
            batch_size = pointcloud_moving_points.size(0)
            for batch_idx in range(batch_size):
                for idx in range(1, self.primitive_num + 1):
                    pred = pred_points[idx - 1][batch_idx, :, :].unsqueeze(0)
                    gt = pointcloud_moving_points[
                        batch_idx, pointcloud_moving_label[
                            batch_idx, :] == idx, :].unsqueeze(0)
                    loss = chamfer_loss.chamfer_loss(pred, gt)
                    losses['part_loss_pm{}'.format(idx)] += loss / batch_size
            total_loss = 0.
            for key, value in losses.items():
                total_loss += value / self.primitive_num
            losses['part_loss'] = total_loss

            loss = F.mse_loss(pred_slide, slide)
            losses['slide_loss'] = loss
            yield losses
        # loss
        canonical_points_len = pointcloud_canonical_points.shape[1]
        moved_pred_points_cat_len = moved_pred_points_cat.shape[1]

        if canonical_points_len < moved_pred_points_cat_len:
            moved_sample_idx = random.sample(range(moved_pred_points_cat_len),
                                             canonical_points_len)
            moved_pred_points_cat = moved_pred_points_cat[:,
                                                          moved_sample_idx, :]
        else:
            sample_idx = random.sample(range(canonical_points_len),
                                       moved_pred_points_cat_len)
            pointcloud_canonical_points = pointcloud_canonical_points[:,
                                                                      sample_idx, :]

        ## Adversarial loss
        ### Discriminator
        D_real = self.model(pointcloud_canonical_points,
                            mode='discriminator')['D']
        D_real_loss = -D_real.mean()
        D_real_loss_weighted = D_real_loss * self.D_real_loss_weight * self.D_loss_weight

        D_fake = self.model(moved_pred_points_cat.detach(),
                            mode='discriminator')['D']
        D_fake_loss = D_fake.mean()
        D_fake_loss_weighted = D_fake_loss * self.D_fake_loss_weight * self.D_loss_weight

        #### Gradient penalty
        batch_size = pointcloud_canonical_points.size(0)

        alpha = torch.rand(batch_size, 1, 1,
                           requires_grad=True).to(self.device)
        # randomly mix real and fake data
        interpolates = pointcloud_canonical_points + alpha * (
            moved_pred_points_cat - pointcloud_canonical_points)
        # compute output of D for interpolated input
        disc_interpolates = self.model(interpolates, mode='discriminator')['D']
        # compute gradients w.r.t the interpolated outputs

        if skip_gp:
            gradient_penalty = torch.FloatTensor([0.]).to(self.device)
            gradient_penalty_weighted = torch.FloatTensor([0.]).to(self.device)
        else:
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

            gradient_penalty_weighted = gradient_penalty * self.gradient_pelnalty_weight * self.D_loss_weight

        total_D_loss = D_real_loss_weighted + D_fake_loss_weighted + gradient_penalty_weighted

        yield {
            'total_D_loss': total_D_loss,
            'D_real_loss': D_real_loss,
            'D_real_loss_weighted': D_real_loss_weighted,
            'D_fake_loss': D_fake_loss,
            'D_fake_loss_weighted': D_fake_loss_weighted,
            'gradient_penalty': gradient_penalty,
            'gradient_penalty_weighted': gradient_penalty_weighted
        }

        # inference for generator
        ret = self.model(pointcloud_moving_points)
        pred_points = ret['point_set']
        pred_slide = ret['param']
        # assuming primitive 1 is moving
        moved_pred_points = [points.clone() for points in pred_points]
        moved_pred_points[1][:, :, 1] -= pred_slide
        moved_pred_points_cat = torch.cat(moved_pred_points, axis=1)
        pred_points_cat = torch.cat(pred_points, axis=1)

        losses = {}
        ## recon loss
        recon_loss = chamfer_loss.chamfer_loss(pred_points_cat,
                                               pointcloud_moving_points)
        losses['recon_loss'] = recon_loss
        recon_loss_weighted = recon_loss * self.shape_recon_chamfer_loss_weight
        losses['recon_loss_weighted'] = recon_loss_weighted

        ### Generator
        moved_pred_points_cat_len = moved_pred_points_cat.shape[1]

        if canonical_points_len < moved_pred_points_cat_len:
            moved_pred_points_cat = moved_pred_points_cat[:,
                                                          moved_sample_idx, :]

        G_loss = -self.model(moved_pred_points_cat,
                             mode='discriminator')['D'].mean()
        losses['G_loss'] = G_loss
        G_loss_weighted = G_loss * self.G_loss_weight
        losses['G_loss_weighted'] = G_loss_weighted

        total_G_loss = G_loss_weighted + recon_loss_weighted

        if self.train_canonical:
            # inference for generator
            pointcloud_canonical_points = data[
                'pointcloud_canonical_points'].to(self.device)
            pointcloud_moving_points = data['pointcloud_moving_points'].to(
                self.device)

            ret = self.model(pointcloud_canonical_points, return_param=False)
            pred_points = ret['point_set']
            # assuming primitive 1 is moving
            moved_pred_points = [points.clone() for points in pred_points]
            random_slide = torch.ones_like(pred_slide).uniform_(
                1. / self.dataset_worldsize,
                (self.dataset_boxsize - 1.) / self.dataset_worldsize +
                np.finfo(np.float32).eps)
            moved_pred_points[1][:, :, 1] += random_slide
            moved_pred_points_cat = torch.cat(moved_pred_points, axis=1)
            pred_points_cat = torch.cat(pred_points, axis=1)

            ## recon loss
            if self.use_canonical_recon_loss:
                recon_loss = chamfer_loss.chamfer_loss(
                    pred_points_cat, pointcloud_canonical_points)
                losses['canonical_recon_loss'] = recon_loss
                recon_loss_weighted = recon_loss * self.shape_recon_chamfer_loss_weight
                losses['canonical_recon_loss_weighted'] = recon_loss_weighted
                total_G_loss += recon_loss_weighted

            if self.use_canonical_self_supervised_slide_loss:
                ret = self.model(moved_pred_points_cat, return_points=False)
                pred_slide = ret['param']
                self_supervised_slide_loss = F.mse_loss(
                    pred_slide, random_slide)
                self_supervised_slide_loss_weighted = self_supervised_slide_loss * self.self_supervised_slide_loss_weight
                losses[
                    'self_supervised_slide_loss'] = self_supervised_slide_loss
                losses[
                    'self_supervised_slide_loss_weighted'] = self_supervised_slide_loss_weighted
                total_G_loss += self_supervised_slide_loss_weighted
            losses['total_G_loss'] = total_G_loss

        yield losses

    def train_implicit_losses(self,
                              data,
                              step,
                              skip_gp,
                              return_eval_loss=False):
        pointcloud_moving_points = data['pointcloud_moving_points'].to(
            self.device)
        slide = data['slide'].to(self.device).unsqueeze(-1)
        if self.is_temporal_pad_slide_dim:
            slide = torch.cat([torch.zeros_like(slide), slide], axis=-1)
        pointcloud_canonical_points = data['pointcloud_canonical_points'].to(
            self.device)

        if self.is_movenet_decode_sdf:
            points_moving_points = data['sdf_points_moving_points'].to(
                self.device)
            points_moving_label = data['sdf_points_moving_label'].to(
                self.device).float()
            points_canonical_points = data['sdf_points_canonical_points'].to(
                self.device)
            points_canonical_label = data['sdf_points_canonical_label'].to(
                self.device).float()
        else:
            points_moving_points = data['points_moving_points'].to(self.device)
            points_moving_label = data['points_moving_label'].to(
                self.device).float()
            points_canonical_points = data['points_canonical_points'].to(
                self.device)
            points_canonical_label = data['points_canonical_label'].to(
                self.device).float()

        if return_eval_loss:
            eval_losses = self.compute_eval_losses(slide,
                                                   points_moving_label,
                                                   points_moving_points,
                                                   pointcloud_moving_points,
                                                   is_A_canonical=True)
            if self.use_acgan:
                canonical_eval_losses = self.compute_eval_losses(
                    torch.zeros_like(slide),
                    points_canonical_label,
                    points_canonical_points,
                    pointcloud_canonical_points,
                    is_A_canonical=False)
                eval_losses.update(canonical_eval_losses)
            yield eval_losses

        # loss
        ## Adversarial loss
        ### Discriminator
        disc_losses = self.train_implicit_discriminator(
            slide,
            points_canonical_label,
            points_canonical_points,
            points_moving_label,
            points_moving_points,
            pointcloud_moving_points,
            skip_gp=skip_gp,
            is_A_canonical=True)
        if self.use_acgan:
            canonical_disc_losses = self.train_implicit_discriminator(
                torch.zeros_like(slide),
                points_moving_label,
                points_moving_points,
                points_canonical_label,
                points_canonical_points,
                pointcloud_canonical_points,
                skip_gp=skip_gp,
                is_A_canonical=False)
            disc_losses.update(canonical_disc_losses)
            disc_losses['total_D_loss'] += disc_losses[
                'canonical_total_D_loss']
        yield disc_losses

        gen_losses = {}
        moving_gen_losses = self.train_implicit_generator(
            points_moving_label,
            points_moving_points,
            pointcloud_moving_points,
            learn_occ_recon=True,
            learn_generator=True,
            overlap_reg=self.use_overlap_regularizer,
            L1_reg=self.use_l1_occupancy_regularizer,
            learn_self_slide_reg=self.learn_moving_self_slide_reg,
            is_B_moving=True)
        gen_losses.update(moving_gen_losses)

        if self.use_acgan:
            canonical_learn_generator_flag = True
            canonical_overlap_reg_flag = self.use_overlap_regularizer
            canonical_L1_reg_flag = self.use_l1_occupancy_regularizer
            canonical_learn_occ_recon_flag = True
        else:
            canonical_learn_generator_flag = False
            canonical_overlap_reg_flag = False
            canonical_L1_reg_flag = False
            canonical_learn_occ_recon_flag = self.use_canonical_occ_loss

        canonical_learn_self_slide_reg_flag = self.use_canonical_self_supervised_slide_loss

        canonical_gen_losses = self.train_implicit_generator(
            points_canonical_label,
            points_canonical_points,
            pointcloud_canonical_points,
            learn_occ_recon=canonical_learn_occ_recon_flag,
            learn_generator=canonical_learn_generator_flag,
            overlap_reg=canonical_overlap_reg_flag,
            L1_reg=canonical_L1_reg_flag,
            learn_self_slide_reg=canonical_learn_self_slide_reg_flag,
            is_B_moving=False)
        gen_losses.update(canonical_gen_losses)
        gen_losses['total_G_loss'] += gen_losses['canonical_total_G_loss']
        yield gen_losses

    def compute_eval_losses(self,
                            slide_A,
                            label_B,
                            points_B,
                            pointcloud_B,
                            is_A_canonical=True):
        with torch.no_grad():
            ret = self.model(pointcloud_B, points_B)
        if self.disable_induction_by_moving:
            pred_slide = None
        else:
            if is_A_canonical:
                pred_slide = ret['param']
            else:
                pred_slide = -torch.ones_like(ret['param']).uniform_(
                    1. / self.dataset_worldsize,
                    (self.dataset_boxsize - 1.) / self.dataset_worldsize +
                    np.finfo(np.float32).eps)

        pred_occ = ret['occupancy']
        discriminator_ret = self.get_discriminator_result(
            pred_occ,
            points_B,
            pred_slide,
            pointcloud_B,
            no_move=self.disable_induction_by_moving,
            detach=True)
        pred_occ_after_move = discriminator_ret['pred_occ_after_move']

        eval_losses = self.get_eval_losses(pred_occ, label_B,
                                           pred_occ_after_move, pred_slide,
                                           slide_A)
        return eval_losses

    def train_implicit_discriminator(self,
                                     slide_A,
                                     label_A,
                                     points_A,
                                     label_B,
                                     points_B,
                                     pointcloud_B,
                                     skip_gp=False,
                                     is_A_canonical=True):
        """
        A = canonical
        B = moving
        """

        disc_losses = {}
        total_D_loss = 0.  #torch.zeros_like(points_A.sum())
        with torch.no_grad():
            ret = self.model(pointcloud_B, points_B)
        if is_A_canonical:
            pred_slide = ret['param']
        else:
            pred_slide = -torch.ones_like(ret['param']).uniform_(
                1. / self.dataset_worldsize,
                (self.dataset_boxsize - 1.) / self.dataset_worldsize +
                np.finfo(np.float32).eps)

        pred_occ = ret['occupancy']

        discriminator_ret = self.get_discriminator_result(
            pred_occ,
            points_B,
            pred_slide,
            pointcloud_B,
            no_move=self.disable_induction_by_moving,
            detach=True)

        if self.use_part_shape_similarity_loss:
            part_shape_similarity_loss_ret, part_shape_similarity_loss_weighted = self.get_part_shape_similarity_loss(
                discriminator_ret['points_value'], pred_occ)
            total_D_loss += part_shape_similarity_loss_weighted
            part_shape_similarity_loss_ret = {
                'D_' + name: value
                for name, value in part_shape_similarity_loss_ret.items()
            }
            disc_losses.update(part_shape_similarity_loss_ret)

        if not self.disable_gan_training:
            discriminator_real_ret = self.get_real_discriminator_result(
                label_A, points_A)
            points_value_B = discriminator_real_ret['points_value']
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

            disc_losses.update({
                'D_real_loss': D_real_loss,
                'D_real_loss_weighted': D_real_loss_weighted,
            })

            D_fake = discriminator_ret['D']

            if self.gan_type == 'lsgan':
                D_fake_loss = F.mse_loss(D_fake,
                                         torch.zeros_like(D_fake),
                                         reduction='mean')
            elif self.gan_type == 'wgangp':
                D_fake_loss = D_fake.mean()
            else:
                raise NotImplementedError
            D_fake_loss_weighted = D_fake_loss * self.D_fake_loss_weight * self.D_loss_weight

            total_D_loss += D_real_loss_weighted + D_fake_loss_weighted

            disc_losses.update({
                'D_fake_loss': D_fake_loss,
                'D_fake_loss_weighted': D_fake_loss_weighted
            })
            #### Gradient penalty

            if self.gan_type == 'wgangp':
                batch_size = points_value_B.size(0)

                alpha = torch.rand(batch_size, 1, 1,
                                   requires_grad=True).to(self.device)
                # randomly mix real and fake data
                if self.use_wgangp_latent_code_interpolation:
                    real_latent = discriminator_real_ret['D_latent']
                    fake_latent = discriminator_ret['D_latent']
                    interpolates = real_latent + alpha * (fake_latent -
                                                          real_latent)
                    interpolated_mask = None
                else:
                    points_value_B = discriminator_real_ret['points_value']
                    moved_pred_points_value = discriminator_ret['points_value']
                    interpolates = points_value_B + alpha * (
                        moved_pred_points_value - points_value_B)
                    # compute output of D for interpolated input

                    interpolated_mask = None
                    if self.use_discriminator_input_mask:
                        if self.is_movenet_classification_head:
                            raise NotImplementedError
                        elif self.is_movenet_decode_sdf:
                            raise NotImplementedError
                        else:
                            pred_mask = discriminator_ret['mask']
                            canonical_mask = discriminator_real_ret['mask']
                            mask_filter = torch.bernoulli(
                                torch.ones_like(pred_mask) * alpha.squeeze(-1))
                            interpolated_mask = pred_mask * mask_filter + canonical_mask * -(
                                mask_filter - 1)

                disc_interpolates = self.model(
                    interpolates,
                    mask=interpolated_mask,
                    direct_input_to_D=self.
                    use_wgangp_latent_code_interpolation,
                    mode='discriminator')['D']
                # compute gradients w.r.t the interpolated outputs

                if skip_gp:
                    gradient_penalty = torch.FloatTensor([0.]).to(self.device)
                    gradient_penalty_weighted = torch.FloatTensor([0.]).to(
                        self.device)
                else:
                    gradients = torch.autograd.grad(
                        outputs=disc_interpolates,
                        inputs=interpolates,
                        grad_outputs=torch.ones(disc_interpolates.size()).to(
                            self.device),
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True)[0].contiguous().view(batch_size, -1)

                    gradient_penalty = (
                        ((gradients.norm(2, dim=1) - self.gp_gamma) /
                         self.gp_gamma)**2).mean() * self.gp_lambda

                    gradient_penalty_weighted = gradient_penalty * self.gradient_pelnalty_weight * self.D_loss_weight

                total_D_loss += gradient_penalty_weighted

                disc_losses.update({
                    'gradient_penalty':
                    gradient_penalty,
                    'gradient_penalty_weighted':
                    gradient_penalty_weighted
                })

            if self.use_acgan:
                if self.learn_D_slide_loss:
                    if is_A_canonical:  # disc_real_ret is result of canonical
                        D_real_slide = discriminator_real_ret['D_slide']
                        D_real_slide_loss = F.mse_loss(
                            D_real_slide, torch.zeros_like(D_real_slide))
                        D_real_slide_loss_weighted = D_real_slide_loss * self.D_slide_loss_weight
                        total_D_loss += D_real_slide_loss_weighted
                        disc_losses.update({
                            'D_real_slide_loss':
                            D_real_slide_loss,
                            'D_real_slide_loss_weighted':
                            D_real_slide_loss_weighted
                        })

                        D_fake_slide = discriminator_ret['D_slide']
                        D_fake_slide_loss = F.mse_loss(
                            D_fake_slide, torch.zeros_like(D_fake_slide))
                        D_fake_slide_loss_weighted = D_fake_slide_loss * self.D_slide_loss_weight
                        total_D_loss += D_fake_slide_loss_weighted
                        disc_losses.update({
                            'D_fake_slide_loss':
                            D_fake_slide_loss,
                            'D_fake_slide_loss_weighted':
                            D_fake_slide_loss_weighted
                        })
                    else:
                        D_fake_slide = discriminator_ret['D_slide']
                        D_fake_slide_loss = F.mse_loss(D_fake_slide,
                                                       -pred_slide)
                        D_fake_slide_loss_weighted = D_fake_slide_loss * self.D_slide_loss_weight
                        total_D_loss += D_fake_slide_loss_weighted
                        disc_losses.update({
                            'D_fake_slide_loss':
                            D_fake_slide_loss,
                            'D_fake_slide_loss_weighted':
                            D_fake_slide_loss_weighted
                        })
                if self.learn_D_canonical_loss:
                    if is_A_canonical:  # disc_real_ret is result of canonical
                        D_real_canonical = discriminator_real_ret[
                            'D_canonical']
                        D_real_canonical_loss = F.mse_loss(
                            D_real_canonical,
                            torch.ones_like(D_real_canonical))
                        D_real_canonical_loss_weighted = D_real_canonical_loss * self.D_canonical_loss_weight
                        total_D_loss += D_real_canonical_loss_weighted
                        disc_losses.update({
                            'D_real_canonical_loss':
                            D_real_canonical_loss,
                            'D_real_canonical_loss_weighted':
                            D_real_canonical_loss_weighted
                        })

                        D_fake_canonical = discriminator_ret['D_canonical']
                        D_fake_canonical_loss = F.mse_loss(
                            D_fake_canonical,
                            torch.ones_like(D_fake_canonical))
                        D_fake_canonical_loss_weighted = D_fake_canonical_loss * self.D_canonical_loss_weight
                        total_D_loss += D_fake_canonical_loss_weighted
                        disc_losses.update({
                            'D_fake_canonical_loss':
                            D_fake_canonical_loss,
                            'D_fake_canonical_loss_weighted':
                            D_fake_canonical_loss_weighted
                        })
                    else:
                        D_real_canonical = discriminator_real_ret[
                            'D_canonical']
                        D_real_canonical_loss = F.mse_loss(
                            D_real_canonical,
                            torch.zeros_like(D_real_canonical))
                        D_real_canonical_loss_weighted = D_real_canonical_loss * self.D_canonical_loss_weight
                        total_D_loss += D_real_canonical_loss_weighted
                        disc_losses.update({
                            'D_real_canonical_loss':
                            D_real_canonical_loss,
                            'D_real_canonical_loss_weighted':
                            D_real_canonical_loss_weighted
                        })
                        D_fake_canonical = discriminator_ret['D_canonical']
                        D_fake_canonical_loss = F.mse_loss(
                            D_fake_canonical,
                            torch.zeros_like(D_fake_canonical))
                        D_fake_canonical_loss_weighted = D_fake_canonical_loss * self.D_canonical_loss_weight
                        total_D_loss += D_fake_canonical_loss_weighted
                        disc_losses.update({
                            'D_fake_canonical_loss':
                            D_fake_canonical_loss,
                            'D_fake_canonical_loss_weighted':
                            D_fake_canonical_loss_weighted
                        })
        if not isinstance(total_D_loss, float):
            disc_losses['total_D_loss'] = total_D_loss
        prefix = 'canonical_' if not is_A_canonical else ''
        prefixed_disc_losses = {(prefix + name): value
                                for name, value in disc_losses.items()}
        return prefixed_disc_losses

    def get_eval_losses(self, pred_occ, label_B, pred_occ_after_move,
                        pred_slide, slide_A):
        disc_losses = {}
        if not self.disable_part_seg_loss_iou_for_eval:
            part_seg_iou_loss_ret = self.get_iou_loss_for_eval(
                pred_occ, label_B)
            disc_losses.update(part_seg_iou_loss_ret)

        iou_loss_ret = self.get_whole_iou_loss_for_eval(pred_occ, label_B)
        disc_losses.update(iou_loss_ret)

        pred_occ_after_move_loss_ret = self.get_overlap_loss_for_eval(
            pred_occ_after_move, pred_occ)
        disc_losses.update(pred_occ_after_move_loss_ret)
        """
        if not self.disable_induction_by_moving:
            param_loss_ret = self.get_param_loss_for_eval(pred_slide, slide_A)
            disc_losses.update(param_loss_ret)
        """

        if not self.disable_part_seg_loss_iou_for_eval and not self.disable_induction_by_moving and 'part_loss_iou' in part_seg_iou_loss_ret and 'slide_loss' in param_loss_ret:
            part_iou_vs_slide_envelope_loss_ret = self.get_part_iou_vs_slide_envelope_loss_for_eval(
                part_seg_iou_loss_ret['part_loss_iou'],
                param_loss_ret['slide_loss'])
            disc_losses.update(part_iou_vs_slide_envelope_loss_ret)

        return disc_losses

    def train_implicit_generator(self,
                                 label_B,
                                 points_B,
                                 pointcloud_B,
                                 learn_occ_recon=True,
                                 learn_generator=False,
                                 L1_reg=False,
                                 overlap_reg=False,
                                 learn_self_slide_reg=False,
                                 is_B_moving=True):
        """
        A = canonical
        B = moving
        """

        losses = {}
        total_G_loss = 0.
        if learn_occ_recon:
            # inference for generator
            ret = self.model(pointcloud_B, points_B)
            pred_occ = ret['occupancy']

            pred_occ_max_logit = self.get_indicator_value(pred_occ)

            ## recon loss

            reconstruction_losses, occ_loss_weighted = self.get_implicit_reconstruction_loss(
                pred_occ, label_B)
            losses.update(reconstruction_losses)

            total_G_loss = total_G_loss + occ_loss_weighted

            ## generator loss
            if not self.disable_induction_by_moving:
                if is_B_moving:
                    pred_slide = ret['param']
                else:
                    pred_slide = -torch.ones_like(ret['param']).uniform_(
                        1. / self.dataset_worldsize,
                        (self.dataset_boxsize - 1.) / self.dataset_worldsize +
                        np.finfo(np.float32).eps)
            else:
                pred_slide = None

        if learn_generator or overlap_reg or self.use_part_shape_similarity_loss or self.use_acgan:
            discriminator_ret = self.get_discriminator_result(
                pred_occ,
                points_B,
                pred_slide,
                pointcloud_B,
                no_move=self.disable_induction_by_moving,
                detach_param=self.is_detach_param_in_generator)

        if learn_generator and not self.disable_gan_training:
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
            G_loss_weighted = G_loss * self.G_loss_weight
            losses['G_loss_weighted'] = G_loss_weighted

            total_G_loss = total_G_loss + G_loss_weighted

        if self.use_part_shape_similarity_loss:
            part_shape_similarity_loss_ret, part_shape_similarity_loss_weighted = self.get_part_shape_similarity_loss(
                discriminator_ret['points_value'], pred_occ)
            part_shape_similarity_loss_ret = {
                'G_' + name: value
                for name, value in part_shape_similarity_loss_ret.items()
            }
            total_G_loss += part_shape_similarity_loss_weighted
            losses.update(part_shape_similarity_loss_ret)

        if self.use_entropy_reduction_loss and is_B_moving:
            entropy_reduction_loss_ret, entropy_reduction_loss_weighted = self.get_shape_entropy_loss(
                pred_occ, points_B)
            total_G_loss += entropy_reduction_loss_weighted
            losses.update(entropy_reduction_loss_ret)

        if overlap_reg:
            primitive_moved_pred_occ = discriminator_ret['pred_occ_after_move']
            overlap_regularizer_losses, total_overlap_regularizer_loss_weighted = self.get_regularizer_overlap(
                pred_occ, primitive_moved_pred_occ)
            losses.update(overlap_regularizer_losses)
            total_G_loss += total_overlap_regularizer_loss_weighted

        if L1_reg:
            l1_regularizer_losses, l1_occupancy_regularizer_loss_weighted = self.get_regularizer_l1(
                pred_occ, primitive_moved_pred_occ)
            losses.update(l1_regularizer_losses)
            total_G_loss += l1_occupancy_regularizer_loss_weighted

        if learn_self_slide_reg and not self.disable_induction_by_moving:
            ret = self.model(pointcloud_B, pointcloud_B, return_param=True)
            random_slide = torch.ones_like(ret['param'].detach()).uniform_(
                1. / self.dataset_worldsize,
                (self.dataset_boxsize - 1.) / self.dataset_worldsize +
                np.finfo(np.float32).eps)
            if is_B_moving:
                random_slide -= ret['param']

            moved_pointcloud_canonical_points = pointcloud_B.clone()
            pred_occ_argmax = ret['occupancy'].argmax(-1)
            if self.is_temporal_pad_slide_dim:
                for moving_idx in range(self.primitive_num):
                    moved_pointcloud_canonical_points[:, :,
                                                      1] = moved_pointcloud_canonical_points[:, :, 1] + (
                                                          pred_occ_argmax
                                                          == moving_idx
                                                      ) * random_slide[:,
                                                                       moving_idx].unsqueeze(
                                                                           -1)
            else:
                moving_idx = self.get_moving_idx()
                moved_pointcloud_canonical_points[:, :,
                                                  1] = moved_pointcloud_canonical_points[:, :, 1] + (
                                                      pred_occ_argmax
                                                      == moving_idx
                                                  ) * random_slide
            ret2 = self.model(moved_pointcloud_canonical_points,
                              None,
                              return_occupancy=False)
            if not is_B_moving:
                self_supervised_slide_loss = F.mse_loss(
                    ret2['param'], random_slide)
            else:
                self_supervised_slide_loss = F.mse_loss(
                    ret2['param'], random_slide + ret['param'])
            self_supervised_slide_loss_weighted = self_supervised_slide_loss * self.self_supervised_slide_loss_weight
            losses['self_supervised_slide_loss'] = self_supervised_slide_loss
            losses[
                'self_supervised_slide_loss_weighted'] = self_supervised_slide_loss_weighted
            total_G_loss += self_supervised_slide_loss_weighted

        if self.use_acgan and not self.disable_gan_training:
            if self.learn_D_slide_loss:
                if is_B_moving:  # disc_real_ret is result of canonical
                    G_fake_slide = discriminator_ret['D_slide']
                    G_fake_slide_loss = F.mse_loss(
                        G_fake_slide, torch.zeros_like(G_fake_slide))
                    G_fake_slide_loss_weighted = G_fake_slide_loss * self.G_slide_loss_weight
                    total_G_loss += G_fake_slide_loss_weighted
                    losses.update({
                        'G_fake_slide_loss':
                        G_fake_slide_loss,
                        'G_fake_slide_loss_weighted':
                        G_fake_slide_loss_weighted
                    })
                else:
                    G_fake_slide = discriminator_ret['D_slide']
                    G_fake_slide_loss = F.mse_loss(G_fake_slide, -pred_slide)
                    G_fake_slide_loss_weighted = G_fake_slide_loss * self.G_slide_loss_weight
                    total_G_loss += G_fake_slide_loss_weighted
                    losses.update({
                        'G_fake_slide_loss':
                        G_fake_slide_loss,
                        'G_fake_slide_loss_weighted':
                        G_fake_slide_loss_weighted
                    })
            if self.learn_D_canonical_loss:
                if is_B_moving:  # disc_real_ret is result of canonical
                    G_fake_canonical = discriminator_ret['D_canonical']
                    G_fake_canonical_loss = F.mse_loss(
                        G_fake_canonical, torch.ones_like(G_fake_canonical))
                    G_fake_canonical_loss_weighted = G_fake_canonical_loss * self.G_canonical_loss_weight
                    total_G_loss += G_fake_canonical_loss_weighted
                    losses.update({
                        'G_fake_canonical_loss':
                        G_fake_canonical_loss,
                        'G_fake_canonical_loss_weighted':
                        G_fake_canonical_loss_weighted
                    })
                else:
                    G_fake_canonical = discriminator_ret['D_canonical']
                    G_fake_canonical_loss = F.mse_loss(
                        G_fake_canonical, torch.zeros_like(G_fake_canonical))
                    G_fake_canonical_loss_weighted = G_fake_canonical_loss * self.G_canonical_loss_weight
                    total_G_loss += G_fake_canonical_loss_weighted
                    losses.update({
                        'G_fake_canonical_loss':
                        G_fake_canonical_loss,
                        'G_fake_canonical_loss_weighted':
                        G_fake_canonical_loss_weighted
                    })

        losses['total_G_loss'] = total_G_loss

        prefix = 'canonical_' if not is_B_moving else ''
        prefixed_losses = {(prefix + name): value
                           for name, value in losses.items()}
        return prefixed_losses

    def get_part_shape_similarity_loss(self, points_value, occ):
        labels = []
        embeddings = []
        rg = (
            self.primitive_num + 1
        ) if self.use_entire_shape_as_a_part_in_similarity_loss else self.primitive_num
        for idx in range(rg):
            if idx == self.primitive_num:
                occ_primitive, _ = occ.max(-1)
            else:
                occ_primitive = occ[:, :, idx]
            pred_mask = self.get_discriminator_input_mask(occ_primitive)
            is_background = ((occ_primitive >= 0).sum(axis=1) == 0)
            label = torch.where(
                is_background, torch.zeros_like(is_background.long()),
                torch.ones_like(is_background.long()) * (idx + 1))
            labels.append(label)

            ret = self.model(points_value,
                             mask=pred_mask,
                             mode='occupancy_points_encoder')
            embeddings.append(ret['latent_normalized'])

        labels = torch.cat(labels, axis=0).to(self.device)
        embeddings = torch.cat(embeddings, axis=0).to(self.device)

        if self.use_part_shape_similarity_loss_hard_miner:
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

        return ret, loss_weighted

    def get_shape_entropy_loss(self, occ, points, gt_occ=None):
        # occ = B, P, primitive_num
        # points = B, P, dim
        pred_occ_max, pred_occ_argmax = occ.max(axis=-1, keepdim=True)

        # B, P, 1
        th = 0.5 if self.occupancy_reduction_loss_type in self.imnet_occupancy_reduction_loss_types else 0.
        points_label = torch.where(pred_occ_max >= th, pred_occ_argmax + 1,
                                   torch.zeros_like(pred_occ_max).long())
        if gt_occ is not None:
            points_label_binary = gt_occ.clamp(max=1).unsqueeze(-1)
        else:
            points_label_binary = points_label.clamp(max=1)

        ret = {}
        if self.entropy_reduction_loss_type in ['obb', 'convex']:
            shape_points = []
            batch_size = pred_occ_max.size(0)
            points_nps = np.split(points.cpu().detach().numpy(), batch_size)
            points_label_nps = np.split(points_label.cpu().detach().numpy(),
                                        batch_size)
            points_binary_label_nps = np.split(
                points_label_binary.cpu().detach().numpy(), batch_size)

            whole_shape_non_zero = []
            pm_non_zero = []
            for points_np, points_label_np, points_binary_label_np in zip(
                    points_nps, points_label_nps, points_binary_label_nps):
                points_np = points_np[0]
                points_label_np = points_label_np[0, :, 0]
                points_binary_label_np = points_binary_label_np[0, :, 0]
                whole_shape_np = points_np[points_binary_label_np == 1, :]
                whole_shape_non_zero.append(len(whole_shape_np) > 4)
                shape_points.append(whole_shape_np)
                pm_shape_nps = [
                    points_np[points_label_np == idx, :]
                    for idx in range(1, self.primitive_num + 1)
                ]
                pm_non_zero.append([len(pm) >= 4 for pm in pm_shape_nps])
                shape_points.extend(pm_shape_nps)
            #print([np.unique(e) for e in points_label_nps])
            #print(pm_non_zero)
            mask = geometry.get_primitive_mask(
                shape_points,
                points,
                self.entropy_reduction_loss_primitive_executor,
                self.primitive_num + 1,
                primitive_type=self.entropy_reduction_loss_type)
            whole_shape_mask = mask[:, 0, :]

            weight = torch.from_numpy(np.array(whole_shape_non_zero)).type(
                pred_occ_max.dtype).unsqueeze(-1).unsqueeze(-1).to(
                    pred_occ_max.device)
            whole_entropy = self.get_occupancy_loss(
                pred_occ_max,
                whole_shape_mask.unsqueeze(-1).type(occ.dtype),
                weight=weight).detach()

            pm_shape_masks = mask[:, 1:, :]
            primitive_entropy_sum = 0.
            each_entropy_reduction_loss = 0.
            pm_weights = torch.from_numpy(np.array(pm_non_zero)).type(
                pred_occ_max.dtype).unsqueeze(-1).to(
                    #pred_occ_max.dtype).unsqueeze(-1).unsqueeze(-1).to(
                    pred_occ_max.device)
            valid_pm_num = pm_weights.sum().float() / batch_size
            ret.update({'num_of_valid_pm': valid_pm_num})
            for idx in range(self.primitive_num):
                pm_weight = pm_weights[:, idx, :]
                entropy = self.get_occupancy_loss(
                    occ[:, :, idx],
                    pm_shape_masks[:, idx, :].type(occ.dtype),
                    weight=pm_weight)
                primitive_entropy_sum += entropy

                each_entropy_reduction_loss += torch.relu(
                    entropy - whole_entropy -
                    self.each_entropy_reduction_loss_margin).mean(
                    ) / self.primitive_num

        else:
            min_bg = (points_label_binary != 1).float() * 1000
            max_bg = -(points_label_binary != 1).float() * 1000
            topleft, _ = (points + min_bg).min(1, keepdim=True)
            rightbottom, _ = (points + max_bg).max(1, keepdim=True)

            mask_in_bbox = (topleft <= points).all(-1, keepdim=True) & (
                points <= rightbottom).all(-1, keepdim=True)

            bce = F.binary_cross_entropy_with_logits(
                pred_occ_max, torch.ones_like(pred_occ_max), reduction='none')
            whole_entropy = (bce * mask_in_bbox).squeeze(1).mean(1)

            primitive_entropy_sum = 0.
            each_entropy_reduction_loss = 0.
            for idx in range(1, self.primitive_num + 1):
                min_bg = (points_label != idx).float() * 1000
                max_bg = -(points_label != idx).float() * 1000
                topleft, _ = (points + min_bg).min(1, keepdim=True)
                rightbottom, _ = (points + max_bg).max(1, keepdim=True)

                mask_in_bbox = (topleft <= points).all(-1, keepdim=True) & (
                    points <= rightbottom).all(-1, keepdim=True)
                # B
                #entropy = -(mask_in_bbox * pred_occ_max *
                #            pred_occ_max.clamp(min=EPS).log()).squeeze(-1).sum(1)
                entropy = (bce * mask_in_bbox).squeeze(1).mean(1)
                primitive_entropy_sum += entropy

                each_entropy_reduction_loss += torch.relu(
                    entropy - whole_entropy -
                    self.each_entropy_reduction_loss_margin).mean(
                    ) / self.primitive_num

        # magin > 0 allows primitive decomp gets more entropy
        # magin < 0 forces primitive decomp gets less entropy
        whole_entropy_reduction_loss = torch.relu(
            primitive_entropy_sum - whole_entropy -
            self.whole_entropy_reduction_loss_margin).mean()
        #print('pes', primitive_entropy_sum)
        #print('we', whole_entropy)

        entropy_reduction_loss = each_entropy_reduction_loss + whole_entropy_reduction_loss
        entropy_reduction_loss_weighted = entropy_reduction_loss * self.entropy_reduction_loss_weight

        ret.update({
            'each_entropy_reduction_loss':
            each_entropy_reduction_loss,
            'whole_entropy_reduction_loss':
            whole_entropy_reduction_loss,
            'entropy_reduction_loss':
            entropy_reduction_loss,
            'entropy_reduction_loss_weighted':
            entropy_reduction_loss_weighted,
        })
        return ret, entropy_reduction_loss_weighted

    def get_whole_iou_loss_for_eval(self, pred_occ, label, prefix=''):
        if self.is_movenet_classification_head:
            pred_occ_label = torch.argmax(pred_occ, dim=-1)
        elif self.is_movenet_decode_sdf:
            pred_occ_max_vis, pred_occ_argmax_vis = pred_occ.min(axis=-1)
            pred_occ_label = torch.where(
                pred_occ_max_vis <= 0, pred_occ_argmax_vis + 1,
                torch.zeros_like(pred_occ_max_vis).long())
        else:
            if self.occupancy_reduction_loss_type in self.imnet_occupancy_reduction_loss_types:
                pred_occ_max_vis, pred_occ_argmax_vis = pred_occ.max(axis=-1)
            elif self.occupancy_reduction_loss_type == 'occnet':
                pred_occ_max_vis, pred_occ_argmax_vis = torch.sigmoid(
                    pred_occ).max(axis=-1)
            else:
                raise NotImplementedError
            pred_occ_label = torch.where(
                pred_occ_max_vis >= self.visualize_isosurface_threshold,
                pred_occ_argmax_vis + 1,
                torch.zeros_like(pred_occ_max_vis).long())

        occ1 = pred_occ_label.clamp(max=1)
        if self.is_movenet_decode_sdf:
            occ2min, occ2argmin = label.min(-1)
            occ2 = torch.where(occ2min <= 0, occ2argmin + 1,
                               torch.zeros_like(occ2min).long())
        else:
            occ2 = label

        occ2 = occ2.clamp(max=1)
        # Convert to boolean values
        occ1 = (occ1 >= self.evaluate_isosurface_threshold)
        occ1 = occ1.squeeze(-1)
        occ2 = (occ2 >= self.evaluate_isosurface_threshold)
        occ2 = occ2.squeeze(-1)

        # Compute IOU
        area_union = (occ1 | occ2).float().sum(axis=-1)
        area_intersect = (occ1 & occ2).sum(axis=-1)

        loss = (area_intersect / area_union).mean()

        losses = {}
        losses['iou_loss'] = loss
        prefix = (prefix + '_') if prefix != '' else prefix
        losses = {prefix + name: value for name, value in losses.items()}
        return losses

    def get_occupancy_loss(self,
                           logit,
                           label,
                           weight=None,
                           is_reduction=True,
                           disable_logits_bce=False):
        if self.occupancy_reduction_loss_type == 'occnet':
            if self.use_imnet_style_occupancy_loss_with_logit:
                logit = torch.max(torch.min(logit, logit * 0.01 + 0.99),
                                  logit * 0.01)
                l2diff = (logit - label)**2
                if weight is not None:
                    l2diff = l2diff * weight
                if is_reduction:
                    occ_loss = l2diff.mean()
                else:
                    occ_loss = l2diff

            elif self.use_tsdf_in_occupancy_loss:
                if weight is not None:
                    occ_loss = (F.l1_loss(logit, label, reduction='none') *
                                weight)
                    if is_reduction:
                        occ_loss = occ_loss.mean()
                else:
                    if is_reduction:
                        occ_loss = F.l1_loss(logit, label)
                    else:
                        occ_loss = F.l1_loss(logit, label, reduction='none')
            elif self.use_mse_in_occupancy_loss:
                if weight is not None:
                    raise NotImplementedError
                else:
                    if is_reduction:
                        occ_loss = F.mse_loss(logit, label)
                    else:
                        occ_loss = F.mse_loss(logit, label, reduction='none')
            else:
                if self.use_bce_without_logits or disable_logits_bce:
                    bce = F.binary_cross_entropy
                else:
                    bce = F.binary_cross_entropy_with_logits
                if is_reduction:
                    occ_loss = bce(logit,
                                   label,
                                   weight=weight,
                                   reduction='none').sum(-1).mean()
                else:
                    occ_loss = bce(logit,
                                   label,
                                   weight=weight,
                                   reduction='none')
            return occ_loss
        elif self.occupancy_reduction_loss_type in self.imnet_occupancy_reduction_loss_types:
            assert is_reduction
            l2diff = (logit - label)**2
            if weight is not None:
                l2diff = l2diff * weight
            if self.occupancy_reduction_loss_type == 'imnet_ignore_gt_one':
                l2diff = l2diff * ((logit <= 1) | (label < 1))
            occ_loss = torch.mean(l2diff)
            return occ_loss
        else:
            raise NotImplementedError

    def post_loss_eval(self, post_loss_dict):
        if 'post_part_loss_iou' in post_loss_dict:
            ret, match = self.get_post_part_loss_iou_for_eval(
                post_loss_dict['post_part_loss_iou'])
            del post_loss_dict['post_part_loss_iou']
        else:
            ret = {}
        for loss_name, losses in post_loss_dict.items():
            if loss_name == 'post_param_loss':
                tmp_ret = self.get_post_param_loss(losses, match)
                ret.update(tmp_ret)
            else:
                raise NotImplementedError
        return ret

    def get_post_part_loss_iou_for_eval(self, losses):
        gt_labels_num = -1
        gt_has_background = False
        for loss_dict in losses:
            gt_labels = loss_dict['gt_labels']
            uniques = np.unique(gt_labels)
            if uniques.min() == 0:
                gt_has_background = True
            num = len(uniques)
            if num > gt_labels_num:
                gt_labels_num = num

        gt_pm_num = (gt_labels_num - 1) if gt_has_background else gt_labels_num
        assert self.primitive_num >= gt_pm_num

        cost = np.zeros([self.primitive_num, gt_pm_num])
        sample_num = len(losses)
        for loss_dict in losses:
            pred_labels = loss_dict['pred_labels']
            gt_labels = loss_dict['gt_labels']

            for pred_idx in range(1, self.primitive_num + 1):
                for gt_idx in range(1, gt_pm_num + 1):
                    occ1 = pred_labels == pred_idx
                    occ2 = gt_labels == gt_idx
                    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
                    area_intersect = (occ1 & occ2).astype(
                        np.float32).sum(axis=-1)

                    loss = (area_intersect / area_union).mean()
                    if not np.isfinite(loss):
                        loss = 0.
                    cost[pred_idx - 1, gt_idx - 1] += loss

        cost /= sample_num
        row_ind, col_ind = optimize.linear_sum_assignment(-cost)
        ret = {}
        total = 0.
        for r, c in zip(row_ind, col_ind):
            ret['part_loss_iou_pm{}'.format(c + 1)] = cost[r, c]
            total += cost[r, c]
        ret['part_loss_iou'] = total / len(ret) if len(ret) > 0 else 0.
        return ret, (row_ind, col_ind)

    def get_post_param_loss(self, losses, match):
        row_ind, col_ind = match
        sample_num = len(losses)

        line_distance_error = 0.
        direction_error = 0.
        rotation_error = 0.
        primitive_type_error = 0.
        cnt = 0
        primitive_type_cnt = 0
        for loss_dict in losses:
            batch_size = loss_dict['pred_rotation_direction'].shape[0]
            for pred_pm_idx, gt_pm_idx in zip(row_ind, col_ind):
                for batch_idx in range(batch_size):
                    primitive_type_cnt += 1
                    pred_primitive_type = torch.from_numpy(
                        loss_dict['pred_primitive_type'][batch_idx,
                                                         pred_pm_idx, ...])

                    target_primitive_type = torch.from_numpy(
                        loss_dict['target_primitive_type'][batch_idx,
                                                           gt_pm_idx, ...])

                    primitive_type_error += (
                        pred_primitive_type == target_primitive_type
                    ).float().mean().numpy()

                    if pred_primitive_type == target_primitive_type == 1:
                        cnt += 1
                        pred_offset = int(loss_dict['pred_rotation_offset']
                                          [batch_idx].item())
                        target_offset = int(loss_dict['target_rotation_offset']
                                            [batch_idx].item())
                        pred_pm_idx_offseted = pred_pm_idx - pred_offset
                        gt_pm_idx_offseted = gt_pm_idx - target_offset
                        """
                        pred_rotation_direction = torch.from_numpy(
                            loss_dict['pred_rotation_direction'][
                                batch_idx, pred_pm_idx_offseted, ...])
                        pred_rotation_anchor_point = torch.from_numpy(
                            loss_dict['pred_rotation_anchor_point'][
                                batch_idx, pred_pm_idx_offseted, ...])
                        pred_rotation_deg = torch.from_numpy(
                            loss_dict['pred_rotation_deg'][
                                batch_idx, pred_pm_idx_offseted, ...])

                        target_rotation_direction = torch.from_numpy(
                            loss_dict['target_rotation_direction'][
                                batch_idx, gt_pm_idx_offseted, ...])
                        target_rotation_anchor_point = torch.from_numpy(
                            loss_dict['target_rotation_anchor_point'][
                                batch_idx, gt_pm_idx_offseted, ...])
                        target_rotation_deg = torch.from_numpy(
                            loss_dict['target_rotation_deg'][
                                batch_idx, gt_pm_idx_offseted, ...])

                        line_distance_error += geometry.get_line_to_line_distance(
                            pred_rotation_anchor_point,
                            pred_rotation_direction,
                            target_rotation_anchor_point,
                            target_rotation_direction).mean().numpy()
                        direction_error += geometry.get_direction_error(
                            pred_rotation_direction,
                            target_rotation_direction).mean().numpy()
                        rotation_error += geometry.get_rotation_error_from_deg_and_direction(
                            pred_rotation_deg, pred_rotation_direction,
                            target_rotation_deg,
                            target_rotation_direction).mean().numpy()
                        """
                        pred_rotation_direction = loss_dict[
                            'pred_rotation_direction'][batch_idx,
                                                       pred_pm_idx_offseted,
                                                       ...]
                        pred_rotation_anchor_point = loss_dict[
                            'pred_rotation_anchor_point'][batch_idx,
                                                          pred_pm_idx_offseted,
                                                          ...]
                        pred_rotation_deg = loss_dict['pred_rotation_deg'][
                            batch_idx, pred_pm_idx_offseted, ...]

                        target_rotation_direction = loss_dict[
                            'target_rotation_direction'][batch_idx,
                                                         gt_pm_idx_offseted,
                                                         ...]
                        target_rotation_anchor_point = loss_dict[
                            'target_rotation_anchor_point'][batch_idx,
                                                            gt_pm_idx_offseted,
                                                            ...]
                        target_rotation_deg = loss_dict['target_rotation_deg'][
                            batch_idx, gt_pm_idx_offseted, ...]

                        line_distance_error += geometry.get_line_to_line_distance_np(
                            pred_rotation_anchor_point,
                            pred_rotation_direction,
                            target_rotation_anchor_point,
                            target_rotation_direction).mean()
                        direction_error += geometry.get_direction_error_in_deg_np(
                            pred_rotation_direction,
                            target_rotation_direction).mean()
                        rotation_error += geometry.get_rotation_error_from_deg_and_direction_np(
                            pred_rotation_deg, pred_rotation_direction,
                            target_rotation_deg,
                            target_rotation_direction).mean()

        if cnt != 0.:
            line_distance_error /= cnt
            direction_error /= cnt
            rotation_error /= cnt
        primitive_type_error /= primitive_type_cnt

        ret = dict(line_distance_error=line_distance_error,
                   direction_error=direction_error,
                   rotation_error=rotation_error)
        ret['primitive_type_error'] = primitive_type_error
        return ret

    def get_overlap_regularizer_loss(self,
                                     pred_occ,
                                     loss_weight=None,
                                     prefix=None):
        if self.overlap_regularizer_type == 'default':
            overlap_regularizer_loss = ((torch.relu(
                self.implicit_discriminator_activation(pred_occ).sum(-1) -
                self.overlap_threshold))**2).mean()
        elif self.overlap_regularizer_type == 'non_top_primitives':
            overlap_regularizer_loss = self.get_non_top_primitives_overlap_loss(
                pred_occ)
        elif self.overlap_regularizer_type == 'nsd_style':
            logits = torch.sigmoid(pred_occ).sum(-1)
            overlap_regularizer_loss = torch.relu(
                logits - self.overlap_threshold).mean()
        elif self.overlap_regularizer_type == 'only_positive':
            overlap_regularizer_loss = (((torch.relu(
                self.implicit_discriminator_activation(pred_occ) -
                0.5).sum(-1) - self.overlap_threshold))**2).mean()
        elif self.overlap_regularizer_type == 'raw_value':
            overlap_regularizer_loss = ((torch.relu(
                self.implicit_discriminator_activation(pred_occ).sum(-1) -
                self.overlap_threshold))**2).mean()
        else:
            raise NotImplementedError

        if loss_weight is None:
            loss_weight = self.overlap_regularizer_loss_weight
        overlap_regularizer_loss_weighted = overlap_regularizer_loss * loss_weight

        losses = {}
        loss_name = ('{}_'.format(prefix) if prefix is not None else
                     '') + 'overlap_regularizer_loss'
        losses[loss_name] = overlap_regularizer_loss
        losses[loss_name + '_weighted'] = overlap_regularizer_loss_weighted
        return losses, overlap_regularizer_loss_weighted
