import torch
from collections import defaultdict
import torch.nn.functional as F
from PIL import Image
import numpy as np

import plotly.graph_objects as go
import seaborn as sns
import io
from loss import chamfer_loss
from utils import visualizer_util
import random


class Trainer:
    def __init__(self,
                 model,
                 optimizers,
                 device,
                 shape_recon_chamfer_loss_weight=1.,
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
                 use_canonical_recon_loss=False):
        self.model = model
        self.optimizers = optimizers
        self.primitive_num = self.model.primitive_num
        self.device = device
        self.shape_recon_chamfer_loss_weight = shape_recon_chamfer_loss_weight
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
        self.dataset_boxsize = 50
        self.dataset_worldsize = 255

        self.use_canonical_self_supervised_slide_loss = use_canonical_self_supervised_slide_loss
        self.use_canonical_recon_loss = use_canonical_recon_loss
        self.train_canonical = self.use_canonical_recon_loss or self.use_canonical_self_supervised_slide_loss
        self.self_supervised_slide_loss_weight = self_supervised_slide_loss_weight

        assert self.train_explicit != self.train_implicit
        self.visualizer = visualizer_util.Visualizer(self.primitive_num)

    def eval_step(self, data):
        self.model.eval()
        if self.train_explicit:
            all_losses = {}
            for losses in self.compute_loss(data,
                                            part_seg_loss=True,
                                            param_loss=True):
                all_losses.update(losses)
            self.model.zero_grad()
            return all_losses
        else:
            raise NotImplementedError

    def visualize(self, data):
        self.model.eval()

        moving_points = data['moving_points'].to(self.device)
        moving_label = data['moving_label'].long()
        slide = data['slide'].unsqueeze(-1)
        # recon result
        # recon moved by pred slide
        # recon moved by gt slide
        # input (gt) moved by gt slide
        # input (gt) moved by pred slide

        result_images = []

        images = self.visualizer.visualize_pointcloud(
            moving_points.clone().detach().cpu().numpy(), moving_label.numpy())
        result_images.append({
            'type': 'image',
            'desc': 'original_moving_points',
            'data': images
        })

        if self.train_explicit:
            with torch.no_grad():
                ret = self.model(moving_points)
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
            moved_gt_points_by_pred_tensor = moving_points.detach().cpu(
            ).clone()
            moved_gt_points_by_pred_batchwise_list = []
            new_label_batchwise_list = []
            for batch_idx in range(moved_gt_points_by_pred_tensor.shape[0]):
                moved_gt_points_by_pred_list = []
                new_label_list = []
                for idx in range(1, self.model.primitive_num + 1):
                    t = moved_gt_points_by_pred_tensor[batch_idx, moving_label[
                        batch_idx, :] == idx, :]
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
            moved_gt_points_by_gt_tensor = moving_points.detach().cpu().clone()
            moved_gt_points_by_gt_batchwise_list = []
            new_label_batchwise_list = []
            for batch_idx in range(moved_gt_points_by_gt_tensor.shape[0]):
                moved_gt_points_by_gt_list = []
                new_label_list = []
                for idx in range(1, self.model.primitive_num + 1):
                    t = moved_gt_points_by_gt_tensor[batch_idx, moving_label[
                        batch_idx, :] == idx, :]
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
        else:
            raise NotImplementedError

        result_images.append({
            'desc': 'pred_slide',
            'type': 'array',
            'data': pred_slide.numpy()
        })
        return result_images

    def train_step(self, data):
        self.model.train()
        all_losses = {}
        for losses in self.compute_loss(data):
            if 'total_D_loss' in losses:
                self.optimizers['discriminator'].zero_grad()
                losses['total_D_loss'].backward()
                self.optimizers['discriminator'].step()
            elif 'total_G_loss' in losses:
                self.optimizers['generator'].zero_grad()
                losses['total_G_loss'].backward()
                self.optimizers['generator'].step()
            all_losses.update(losses)
        return all_losses

    def compute_loss(self,
                     data,
                     skip_gp=False,
                     part_seg_loss=False,
                     param_loss=False):
        moving_points = data['moving_points'].to(self.device)
        moving_label = data['moving_label'].to(self.device).long()
        slide = data['slide'].to(self.device).unsqueeze(-1)

        canonical_points = data['canonical_points'].to(self.device)
        #canonical_label = data['canonical_label'].to(self.device).long()

        if self.train_explicit:

            # Infererence for discrminator
            ret = self.model(moving_points)
            pred_points = ret['point_set']
            pred_slide = ret['param']
            # assuming primitive 1 is moving
            moved_pred_points = [points.clone() for points in pred_points]
            moved_pred_points[1][:, :, 1] -= pred_slide
            moved_pred_points_cat = torch.cat(moved_pred_points, axis=1)

            if part_seg_loss:
                losses = defaultdict(lambda: 0)
                batch_size = moving_points.size(0)
                for batch_idx in range(batch_size):
                    for idx in range(1, self.primitive_num + 1):
                        pred = pred_points[idx -
                                           1][batch_idx, :, :].unsqueeze(0)
                        gt = moving_points[batch_idx, moving_label[
                            batch_idx, :] == idx, :].unsqueeze(0)
                        loss = chamfer_loss.chamfer_loss(pred, gt)
                        losses['part_loss_pm{}'.format(
                            idx)] += loss / batch_size
                total_loss = 0.
                for key, value in losses.items():
                    total_loss += value / self.primitive_num
                losses['part_loss'] = total_loss
                yield losses
            if param_loss:
                loss = F.mse_loss(pred_slide, slide)
                losses = {'slide_loss': loss}
                yield losses
            # loss
            canonical_points_len = canonical_points.shape[1]
            moved_pred_points_cat_len = moved_pred_points_cat.shape[1]

            if canonical_points_len < moved_pred_points_cat_len:
                moved_sample_idx = random.sample(
                    range(moved_pred_points_cat_len), canonical_points_len)
                moved_pred_points_cat = moved_pred_points_cat[:,
                                                              moved_sample_idx, :]
            else:
                sample_idx = random.sample(range(canonical_points_len),
                                           moved_pred_points_cat_len)
                canonical_points = canonical_points[:, sample_idx, :]

            ## Adversarial loss
            ### Discriminator
            D_real = self.model(canonical_points, mode='discriminator')['D']
            D_real_loss = -D_real.mean()
            D_real_loss_weighted = D_real_loss * self.D_real_loss_weight * self.D_loss_weight

            D_fake = self.model(moved_pred_points_cat.detach(),
                                mode='discriminator')['D']
            D_fake_loss = D_fake.mean()
            D_fake_loss_weighted = D_fake_loss * self.D_fake_loss_weight * self.D_loss_weight

            #### Gradient penalty
            batch_size = canonical_points.size(0)

            alpha = torch.rand(batch_size, 1, 1,
                               requires_grad=True).to(self.device)
            # randomly mix real and fake data
            interpolates = canonical_points + alpha * (moved_pred_points_cat -
                                                       canonical_points)
            # compute output of D for interpolated input
            disc_interpolates = self.model(interpolates,
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

                gradient_penalty = ((
                    (gradients.norm(2, dim=1) - self.gp_gamma) / self.gp_gamma)
                                    **2).mean() * self.gp_lambda

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
            ret = self.model(moving_points)
            pred_points = ret['point_set']
            pred_slide = ret['param']
            """
            print('pred_slide.shape', pred_slide.shape)
            print('point_set[0].shape', point_set[0].shape)
            print('moving_points.shape', moving_points.shape)
            print('moving_label.shape', moving_label.shape)
            """

            # assuming primitive 1 is moving
            moved_pred_points = [points.clone() for points in pred_points]
            moved_pred_points[1][:, :, 1] -= pred_slide
            moved_pred_points_cat = torch.cat(moved_pred_points, axis=1)
            pred_points_cat = torch.cat(pred_points, axis=1)

            losses = {}
            ## recon loss
            recon_loss = chamfer_loss.chamfer_loss(pred_points_cat,
                                                   moving_points)
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
                canonical_points = data['canonical_points'].to(self.device)
                moving_points = data['moving_points'].to(self.device)

                ret = self.model(canonical_points, return_param=False)
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
                        pred_points_cat, canonical_points)
                    losses['canonical_recon_loss'] = recon_loss
                    recon_loss_weighted = recon_loss * self.shape_recon_chamfer_loss_weight
                    losses[
                        'canonical_recon_loss_weighted'] = recon_loss_weighted
                    total_G_loss += recon_loss_weighted

                if self.use_canonical_self_supervised_slide_loss:
                    ret = self.model(moved_pred_points_cat,
                                     return_points=False)
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
                """
                ### Generator
                moved_pred_points_cat_len = moved_pred_points_cat.shape[1]
                moving_points_len = moving_points.shape[1]

                if moving_points_len < moved_pred_points_cat_len:
                    moved_sample_idx = random.sample(
                        range(moved_pred_points_cat_len), moving_points_len)
                    moved_pred_points_cat = moved_pred_points_cat[:,
                                                                  moved_sample_idx, :]
                else:
                    moving_points_sample_idx = random.sample(
                        range(moving_points_len), moved_pred_points_cat_len)
                    moving_points = moved_pred_points_cat[:,
                                                          moving_points_sample_idx, :]
                """

            yield losses
        else:
            raise NotImplementedError


class OverfitTrainer:
    def __init__(self,
                 model,
                 optimizer,
                 device,
                 shape_recon_loss_weight=1.,
                 shape_recon_chamfer_loss_weight=1.,
                 shape_recon_chamfer_loss_weight_moved=1.,
                 param_loss_weight=1.,
                 learn_by_moving=False,
                 learn_by_moving_param_supervised=False,
                 learn_only_by_chamfer_distance=False,
                 train_explicit=False,
                 **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.primitive_num = self.model.primitive_num
        self.device = device
        self.shape_recon_loss_weight = shape_recon_loss_weight
        self.shape_recon_chamfer_loss_weight = shape_recon_chamfer_loss_weight
        self.shape_recon_chamfer_loss_weight_moved = shape_recon_chamfer_loss_weight_moved
        self.param_loss_weight = param_loss_weight
        self.learn_by_moving = learn_by_moving
        self.learn_by_moving_param_supervised = learn_by_moving_param_supervised
        self.learn_only_by_chamfer_distance = learn_only_by_chamfer_distance
        self.train_explicit = train_explicit

        self.visualizer = visualize_util.Visualizer(self.primitive_num)

    def compute_part_label_supervised_param_loss(self, pred_param, param, occ,
                                                 label):
        # losses
        partwise_shape_recon_loss = F.cross_entropy(occ, label)
        param_loss = F.mse_loss(pred_param, param.unsqueeze(-1))

        total_loss = partwise_shape_recon_loss * self.shape_recon_loss_weight + param_loss * self.param_loss_weight

        return {
            'total_loss': total_loss,
            'partwise_shape_recon_loss': partwise_shape_recon_loss,
            'param_loss': param_loss
        }

    def compute_supervised_param_loss_explicit(self, pred_param, param,
                                               pred_points, points, label):
        # losses
        partwise_shape_recon_chamfer_loss = chamfer_loss.chamfer_loss(
            torch.cat(pred_points, 1), points)
        param_loss = F.mse_loss(pred_param, param.unsqueeze(-1))

        total_loss = partwise_shape_recon_chamfer_loss * self.shape_recon_chamfer_loss_weight + param_loss * self.param_loss_weight

        return {
            'total_loss': total_loss,
            'partwise_shape_recon_chamfer_loss':
            partwise_shape_recon_chamfer_loss,
            'param_loss': param_loss
        }

    def compute_moving_part_loss_explicit(self, pred_points, moved_pred_points,
                                          points, points_can):
        partwise_shape_recon_chamfer_loss = chamfer_loss.chamfer_loss(
            torch.cat(pred_points, 1), points)
        partwise_shape_recon_chamfer_loss_moved = chamfer_loss.chamfer_loss(
            torch.cat(moved_pred_points, 1), points_can)

        total_loss = partwise_shape_recon_chamfer_loss * self.shape_recon_chamfer_loss_weight + partwise_shape_recon_chamfer_loss_moved * self.shape_recon_chamfer_loss_weight_moved
        return {
            'total_loss':
            total_loss,
            'partwise_shape_recon_chamfer_loss':
            partwise_shape_recon_chamfer_loss,
            'partwise_shape_recon_chamfer_loss_moved':
            partwise_shape_recon_chamfer_loss_moved
        }

    def compute_moving_part_loss_explicit_param_supervised(
            self, pred_points, moved_pred_points, points, points_can,
            pred_param, param):
        partwise_shape_recon_chamfer_loss = chamfer_loss.chamfer_loss(
            torch.cat(pred_points, 1), points)
        partwise_shape_recon_chamfer_loss_moved = chamfer_loss.chamfer_loss(
            torch.cat(moved_pred_points, 1), points_can)

        param_loss = F.mse_loss(pred_param, param.unsqueeze(-1))

        total_loss = partwise_shape_recon_chamfer_loss * self.shape_recon_chamfer_loss_weight + partwise_shape_recon_chamfer_loss_moved * self.shape_recon_chamfer_loss_weight_moved + param_loss * self.param_loss_weight

        return {
            'total_loss': total_loss,
            'partwise_shape_recon_chamfer_loss':
            partwise_shape_recon_chamfer_loss,
            'partwise_shape_recon_chamfer_loss_moved':
            partwise_shape_recon_chamfer_loss_moved,
            'param_loss': param_loss
        }

    def compute_explicit_loss_without_param_supervision(
            self, pred_points, points):
        partwise_shape_recon_chamfer_loss = chamfer_loss.chamfer_loss(
            torch.cat(pred_points, 1), points)

        total_loss = partwise_shape_recon_chamfer_loss * self.shape_recon_chamfer_loss_weight

        return {
            'total_loss': total_loss,
            'partwise_shape_recon_chamfer_loss':
            partwise_shape_recon_chamfer_loss
        }

    def eval_step(self, data):
        self.model.eval()
        points = data['points'].to(self.device)
        points_can = data['points_can'].to(self.device)
        label = data['label'].to(self.device).long()
        label_can = data['label_can'].to(self.device)
        slide = data['slide'].to(self.device)

        if self.train_explicit:
            with torch.no_grad():
                ret = self.model(points)
            pred_points = ret['point_set']
        else:
            with torch.no_grad():
                ret = self.model(points, points)
            occ = ret['occupancy']
        pred_slide = ret['param']
        # losses

        if self.train_explicit:
            return self.compute_supervised_param_loss_explicit(
                pred_slide, slide, pred_points, points, label)
        else:
            return self.compute_part_label_supervised_param_loss(
                pred_slide, slide, occ, label)

    def visualize(self, data):
        self.model.eval()
        points = data['points'].to(self.device)

        if self.train_explicit:
            with torch.no_grad():
                ret = self.model(points)
            points = []
            occ = []
            for idx, point_set in enumerate(ret['point_set']):
                points.append(point_set)
                occ.append(torch.zeros(point_set.shape[:2]) + idx)
            points = torch.cat(points, axis=1)

            moved_pred_points = [points.clone() for points in pred_points]
            moved_pred_points[0][:, :, 1] -= pred_slide

            part = torch.cat(occ, axis=1).detach().cpu().numpy()

        else:
            with torch.no_grad():
                ret = self.model(points, points)
            occ = ret['occupancy']
            part = occ.argmax(1).detach().cpu().numpy()

        pred_slide = ret['param']
        images = self.visualizer.visualize_pointcloud(
            points.detach().cpu().numpy(), part)
        return [{
            'type': 'image',
            'data': images
        }, {
            'type': 'image',
            'data': moved_images
        }, {
            'type': 'array',
            'data': pred_slide.detach().cpu().numpy()
        }]

    def train_step(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        losses = self.compute_loss(data)
        losses['total_loss'].backward()
        self.optimizer.step()
        return losses

    def compute_loss(self, data):
        points = data['points'].to(self.device)
        points_can = data['points_can'].to(self.device)
        label = data['label'].to(self.device).long()
        label_can = data['label_can'].to(self.device)
        slide = data['slide'].to(self.device)

        # args = points, inputs (condition)

        if self.train_explicit:
            ret = self.model(points)
            pred_points = ret['point_set']
        else:
            ret = self.model(points, points)
            occ = ret['occupancy']
        pred_slide = ret['param']
        # losses
        if self.learn_by_moving:
            if self.train_explicit:
                # assuming primitive 0 is moving
                moved_pred_points = [points.clone() for points in pred_points]
                moved_pred_points[0][:, :, 1] -= pred_slide
                return self.compute_moving_part_loss_explicit(
                    pred_points, moved_pred_points, points, points_can)
            else:
                raise NotImplementedError
        elif self.learn_by_moving_param_supervised:
            if self.train_explicit:
                # assuming primitive 0 is moving
                moved_pred_points = [points.clone() for points in pred_points]
                moved_pred_points[0][:, :, 1] -= pred_slide
                return self.compute_moving_part_loss_explicit_param_supervised(
                    pred_points, moved_pred_points, points, points_can,
                    pred_slide, slide)
            else:
                raise NotImplementedError

        elif self.learn_only_by_chamfer_distance:
            if self.train_explicit:
                return self.compute_explicit_loss_without_param_supervision(
                    pred_points, points)
            else:
                raise NotImplementedError
        else:
            if self.train_explicit:
                return self.compute_supervised_param_loss_explicit(
                    pred_slide, slide, pred_points, points, label)
            else:
                return self.compute_part_label_supervised_param_loss(
                    pred_slide, slide, occ, label)
