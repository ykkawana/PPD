from trainer import imex_trainer
import torch
from utils import visualizer_util
import numpy as np
import torch.nn.functional as F
from utils import geometry
import random
from pytorch_metric_learning import losses as metric_losses
import numpy as np


class Trainer(imex_trainer.Trainer):
    def __init__(self, model, optimizers, device, **kwargs):
        super().__init__(model, optimizers, device, **kwargs)

    def visualize(self, data):
        result_images = []

        points = data['points']
        values = data['values']
        images = self.visualizer.visualize_pointcloud(points.numpy(),
                                                      values.numpy())
        result_images.append({
            'type': 'image',
            'desc': 'pretrain_gt',
            'data': images
        })
        cat = torch.cat([points, values.clamp(max=1).unsqueeze(-1)],
                        dim=-1).to(self.device)
        with torch.no_grad():
            ret = self.model(cat,
                             points.clone().to(self.device),
                             return_generator=True)
        pred_occ = ret['occupancy']
        pred_occ_max, pred_occ_argmax = torch.sigmoid(pred_occ).max(axis=-1)
        pred_occ_label = torch.where(
            pred_occ_max >= self.visualize_isosurface_threshold,
            pred_occ_argmax + 1,
            torch.zeros_like(pred_occ_max).long())

        # Original pred points
        images = self.visualizer.visualize_pointcloud(
            points.numpy(),
            (torch.sigmoid(pred_occ) >= self.visualize_isosurface_threshold
             ).squeeze(-1).detach().cpu().numpy())

        result_images.append({
            'type': 'image',
            'desc': 'pretrain_pred',
            'data': images
        })

        return result_images

    def train_implicit_losses(self,
                              data,
                              step,
                              skip_gp,
                              return_eval_loss=False):

        points = data['points'].to(self.device)
        values = data['values'].to(self.device).float()
        inputs = torch.cat([points, values.clamp(max=1).unsqueeze(-1)], dim=-1)
        target_params = {}
        # loss

        if return_eval_loss:
            eval_losses = self.compute_eval_losses(values, points, inputs)
            yield eval_losses

        gen_losses = self.train_pretrain_regularizer_net(
            values, points, inputs)
        yield gen_losses

    def compute_eval_losses(self, values, points, inputs):
        with torch.no_grad():
            ret = self.model(inputs, points, return_generator=True)
        pred_values = ret['occupancy']

        eval_losses = self.get_eval_losses(pred_values, values)

        return eval_losses

    def get_eval_losses(self, pred_values, values):
        disc_losses = {}
        if not self.disable_part_seg_loss_iou_for_eval:
            part_seg_iou_loss_ret = self.get_iou_loss_for_eval(
                pred_values, values)
            disc_losses.update(part_seg_iou_loss_ret)

        return disc_losses

    def train_pretrain_regularizer_net(self, values, points, inputs):
        inputs = torch.cat([points, values.clamp(max=1).unsqueeze(-1)], dim=-1)
        losses = {}
        total_G_loss = 0.
        """
        Inference for generator
        """
        ret = self.model(inputs, points, return_generator=True)

        # occupancy value of "canonically posed" input shape
        pred_values = ret['occupancy']
        occ_loss = F.binary_cross_entropy_with_logits(
            pred_values, values.clamp(max=1).unsqueeze(-1),
            reduction='none').sum(-1).mean()

        occ_loss_weighted = occ_loss * self.occupancy_loss_weight
        losses['recon_loss'] = occ_loss
        losses['recon_loss_weighted'] = occ_loss_weighted

        losses['total_G_loss'] = occ_loss_weighted
        return losses
