import numpy as np
import torch
from pykeops.torch import LazyTensor
from torch.nn import functional as F


def chamfer_loss(source_points,
                 target_points,
                 source_mask=None,
                 target_mask=None,
                 invalid_const=100,
                 invalid_th=10,
                 mode='l2'):
    """
    Args:
        source_points: B x Ps x 2 or 3
        target_points: B x Pt x 2 or 3
        source_mask: B x Pt x 1
        target_mask: B x Pt x 1
    """

    B, P1, D1 = source_points.shape
    _, P2, D2 = target_points.shape
    is_source_mask = False
    if source_mask is not None:
        is_source_mask = True
        _, mP1, _ = source_mask.shape
        assert mP1 == P1
        source_points = source_points * source_mask + invalid_const * (
            ~source_mask)
    is_target_mask = False
    if target_mask is not None:
        is_target_mask = True
        _, mP2, _ = target_mask.shape
        assert mP2 == P2
        target_points = target_points * target_mask - invalid_const * (
            ~target_mask)

    G_i1 = LazyTensor(source_points.unsqueeze(2))
    X_j1 = LazyTensor(target_points.unsqueeze(1))

    if mode == 'l2':
        dist = (G_i1 - X_j1).sqnorm2()
    elif mode == 'l1':
        dist = (G_i1 - X_j1).norm2()
    else:
        raise NotImplementedError

    # B x (Ps * N)
    idx1 = dist.argmin(dim=2)
    target_points_selected = batched_index_select(target_points, 1,
                                                  idx1.view(B, -1))
    diff_primitives2target = source_points - target_points_selected

    if mode == 'l2':
        loss_source2target = (diff_primitives2target**2).sum(-1)
    elif mode == 'l1':
        loss_source2target = torch.norm(diff_primitives2target, None,
                                        dim=2).squeeze(-1)

    if is_source_mask or is_target_mask:
        source2target_mask = (loss_source2target < invalid_th)
        loss_source2target = loss_source2target * source2target_mask

    idx2 = dist.argmin(dim=1)  # Grid

    source_points_selected = batched_index_select(source_points, 1,
                                                  idx2.view(B, -1))
    diff_target2source = source_points_selected - target_points
    if mode == 'l2':
        loss_target2source = (diff_target2source**2).sum(-1)
    elif mode == 'l1':
        loss_target2source = torch.norm(diff_target2source, None,
                                        dim=2).squeeze(-1)

    if is_source_mask or is_target_mask:
        target2source_mask = (loss_target2source < invalid_th)
        loss_target2source = loss_target2source * target2source_mask

    if is_source_mask or is_target_mask:
        return (loss_target2source.sum(-1) / target2source_mask.sum(-1)).mean(
        ) + (loss_source2target.sum(-1) / source2target_mask.sum(-1)).mean()
    else:
        return loss_target2source.mean() + loss_source2target.mean()


def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)
