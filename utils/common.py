import numpy as np
import torch


def subsample_points(points,
                     num,
                     index=None,
                     axis=0,
                     return_index=False,
                     replace=False):
    if isinstance(points, np.ndarray):
        if index is None:
            index = np.random.choice(np.arange(points.shape[axis]),
                                     size=num,
                                     replace=replace)
        else:
            assert num == len(index)
        subsampled = np.take(points, index, axis=axis)
        if return_index:
            return subsampled, index
        return subsampled
    elif isinstance(points, torch.Tensor):
        if index is None:
            if replace:
                index = torch.randint(points.shape[axis],
                                      size=(points.size(0), num),
                                      device=points.device)
            else:
                index = torch.randperm(
                    points.shape[axis],
                    device=points.device)[:num].unsqueeze(0).expand(
                        points.size(0), -1)
        else:
            assert num == index.shape[0]
        subsampled = batched_index_select(points, axis, index)
        if return_index:
            return subsampled, index
        return subsampled


def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)
