import numpy as np
import torch


def get_iou(pred_values, gt_values, pred_part_idx, gt_part_idx):
    return ((gt_values == gt_part_idx) &
            (pred_values == pred_part_idx)).sum().astype(np.float32) / (
                (gt_values == gt_part_idx) |
                (pred_values == pred_part_idx)).sum()


def get_unsupervised_part_iou(pred_values, gt_values):
    gt_labels = np.unique(gt_values).astype(np.uint8).tolist()
    pred_labels = np.unique(pred_values).astype(np.uint8).tolist()
    ious = []
    ret = {}
    for pred_part_idx in pred_labels:
        part_ious = {}
        if len(gt_labels) == 0:
            iou = 0.
        else:
            for gt_part_idx in gt_labels:
                part_ious[gt_part_idx] = get_iou(pred_values, gt_values,
                                                 pred_part_idx, gt_part_idx)
            part_idx = max(part_ious, key=part_ious.get)
            gt_labels.remove(part_idx)
            iou = part_ious[part_idx]
            ret['part_loss_iou_pm{}'.format(part_idx)] = iou
        ious.append(iou)
    if len(gt_labels) > 0:
        ious.extend([0] * len(gt_labels))

    mean = np.array(ious).mean()
    ret['part_loss_iou'] = mean
    return ret

def nested_tensor2np(dic):
    d = {}

    def dict_or_tensor(v):
        if isinstance(v, dict):  # assumes v is also list of pairs
            v = nested_tensor2np(v)
        elif isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        return v

    for k, v in dic.items():
        if isinstance(v, list):
            v = [dict_or_tensor(vv) for vv in v]
        else:
            v = dict_or_tensor(v)
        d[k] = v
    return d


def nested_batched_tensor2tensor(dic,
                                 bidx,
                                 batch_size,
                                 to_np=False,
                                 clone=False):
    d = {}

    def dict_or_tensor(v):
        if isinstance(v, dict):  # assumes v is also list of pairs
            v = nested_batched_tensor2tensor(v, bidx, batch_size, to_np=to_np)
        elif (isinstance(v, np.ndarray) or isinstance(
                v, torch.Tensor)) and v.ndim > 0 and v.shape[0] == batch_size:
            v = v[bidx, ...]
            if clone and isinstance(v, torch.Tensor):
                v = v.clone()
        if to_np and isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        return v

    for k, v in dic.items():
        if isinstance(v, list):
            v = [dict_or_tensor(vv) for vv in v]
        else:
            v = dict_or_tensor(v)
        d[k] = v
    return d


def clone_nested_tensors(dic):
    d = {}

    def dict_or_tensor(v):
        if isinstance(v, dict):  # assumes v is also list of pairs
            v = clone_nested_tensors(v)
        elif isinstance(v, torch.Tensor):
            v = v.clone()
        return v

    for k, v in dic.items():
        if isinstance(v, list):
            v = [dict_or_tensor(vv) for vv in v]
        else:
            v = dict_or_tensor(v)
        d[k] = v
    return d
