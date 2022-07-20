# from im2mesh import icp
import logging
import random
import numpy as np
import trimesh
# from scipy.spatial import cKDTree
from pykeops.torch import LazyTensor
import torch
import warnings
import time
import trimesh
random.seed(0)
import sys
# sys.path.insert(0, '.')
# from utils.libmesh import check_mesh_contains
# Maximum values for bounding box [-0.5, 0.5]^3

K100 = 100000
EMPTY_PCL_DICT = {
    'completeness': np.sqrt(3),
    'accuracy': np.sqrt(3),
    'completeness2': 3,
    'accuracy2': 3,
    'chamfer': 6,
}

EMPTY_PCL_DICT_NORMALS = {
    'normals completeness': -1.,
    'normals accuracy': -1.,
    'normals': -1.,
}


def chamferl1(pred_points, target_points):
    if isinstance(pred_points, np.ndarray):
        pred_points = torch.from_numpy(pred_points).to('cuda')
    if isinstance(target_points, np.ndarray):
        target_points = torch.from_numpy(target_points).to('cuda')
    assert pred_points.ndim == 2 and target_points.ndim == 2
    gt_distances, pred_distances = chamfer_distance(pred_points.unsqueeze(0),
                                                    target_points.unsqueeze(0))
    score = 0.5 * (gt_distances.mean() +
                   pred_distances.mean()).detach().to('cpu').numpy().item()
    return {'Chamfer-L1': score}


def chamfer_distance(pred, target, pykeops=True, mode='L1'):
    assert pykeops
    # B, P, 1, dim
    pred_lazy = LazyTensor(pred.unsqueeze(2))
    # B, 1, P2, dim
    target_lazy = LazyTensor(target.unsqueeze(1))

    # B, P, P2, dim
    if mode == 'L1':
        dist = (pred_lazy - target_lazy).norm2()
    elif mode == 'L2':
        dist = (pred_lazy - target_lazy) ** 2
    else:
        raise NotImplementedError

    # B, P, dim
    pred2target = dist.min(2).squeeze(-1)

    # B, P2, dim
    target2pred = dist.min(1).squeeze(-1)

    return pred2target, target2pred


def mesh_iou(mesh_for_iou, points, values, return_occ=False, hash_resolution=512):
    out_dict = {}
    valid_meshes = []
    if isinstance(mesh_for_iou, list):
        for mesh in mesh_for_iou:
            if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
                valid_meshes.append(mesh)
    elif isinstance(mesh_for_iou,
                    trimesh.Trimesh) and (len(mesh_for_iou.vertices) < 3
                                          or len(mesh_for_iou.faces) < 1):
        out_dict['iou'] = 0.0
        return out_dict

    if isinstance(mesh_for_iou, trimesh.Scene):
        meshes = mesh_for_iou.dump()
    elif isinstance(mesh_for_iou, list):
        meshes = valid_meshes
    else:
        meshes = [mesh_for_iou]
    for m in meshes:
        trimesh.repair.fix_normals(m)
        trimesh.repair.fix_inversion(m)
        trimesh.repair.fix_winding(m)

    if len(meshes) != 0:
        for idx, mesh in enumerate(meshes):
            if idx == 0:
                occ = check_mesh_contains(mesh, points, hash_resolution=hash_resolution)
            else:
                occ |= check_mesh_contains(mesh, points, hash_resolution=hash_resolution)
    else:
        occ = check_mesh_contains(mesh_for_iou, points, hash_resolution=hash_resolution)
    out_dict['iou'] = compute_iou(occ, values)
    if return_occ:
        out_dict['occ'] = occ

    #print("eval_mesh", time.time() - t0)
    return out_dict


def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou


def one_sided_chamfer_distance_with_index(source_points, target_points):
    assert source_points.ndim in [2, 3]
    assert target_points.ndim in [2, 3]
    assert target_points.ndim == source_points.ndim
    original_ndim = target_points.ndim
    if isinstance(source_points, np.ndarray):
        source_points = torch.from_numpy(source_points).to('cuda')
    if isinstance(target_points, np.ndarray):
        target_points = torch.from_numpy(target_points).to('cuda')

    if source_points.ndim == 2:
        source_points = source_points.unsqueeze(0)
    if target_points.ndim == 2:
        target_points = target_points.unsqueeze(0)

    G_i1 = LazyTensor(source_points.unsqueeze(2))
    X_j1 = LazyTensor(target_points.unsqueeze(1))

    dist = (G_i1 - X_j1).norm2()

    # N
    idx = dist.argmin(dim=2).squeeze(-1)
    pred2target = dist.min(2).squeeze(-1)
    if original_ndim == 2:
        idx = idx[0]
        pred2target = pred2target[0]

    return pred2target.detach().to('cpu').numpy(), idx.detach().to(
        'cpu').numpy()


def sample_surface_points_from_meshes(part_meshes, num):
    assert isinstance(part_meshes, list)
    for m in part_meshes:
        trimesh.repair.fix_normals(m)
        trimesh.repair.fix_inversion(m)
        trimesh.repair.fix_winding(m)
    next_sample_cnt = 2 * num
    #next_sample_cnt = K100 + 1000
    sampled_cnt = 0
    cnt = 0
    sampled_points = []
    sampled_normals = []
    is_undersampled = False
    whole_mesh_concatenated = trimesh.util.concatenate(part_meshes)
    while sampled_cnt < num:
        cnt += 1
        if cnt > 100:
            is_undersampled = True
            break
        points, nidx = whole_mesh_concatenated.sample(next_sample_cnt,
                                                      return_index=True)
        normals = whole_mesh_concatenated.face_normals[nidx, :]
        spoints = points + normals * 1e-4
        for pidx, part_mesh in enumerate(part_meshes):
            if pidx == 0:
                occ = check_mesh_contains(part_mesh, spoints)
            else:
                occ |= check_mesh_contains(part_mesh, spoints)
        occ = ~occ
        surface_points = points[occ, :]
        sampled_points.append(surface_points)
        sampled_cnt += occ.sum()
        #next_sample_cnt = min(int(K100 / 100), K100 - sampled_cnt + 100)
        #next_sample_cnt = int(next_sample_cnt / 2) + 1000
        if next_sample_cnt <= 0:
            break

    final_surface_points = np.concatenate(sampled_points)
    if is_undersampled:
        undersampled = num - final_surface_points.shape[0]
        print('under sampled', final_surface_points.shape[0], undersampled)
        select = np.random.choice(np.arange(final_surface_points.shape[0]),
                                  size=undersampled,
                                  replace=False)
        s_points = final_surface_points[select, :]
        final_surface_points = np.concatenate([final_surface_points, s_points])
    final_surface_points = final_surface_points[:num, :]
    return final_surface_points