from collections import abc
from scipy import spatial
import numpy as np
import tensorflow as tf
import trimesh
from utils import geometry


def get_primitive_mask(
    points_np,
    eval_points,
    executor,
    primitive_num,
    primitive_type='obb',
    chunksize=5,
    return_sdf=False,
):
    assert isinstance(points_np, abc.Iterable)
    assert primitive_type in ['obb', 'chull']
    assert len(eval_points.shape) == 3
    assert len(points_np) == primitive_num * eval_points.shape[0].value
    rbatch = eval_points.shape[0].value
    dim = eval_points.shape[-1].value
    points_num = eval_points.shape[-2].value
    dtype = eval_points.dtype

    if primitive_type == 'obb':
        pm_func = geometry.get_obb
    elif primitive_type == 'chull':
        pm_func = geometry.get_chull
    results = list(
        executor.map(pm_func,
                     points_np,
                     range(len(points_np)),
                     chunksize=chunksize))

    results = sorted(results, key=lambda x: x[1])
    eqs = [r[0] for r in results]

    eq_list = [len(eq) for eq in eqs if eq is not None]
    if len(eq_list) == 0:
        if return_sdf:
            return tf.keras.backend.zeros(
                [rbatch, primitive_num, points_num],
                dtype=tf.bool), tf.keras.backend.zeros(
                    [rbatch, primitive_num, points_num], dtype=dtype)
        else:
            return tf.keras.backend.zeros([rbatch, primitive_num, points_num],
                                          dtype=tf.bool), None

    max_faces_num = max(eq_list)
    beqs = []
    beqs_mask = []
    for b in range(rbatch):
        peqs = []
        peqs_mask = []
        for p in range(primitive_num):
            idx = b + p * rbatch
            #print(len(results), idx, b, p)
            eq = eqs[idx]
            if eq is None:
                peq_0 = tf.keras.backend.zeros([max_faces_num, dim],
                                               dtype=dtype)
                peq_p1 = tf.keras.backend.ones([max_faces_num, 1], dtype=dtype)
                peq = tf.concat([peq_0, peq_p1], 1)
                if return_sdf:
                    peqs_mask.append(
                        tf.keras.backend.zeros([points_num], dtype=tf.bool))
            else:
                elen = len(eq)
                eq = tf.constant(np.stack(eq), dtype=dtype)
                pad_0 = tf.keras.backend.zeros([max_faces_num - elen, dim],
                                               dtype=dtype)
                pad_p1 = -tf.keras.backend.ones([max_faces_num - elen, 1],
                                                dtype=dtype)
                pad = tf.concat([pad_0, pad_p1], 1)
                peq = tf.concat([eq, pad], 0)
                if return_sdf:
                    peqs_mask.append(
                        tf.keras.backend.ones([points_num], dtype=tf.bool))
            peqs.append(peq)
        beqs.append(tf.stack(peqs, 0))
        if return_sdf:
            beqs_mask.append(tf.stack(peqs_mask, 0))
    # B, pm, faces, dim + 1
    beqs = tf.stack(beqs, 0)
    if return_sdf:
        mask = tf.stack(beqs_mask, 0)
    # B, pm, 1, faces, dim
    normal = tf.expand_dims(beqs[..., :-1], 2)
    # B, pm, 1, faces
    bias = tf.expand_dims(beqs[..., -1], 2)
    o = tf.expand_dims(eval_points, 1)
    o = tf.expand_dims(o, 3)  # B, 1, P, 1, dim
    if return_sdf:
        sdf = tf.reduce_min(tf.reduce_sum(normal * o, -1) + bias, -1)
    else:
        mask = tf.reduce_all((tf.reduce_sum(normal * o, -1) + bias) <= 1e-3,
                             -1)
        sdf = None

    # B, pm, P
    return mask, sdf
