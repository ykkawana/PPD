from collections import abc
from scipy import spatial
import numpy as np
import torch
import trimesh
from torch.nn import functional as F

EPS = 1e-7


def get_chull(points_np, idx):
    try:
        hull = spatial.ConvexHull(points_np)
        eq = hull.equations
    except BaseException as e:
        eq = None
    return eq, idx


def get_obb(points_np, idx):
    try:
        to, ex = trimesh.bounds.oriented_bounds(points_np)
        if len(ex) == 2:
            n = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
        elif len(ex) == 3:
            n = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                          [-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
        else:
            raise NotImplementedError
        n *= ex[np.newaxis, ...] / 2
        to_inv = trimesh.transformations.inverse_matrix(to)
        to_tr = to_inv.copy()[:-1, -1]
        to_inv[:-1, -1] = 0.
        rotated_n = trimesh.transformations.transform_points(n, to_inv)
        tr_n = rotated_n / np.clip(
            np.linalg.norm(rotated_n, axis=-1, keepdims=True), 1e-7, None)
        translated_rotated_n = rotated_n + to_tr[np.newaxis, ...]
        b = -(tr_n * translated_rotated_n).sum(-1, keepdims=True)

        eq = np.concatenate([tr_n, b], axis=1)
        eq_sp = [e[0, :] for e in np.split(eq, eq.shape[0])]
    except BaseException as e:
        #raise e
        eq_sp = None
    return eq_sp, idx


def get_primitive_mask(points_np,
                       eval_points,
                       executor,
                       primitive_num,
                       primitive_type='obb',
                       chunksize=5):
    assert isinstance(points_np, abc.Iterable)
    assert primitive_type in ['obb', 'convex']
    assert eval_points.ndim == 3
    assert len(points_np) == primitive_num * eval_points.size(0)
    if primitive_type == 'obb':
        pm_func = get_obb
    elif primitive_type == 'convex':
        pm_func = get_chull
    points_num = eval_points.size(-2)
    rbatch = eval_points.size(0)
    results = list(
        executor.map(pm_func,
                     points_np,
                     range(len(points_np)),
                     chunksize=chunksize))

    results = sorted(results, key=lambda x: x[1])
    eqs = [r[0] for r in results]
    eq_list = [len(eq) for eq in eqs if eq is not None]
    if len(eq_list) == 0:
        return torch.zeros([rbatch, primitive_num, points_num],
                           dtype=torch.bool,
                           device=eval_points.device)

    max_faces_num = max(eq_list)
    beqs = []
    dim = eval_points.size(-1)
    dtype = eval_points.dtype
    device = eval_points.device
    for b in range(rbatch):
        peqs = []
        for p in range(primitive_num):
            idx = b + p * rbatch
            #print(len(results), idx, b, p)
            eq = eqs[idx]
            if eq is None:
                peq = torch.zeros([max_faces_num, dim + 1],
                                  device=device,
                                  dtype=dtype)
                peq[:, -1] = 1.
            else:
                elen = len(eq)
                eq = torch.from_numpy(np.stack(eq)).type(dtype).to(device)
                pad = torch.zeros([max_faces_num - elen, dim + 1],
                                  device=device,
                                  dtype=dtype)
                pad[:, -1] = -1.
                peq = torch.cat([eq, pad])
            peqs.append(peq)
        beqs.append(torch.stack(peqs))
    # B, pm, faces, dim + 1
    beqs = torch.stack(beqs)
    # B, pm, 1, faces, dim
    normal = beqs[..., :-1].unsqueeze(2)
    # B, pm, 1, faces
    bias = beqs[..., -1].unsqueeze(2)
    o = eval_points.unsqueeze(1)
    o = o.unsqueeze(3)  # B, 1, P, 1, dim
    mask = (((normal * o).sum(-1) + bias) <= 1e-3).all(dim=-1)
    # B, pm, P
    return mask


def apply_3d_rotation(coord, rotation, inv=False):
    if coord.ndim == 4 and rotation.ndim == 3:
        B, N, P, dim = coord.shape
        B2, N2, D = rotation.shape
        assert N == N2, (N, N2)
    elif coord.ndim == 3 and rotation.ndim == 2:
        B, P, dim = coord.shape
        B2, D = rotation.shape
        assert B == B2
    else:
        raise NotImplementedError
    assert B == B2

    # rotation quaternion in [w, x, y, z]

    q = rotation
    if inv:
        w, x, y, z = q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]
    else:
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    qq = rotation**2
    w2, x2, y2, z2 = qq[..., 0], qq[..., 1], qq[..., 2], qq[..., 3]
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    if coord.ndim == 4 and rotation.ndim == 3:
        shape = (B2, N2, 1, 3, 3)
    elif coord.ndim == 3 and rotation.ndim == 2:
        shape = (B2, 1, 3, 3)
    else:
        raise NotImplementedError

    rot_mat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                          dim=-1).view(*shape)

    rotated_coord = (coord.unsqueeze(-2) * rot_mat).sum(-1)
    return rotated_coord


def get_quaternion(direction, angle, normalize_direction=False):
    if normalize_direction:
        direction = F.normalize(direction, dim=-1)
    quat_xyz = direction * torch.sin(angle / 2)
    quat_w = torch.cos(angle / 2)
    # B, pm, 4
    quat = torch.cat([quat_w, quat_xyz], dim=-1)
    return quat


def get_line_to_line_distance(point1, direction1, point2, direction2):
    orth_vect = torch.cross(direction1, direction2, dim=-1)
    dist = ((point1 - point2) *
            torch.nn.functional.normalize(orth_vect, dim=-1)).sum(-1)

    return dist.abs()


def get_line_to_line_distance_np(point1, direction1, point2, direction2):
    p1 = point1.reshape(-1)
    p2 = point2.reshape(-1)
    e1 = direction1.reshape(-1)
    e2 = direction2.reshape(-1)
    orth_vect = np.cross(e1, e2)
    product = np.sum(orth_vect * (p1 - p2))
    dist = product / np.linalg.norm(orth_vect)

    return np.abs(dist)


def get_direction_error(direction1, direction2):
    r_diff = torch.nn.functional.cosine_similarity(direction1,
                                                   direction2,
                                                   dim=-1)
    # print(r_diff)
    return r_diff


def get_direction_error_in_deg_np(direction1, direction2):
    v1 = direction1.reshape(-1)
    v2 = direction2.reshape(-1)
    r_diff = np.sum(v1 * v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(r_diff)


def get_rotation_matrix_from_deg_and_direction(deg, direction):
    if deg.ndim == direction.ndim:
        if deg.shape[-1] == 1:
            deg = deg[..., 0]
        else:
            raise NotImplementedError
    sina = deg.sin()
    cosa = deg.cos()

    # rotation matrix around unit vector
    M = torch.zeros(*deg.shape, 4, 4, dtype=deg.dtype, device=deg.device)
    M[..., 0, 0] = cosa
    M[..., 1, 1] = cosa
    M[..., 2, 2] = cosa
    M[..., 3, 3] = 1
    outer = direction.unsqueeze(-2) * direction.unsqueeze(-1)
    M[..., :3, :3] += outer * (1.0 - cosa.unsqueeze(-1).unsqueeze(-1))

    direction = direction * sina.unsqueeze(-1)
    row1 = torch.stack([
        torch.zeros_like(direction[..., 0]), -direction[..., 2], direction[...,
                                                                           1]
    ],
                       dim=-1)
    row2 = torch.stack([
        direction[..., 2],
        torch.zeros_like(direction[..., 0]), -direction[..., 0]
    ],
                       dim=-1)
    row3 = torch.stack([
        -direction[..., 1], direction[..., 0],
        torch.zeros_like(direction[..., 0])
    ],
                       dim=-1)
    M[..., :3, :3] += torch.stack([row1, row2, row3], dim=-2)
    return M


def get_rotation_error_from_deg_and_direction(deg1, direction1, deg2,
                                              direction2):
    rot1 = get_rotation_matrix_from_deg_and_direction(deg1, direction1)
    rot2 = get_rotation_matrix_from_deg_and_direction(deg2, direction2)

    return get_rotation_error_from_rotation_matrix(rot1, rot2)


def get_rotation_error_from_rotation_matrix(rot1, rot2):
    mm = torch.matmul(rot1, rot2.transpose(-1, -2))
    trace = mm[..., 0, 0] + mm[..., 1, 1] + mm[..., 2, 2]
    return torch.acos((trace - 1) / 2) % (2 * np.pi)


def get_translation_error(translation1, translation2):
    return torch.norm(translation1 - translation2, dim=-1)


def get_rotation_error_from_deg_and_direction_np(deg1, direction1, deg2,
                                                 direction2):
    rot1 = trimesh.transformations.rotation_matrix(deg1, direction1)
    rot2 = trimesh.transformations.rotation_matrix(deg2, direction2)

    return get_rotation_error_from_rotation_matrix_np(rot1, rot2)


def get_rotation_error_from_rotation_matrix_np(R1, R2):
    return np.arccos((np.trace(np.matmul(R1, R2.T)) - 1) / 2) % (2 * np.pi)


def get_align_rotation_between_two_np(a, b):
    if a.shape != (3, ) or b.shape != (3, ):
        raise ValueError('vectors must be (3,)!')

    # find the SVD of the two vectors
    au = np.linalg.svd(a.reshape((-1, 1)))[0]
    bu = np.linalg.svd(b.reshape((-1, 1)))[0]

    if np.linalg.det(au) < 0:
        au[:, -1] *= -1.0
    if np.linalg.det(bu) < 0:
        bu[:, -1] *= -1.0

    # put rotation into homogeneous transformation
    matrix = np.eye(4)
    return bu.dot(au.T)


def get_align_rotation_between_two(a, b):
    if a.shape[-1] != 3 or b.shape[-1] != 3:
        raise ValueError('vectors dim must be 3!')
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError('vectors ndim must be 2 (batch, 3)')

    # find the SVD of the two vectors
    au = torch.svd(a.unsqueeze(-1), some=False)[0]
    bu = torch.svd(b.unsqueeze(-1), some=False)[0]

    ndeta = torch.det(au) < 0
    maska = torch.ones_like(au)
    maska[ndeta, :, -1] = -1
    au = au * maska
    ndetb = torch.det(bu) < 0
    maskb = torch.ones_like(bu)
    maskb[ndetb, :, -1] = -1
    bu = bu * maskb

    # put rotation into homogeneous transformation
    return torch.matmul(bu, au.transpose(1, 2))


def convert_axis_angle_from_quaternion_np(q):
    """Compute axis-angle from quaternion.
    This operation is called logarithmic map.
    We usually assume active rotations.
    Parameters
    ----------
    q : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)
    Returns
    -------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle). The angle is
        constrained to [0, pi) so that the mapping is unique.
    """
    p = q[1:]
    p_norm = np.linalg.norm(p)
    if p_norm < np.finfo(float).eps:
        return np.array([1.0, 0.0, 0.0]), 0.0
    else:
        axis = p / p_norm
        angle = (2.0 * np.arccos(q[0]), )

    return axis, angle


def convert_axis_angle_from_quaternion(q):
    """Compute axis-angle from quaternion.
    This operation is called logarithmic map.
    We usually assume active rotations.
    Parameters
    ----------
    q : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)
    Returns
    -------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle). The angle is
        constrained to [0, pi) so that the mapping is unique.
    """
    p = q[..., 1:]
    p_norm = torch.norm(p, dim=-1, keepdim=True)

    x_axis = torch.zeros_like(p)
    x_axis[..., 0] = 1.0
    zero_angle = torch.zeros_like(q)[..., 0]

    mask = p_norm >= torch.finfo(q.dtype).eps

    axis = F.normalize(p, dim=-1)
    angle = 2 * torch.acos(q[..., 0])

    axis = axis * mask + x_axis * (~mask)
    angle = angle * mask[..., 0] + zero_angle * (~mask[..., 0])
    return axis, angle


def normalize_vector(v, return_mag=False):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(
        v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    if (return_mag == True):
        return v, v_mag[:, 0]
    else:
        return v


def compute_rotation_matrix_from_quaternion(quaternion):
    batch = quaternion.shape[0]

    quat = normalize_vector(quaternion).contiguous()

    qw = quat[..., 0].contiguous().view(batch, 1)
    qx = quat[..., 1].contiguous().view(batch, 1)
    qy = quat[..., 2].contiguous().view(batch, 1)
    qz = quat[..., 3].contiguous().view(batch, 1)

    # Unit quaternion rotation matrices computatation
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    xw = qx * qw
    yw = qy * qw
    zw = qz * qw

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw),
                     1)  #batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw),
                     1)  #batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy),
                     1)  #batch*3

    matrix = torch.cat(
        (row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(
            batch, 1, 3)), 1)  #batch*3*3

    return matrix


def convert_axis_angle_from_matrix_np(R, strict_check=True):
    """Compute axis-angle from rotation matrix.
    This operation is called logarithmic map. Note that there are two possible
    solutions for the rotation axis when the angle is 180 degrees (pi).
    We usually assume active rotations.
    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix
    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.
    Returns
    -------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle). The angle is
        constrained to [0, pi].
    """
    angle = np.arccos((np.trace(R) - 1.0) / 2.0)

    if angle == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])

    a = np.empty(4)

    # We can usually determine the rotation axis by inverting Rodrigues'
    # formula. Subtracting opposing off-diagonal elements gives us
    # 2 * sin(angle) * e,
    # where e is the normalized rotation axis.
    axis_unnormalized = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

    if abs(angle - np.pi) < 1e-4:
        # The threshold is a result from this discussion:
        # https://github.com/rock-learning/pytransform3d/issues/43
        # The standard formula becomes numerically unstable, however,
        # Rodrigues' formula reduces to R = I + 2 (ee^T - I), with the
        # rotation axis e, that is, ee^T = 0.5 * (R + I) and we can find the
        # squared values of the rotation axis on the diagonal of this matrix.
        # We can still use the original formula to reconstruct the signs of
        # the rotation axis correctly.
        a[:3] = np.sqrt(0.5 * (np.diag(R) + 1.0)) * np.sign(axis_unnormalized)
    else:
        a[:3] = axis_unnormalized
        # The norm of axis_unnormalized is 2.0 * np.sin(angle), that is, we
        # could normalize with a[:3] = a[:3] / (2.0 * np.sin(angle)),
        # but the following is much more precise for angles close to 0 or pi:
    axis = a[:3] / np.linalg.norm(a[:3])

    angle = angle
    return axis, angle


def convert_axis_angle_from_matrix(R, strict_check=True):
    """Compute axis-angle from rotation matrix.
    This operation is called logarithmic map. Note that there are two possible
    solutions for the rotation axis when the angle is 180 degrees (pi).
    We usually assume active rotations.
    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix
    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.
    Returns
    -------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle). The angle is
        constrained to [0, pi].
    """
    R_trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
    angle = torch.acos((R_trace - 1.0) / 2.0)

    # We can usually determine the rotation axis by inverting Rodrigues'
    # formula. Subtracting opposing off-diagonal elements gives us
    # 2 * sin(angle) * e,
    # where e is the normalized rotation axis.
    axis_unnormalized = torch.stack([
        R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1]
    ],
                                    dim=-1)
    x_axis = torch.zeros_like(axis_unnormalized)
    x_axis[..., 0] = 1.0
    zero_angle = torch.zeros_like(axis_unnormalized)[..., 0]

    mask = angle.unsqueeze(-1) != 0.0
    axis = axis_unnormalized * mask + x_axis * (~mask)
    angle = angle * mask[..., 0] + zero_angle * (~mask[..., 0])

    mask = (angle.unsqueeze(-1) - np.pi).abs() >= 1e-4
    sign_axis = torch.sqrt(0.5 *
                           (torch.diagonal(R, dim1=-2, dim2=-1) + 1.0)) * (
                               axis_unnormalized).sign()
    axis = axis_unnormalized * mask + sign_axis * (~mask)
    axis = F.normalize(axis, dim=-1)

    return axis, angle


def sphere_polar2cartesian(radius, angles, a=1, b=1, c=1):
    if isinstance(radius, torch.Tensor):
        assert len(radius.shape) == len(angles.shape)
        assert radius.shape[-1] == 1
        radius = radius.squeeze(-1)
    theta = angles[..., 0]
    phi = torch.zeros(
        [1], device=theta.device) if angles.shape[-1] == 1 else angles[..., 1]

    x = a * radius * theta.cos() * phi.cos()
    y = b * radius * theta.sin() * phi.cos()
    coord = [x, y]
    if angles.shape[-1] == 2:
        z = c * phi.sin() * radius
        coord.append(z)
    return torch.stack(coord, axis=-1)


def sphere_cartesian2polar(coord):
    r = ((coord**2).sum(-1) + EPS).sqrt().unsqueeze(-1)
    theta = torch.atan2(coord[..., 1], coord[..., 0]).unsqueeze(-1)
    angles = [theta]
    if coord.shape[-1] == 3:
        phi = torch.atan(coord[..., 2].unsqueeze(-1) * r * theta.sin() /
                         (coord[..., 1].unsqueeze(-1) + EPS))
        angles.append(phi)
    angles = torch.cat(angles, axis=-1)
    return r, angles


def compute_rotation_matrix_from_euler(euler):
    batch = euler.shape[0]
    dim = euler.size(-1)
    if dim == 3:
        c1 = torch.cos(euler[..., 0])  #batch*1
        s1 = torch.sin(euler[..., 0])  #batch*1
        c2 = torch.cos(euler[..., 2])  #batch*1
        s2 = torch.sin(euler[..., 2])  #batch*1
        c3 = torch.cos(euler[..., 1])  #batch*1
        s3 = torch.sin(euler[..., 1])  #batch*1

        row1 = torch.stack((c2 * c3, -s2, c2 * s3), -1).unsqueeze(-2)
        row2 = torch.stack(
            (c1 * s2 * c3 + s1 * s3, c1 * c2, c1 * s2 * s3 - s1 * c3),
            -1).unsqueeze(-2)
        row3 = torch.stack(
            (s1 * s2 * c3 - c1 * s3, s1 * c2, s1 * s2 * s3 + c1 * c3),
            -1).unsqueeze(-2)

        matrix = torch.cat((row1, row2, row3), -2)  #batch*3*3
    elif dim == 1:
        c = torch.cos(euler[..., 0])  #batch*1
        s = torch.sin(euler[..., 0])  #batch*1

        row1 = torch.stack((c, -s), -1).unsqueeze(-2)
        row2 = torch.stack((s, c), -1).unsqueeze(-2)

        matrix = torch.cat((row1, row2), -2)  #batch*3*3
    else:
        raise NotImplementedError

    return matrix
