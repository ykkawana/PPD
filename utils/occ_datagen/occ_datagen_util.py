# %%
import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
import trimesh
import yaml
import dotenv
import tempfile
import numpy as np
import pyrender
from scipy import ndimage
dotenv.load_dotenv()
# import neural_renderer as nr
import os
# from utils import blender_utils
import math
from tqdm import tqdm
import torch
import sys
# sys.path.insert(0, 'external/pyfusion')
# from external import pyfusion
# from utils.occ_datagen import modified_renderer
import mcubes
import subprocess

# padding = 0.1
# up = 1
# image_height = int(640 * up)
# image_width = int(640 * up)
# focal_length_x = int(640 * up)
# focal_length_y = int(640 * up)
# principal_point_x = int(320 * up)
# principal_point_y = int(320 * up)
# depth_offset_factor = 1.5
# resolution = 256
# voxel_size = 1. / resolution
# truncation_factor = 10


# def meshify(mesh,
#             bbs=None,
#             truncation=None,
#             disable_erosion=False,
#             z_axis_offset=0.3,
#             n_views=100,
#             return_raw_depthmap=False,
#             return_depthmap=False):
#     return_dict = return_raw_depthmap or return_depthmap
#     # Scale
#     # Get extents of model.
#     mesh = mesh.copy()
#     if bbs is None:
#         bb_min, bb_max = mesh.vertices.min(0), mesh.vertices.max(0)
#     else:
#         bb_min, bb_max = bbs
#     total_size = (bb_max - bb_min).max()

#     # Set the center (although this should usually be the origin already).
#     centers = (bb_min + bb_max) / 2
#     # Scales all dimensions equally.
#     scale = total_size / (1 - padding)

#     translation = -centers
#     scales_inv = 1 / scale

#     mesh.vertices += translation[np.newaxis, ...]
#     mesh.vertices *= scales_inv

#     t_loc = centers
#     t_scale = scale
#     bb0_min = bb_min
#     bb0_max = bb_max
#     bb1_min, bb1_max = mesh.vertices.min(0), mesh.vertices.max(0)
#     bb1_min = bb1_min
#     bb1_max = bb1_max

#     # %%
#     # Render
#     Rs = []
#     rnd = 1.
#     points = []
#     offset = 2. / n_views
#     increment = math.pi * (3. - math.sqrt(5.))

#     for i in range(n_views):
#         y = ((i * offset) - 1) + (offset / 2)
#         r = math.sqrt(1 - pow(y, 2))

#         phi = ((i + rnd) % n_views) * increment

#         x = math.cos(phi) * r
#         z = math.sin(phi) * r

#         points.append([x, y, z])
#     points = np.array(points)

#     for i in range(points.shape[0]):
#         # https://math.stackexchange.com/questions/1465611/given-a-point-on-a-sphere-how-do-i-find-the-angles-needed-to-point-at-its-ce
#         longitude = -math.atan2(points[i, 0], points[i, 1])
#         latitude = math.atan2(points[i, 2],
#                               math.sqrt(points[i, 0]**2 + points[i, 1]**2))

#         R_x = np.array([[1, 0, 0],
#                         [0, math.cos(latitude), -math.sin(latitude)],
#                         [0, math.sin(latitude),
#                          math.cos(latitude)]])
#         R_y = np.array([[math.cos(longitude), 0,
#                          math.sin(longitude)], [0, 1, 0],
#                         [-math.sin(longitude), 0,
#                          math.cos(longitude)]])

#         R = R_y.dot(R_x)
#         Rs.append(R)

#     depthmaps = []
#     raw_depthmaps = []

#     renderer = modified_renderer.Renderer(near=1 - .75 + z_axis_offset,
#                                           far=1 + .75 + z_axis_offset,
#                                           orig_size=image_height,
#                                           anti_aliasing=False,
#                                           fill_back=False,
#                                           image_size=image_height)
#     fusion_intrisics = np.array([[focal_length_x, 0, principal_point_x],
#                                  [0, focal_length_y, principal_point_y],
#                                  [0, 0, 1]])
#     K = torch.from_numpy(fusion_intrisics).to('cuda').unsqueeze(0).float()
#     vertices = torch.from_numpy(mesh.vertices).to('cuda').unsqueeze(0).float()
#     faces = torch.from_numpy(mesh.faces).to('cuda').unsqueeze(0).float()
#     texture_size = 2
#     textures = torch.ones(faces.shape[1], texture_size, texture_size,
#                           texture_size, 3).to('cuda').unsqueeze(0).float()

#     for i in range(len(Rs)):
#         """
#         rotation_matrix = np.linalg.inv(Rs[i])
#         R = torch.from_numpy(rotation_matrix).to('cuda').unsqueeze(0).float()
#         T = torch.from_numpy(np.array([0., 0., -1])).to('cuda').unsqueeze(0).float()
#         """
#         rotation_matrix = Rs[i]
#         R = torch.from_numpy(rotation_matrix).to('cuda').unsqueeze(0).float()
#         T = torch.from_numpy(np.array([0., 0., 1. + z_axis_offset
#                                        ])).to('cuda').unsqueeze(0).float()

#         faces_inverted = faces[..., [0, 2, 1]]
#         images = renderer(
#             vertices, faces, textures=textures, K=K, R=R, t=T,
#             mode='depth')  # [batch_size, RGB, image_size, image_size]
#         inverted_images = renderer(
#             vertices,
#             faces_inverted,
#             textures=textures,
#             K=K,
#             R=R,
#             t=T,
#             mode='depth')  # [batch_size, RGB, image_size, image_size]
#         images = torch.cat([images, inverted_images],
#                            dim=0).min(0, keepdim=True)[0]
#         raw_depthmap = images.detach().cpu().numpy()[
#             0]  # [image_size, image_size, RGB]
#         # This is mainly result of experimenting.
#         # The core idea is that the volume of the object is enlarged slightly
#         # (by subtracting a constant from the depth map).
#         # Dilation additionally enlarges thin structures (e.g. for chairs).
#         depthmap = raw_depthmap - depth_offset_factor * voxel_size
#         if not disable_erosion:
#             depthmap = ndimage.morphology.grey_erosion(depthmap, size=(3, 3))

#         depthmaps.append(depthmap)
#         if return_raw_depthmap:
#             raw_depthmaps.append(raw_depthmap)

#     # %%
#     # Fusion
#     if truncation is None:
#         truncation = truncation_factor * voxel_size
#     Ks = fusion_intrisics.reshape((1, 3, 3))
#     Ks = np.repeat(Ks, len(depthmaps), axis=0).astype(np.float32)

#     Ts = []
#     for i in range(len(Rs)):
#         Rs[i] = Rs[i]
#         Ts.append(np.array([0, 0, 1. + z_axis_offset]))

#     Ts = np.array(Ts).astype(np.float32)
#     Rs = np.array(Rs).astype(np.float32)

#     depthmaps = np.array(depthmaps).astype(np.float32)
#     views = pyfusion.PyViews(depthmaps, Ks, Rs, Ts)

#     # Note that this is an alias defined as libfusiongpu.tsdf_gpu or libfusioncpu.tsdf_cpu!
#     tsdf = pyfusion.tsdf_gpu(views, resolution, resolution, resolution,
#                              voxel_size, truncation, False)

#     tsdf = np.transpose(tsdf[0], [2, 1, 0])
#     padded_tsdf = np.pad(tsdf, 1, 'constant', constant_values=1e6)
#     vertices, triangles = mcubes.marching_cubes(-padded_tsdf, 0)
#     # Remove padding offset
#     vertices -= 1
#     # Normalize to [-0.5, 0.5]^3 cube
#     vertices /= resolution
#     vertices -= 0.5

#     wmesh_normalized = trimesh.Trimesh(vertices, triangles)
#     vertices = t_loc + t_scale * vertices

#     wmesh = trimesh.Trimesh(vertices, triangles)
#     ret = {}
#     ret['mesh'] = wmesh
#     ret['mesh_normalized'] = wmesh_normalized
#     ret['tsdf'] = tsdf

#     if return_dict:
#         ret_dict = {'ret': ret}
#         if return_raw_depthmap:
#             ret_dict['raw_depthmaps'] = raw_depthmaps
#         if return_depthmap:
#             ret_dict['depthmaps'] = depthmaps
#         return ret_dict
#     else:
#         return ret


# def simplify_mesh(inmesh):
#     with tempfile.NamedTemporaryFile(
#             suffix='.obj') as fin, tempfile.NamedTemporaryFile(
#                 suffix='.obj') as fout:
#         inmesh.export(fin.name)
#         mlxpath = os.path.join(os.path.dirname(__file__), 'simplification.mlx')
#         command = 'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i {inpath} -o {outpath} -s {mlxpath}'.format(
#             inpath=fin.name, outpath=fout.name, mlxpath=mlxpath)
#         subprocess.run(command, shell=True)
#         mesh = trimesh.load(fout.name)
#     if isinstance(mesh, trimesh.Scene):
#         meshes = mesh.dump()
#         if len(meshes) < 1:
#             return inmesh
#         mesh = trimesh.util.concatenate(meshes)
#     return mesh


def occ2mesh(occ, th, res):
    padded = np.pad(occ, 1, 'constant', constant_values=-1e6)
    vertices, triangles = mcubes.marching_cubes(padded, th)
    if vertices.shape[0] < 3 or triangles.shape[0] < 1:
        return None
    vertices -= 1
    vertices = (vertices.astype(np.float32) - 0.5) / res - 0.5
    m = trimesh.Trimesh(vertices, triangles)
    trimesh.repair.fill_holes(m)
    trimesh.repair.fix_winding(m)
    trimesh.repair.fix_normals(m)
    trimesh.repair.fix_inversion(m)
    return m
