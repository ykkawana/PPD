# %%
import numpy as np
import os

import torch
from torch.nn import functional as F
from torch.utils import data
import random
from utils import common
import trimesh


# %%
class IMNetShapeNetDataset(data.Dataset):
    def __init__(self,
                 primitive_num=None,
                 list_path=None,
                 data_dir=None,
                 classes=[],
                 subset_num=None,
                 voxels_filename=None,
                 voxel_resolution=64,
                 points_values_filename=None,
                 points_values_subsample_num=-1,
                 is_imnet_res64_sampling=False,
                 pixels_filename=None,
                 surface_points_filename=None,
                 surface_points_subsample_num=-1,
                 volume_points_filename=None,
                 part_surface_points_filename=None,
                 part_surface_points_subsample_num=-1,
                 part_surface_points_subsample_replace=False,
                 points_tsdf_filename=None,
                 points_tsdf_subsample_num=-1,
                 is_imnet_res64_sampling_tsdf=False,
                 use_tsdf_opposite_sign=False,
                 downsample_tsdf_voxel_resolution=16,
                 param_filename=None,
                 randomly_rot_x=False):
        assert data_dir is not None
        assert list_path is not None
        self.primitive_num = primitive_num
        self.data_dir = data_dir
        self.voxels_filename = voxels_filename
        self.voxel_resolution = voxel_resolution
        self.points_values_filename = points_values_filename
        self.pixels_filename = pixels_filename
        self.surface_points_filename = surface_points_filename
        self.surface_points_subsample_num = surface_points_subsample_num
        self.volume_points_filename = volume_points_filename
        self.is_imnet_res64_sampling = is_imnet_res64_sampling
        self.points_values_subsample_num = points_values_subsample_num
        self.part_surface_points_filename = part_surface_points_filename
        self.part_surface_points_subsample_num = part_surface_points_subsample_num
        self.part_surface_points_subsample_replace = part_surface_points_subsample_replace
        self.points_tsdf_filename = points_tsdf_filename
        self.points_tsdf_subsample_num = points_tsdf_subsample_num
        self.is_imnet_res64_sampling_tsdf = is_imnet_res64_sampling_tsdf
        self.use_tsdf_opposite_sign = use_tsdf_opposite_sign
        self.downsample_tsdf_voxel_resolution = downsample_tsdf_voxel_resolution
        self.randomly_rot_x = randomly_rot_x

        self.param_filename = param_filename
        if self.param_filename is not None:
            assert primitive_num is not None

        self.classes = classes
        self.shapenet_supported_classes = [
            '02691156', '03001627', '03636649', '04379243'
        ]
        self.shape2motion_supported_classes = [
            'oven', 'eyeglasses', 'washing_machine', 'laptop', 'drawer'
        ]
        self.class_id_to_part_ids = {
            # id in partnet to id in here
            '02691156': {
                0: 0,
                1: 1,
                2: 2,
                3: 3
            },
            '03001627': {
                12: 0,
                13: 1,
                14: 2,
                15: 2
            },
            '04379243': {
                47: 0,
                48: 1,
                49: 1
            },
            '03636649': {
                24: 0,
                25: 1,
                26: 2,
                27: 2
            }
        }
        annotated_classes = set(self.shapenet_supported_classes)
        self.is_shapenet_dataset = False
        if set(self.classes).issubset(annotated_classes):
            self.is_shapenet_dataset = True

        annotated_classes = set(self.shape2motion_supported_classes)
        self.is_shape2motion_dataset = False
        if set(self.classes).issubset(annotated_classes):
            self.is_shape2motion_dataset = True

        if self.part_surface_points_filename is not None:
            assert self.classes

        if isinstance(list_path, list):
            assert not self.is_shapenet_dataset
            self.class_id_modelname = []
            for lp in list_path:
                self.class_id_modelname.extend(
                    [l.strip().split('/') for l in open(lp).readlines()])
            self.class_id_modelname = [[l[0], l[1] + '/' + l[2]]
                                       for l in self.class_id_modelname]
        else:
            if self.is_shapenet_dataset:
                self.class_id_modelname = [
                    l.strip().split('/') for l in open(list_path).readlines()
                ]
            elif self.is_shape2motion_dataset:
                self.class_id_modelname = [
                    l.strip().split('/') for l in open(list_path).readlines()
                ]
                self.class_id_modelname = [[l[0], l[1] + '/' + l[2]]
                                           for l in self.class_id_modelname]
            else:
                raise NotImplementedError

        if self.classes:
            self.class_id_modelname = [
                pair for pair in self.class_id_modelname if pair[0] in classes
            ]

        if subset_num:
            self.class_id_modelname = self.class_id_modelname[:subset_num]
        self.len = len(self.class_id_modelname)

        range_0 = range(self.downsample_tsdf_voxel_resolution)
        range_1 = range(self.downsample_tsdf_voxel_resolution)
        range_2 = range(self.downsample_tsdf_voxel_resolution)

        Y, X, Z = np.meshgrid(range_0, range_1, range_2, indexing='ij')
        Y = (Y.astype(np.float32) +
             0.5) / self.downsample_tsdf_voxel_resolution - 0.5
        X = (X.astype(np.float32) +
             0.5) / self.downsample_tsdf_voxel_resolution - 0.5
        Z = (Z.astype(np.float32) +
             0.5) / self.downsample_tsdf_voxel_resolution - 0.5
        self.sample_grid = np.stack(
            [X.flatten(), Y.flatten(), Z.flatten()], axis=-1)

        self.rotx_matrix = trimesh.transformations.rotation_matrix(
            -np.pi / 2, [1, 0, 0])[:3, :3][np.newaxis, ...].astype(np.float32)

        Y, X, Z = np.meshgrid(range_0, range_1, range_2, indexing='ij')
        Y = (Y.astype(np.float32) + 0.5) / self.voxel_resolution - 0.5
        X = (X.astype(np.float32) + 0.5) / self.voxel_resolution - 0.5
        Z = (Z.astype(np.float32) + 0.5) / self.voxel_resolution - 0.5
        self.voxel_grid = np.stack(
            [X.flatten(), Y.flatten(), Z.flatten()], axis=-1)

    def get_data(self, tp):
        class_id, modelname, in_class_id = tp
        modelname_in_class_id = modelname + '/' + in_class_id
        for idx in range(len(self)):
            cls_id, mdname = self.class_id_modelname[idx]
            if (cls_id, mdname) == (class_id, modelname_in_class_id):
                return self[idx]


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        binary_flag = random.uniform(0, 1) >= 0.5
        class_id, modelname = self.class_id_modelname[idx]
        model_dir = os.path.join(self.data_dir, class_id, modelname)

        data = {}
        data['class_id'] = class_id
        data['modelname'] = modelname
        if self.voxels_filename:
            voxels_filepath = os.path.join(model_dir, self.voxels_filename)
            voxels_data = np.load(voxels_filepath)
            voxels = voxels_data['voxels'].astype(np.float32)
            voxels = voxels.reshape(
                [1, voxels.shape[0], voxels.shape[1], voxels.shape[2]])
            down_time = int(voxels.shape[-1] // self.voxel_resolution // 2)
            voxels_tmp = torch.from_numpy(voxels).view(1, 64, 64, 64)
            for _ in range(down_time):
                voxels_tmp = F.max_pool3d(voxels_tmp, 2)
            data['voxels'] = voxels_tmp
            data['voxel_grid'] = self.voxel_grid.copy()

        if self.points_values_filename:
            points_values_filepath = os.path.join(model_dir,
                                                  self.points_values_filename)
            points_values_data = np.load(points_values_filepath)
            points = (points_values_data['points'].astype(np.float32) +
                      0.5) / 256 - 0.5

            values = points_values_data['values'].astype(np.float32)

            if self.is_imnet_res64_sampling:
                point_batch_num = 4
                point_batch_size = 4096
                assert points.shape[0] == point_batch_num * point_batch_size
                which_batch = np.random.randint(point_batch_num)
                points = points[which_batch *
                                point_batch_size:(which_batch + 1) *
                                point_batch_size, :]
                values = values[which_batch *
                                point_batch_size:(which_batch + 1) *
                                point_batch_size, :]
            if class_id == 'oven' and modelname.startswith('0015'):
                shape = values.shape
                tmp = values.reshape(-1)
                tmp[tmp == 3] = 1
                values = tmp.reshape(*shape)
            if self.points_values_subsample_num > 0:
                points, selected = common.subsample_points(
                    points,
                    self.points_values_subsample_num,
                    return_index=True)
                values = common.subsample_points(
                    values, self.points_values_subsample_num, index=selected)
            data['values'] = values[:, 0]
            if self.randomly_rot_x and binary_flag:
                points = np.matmul(self.rotx_matrix,
                                   points[..., np.newaxis])[..., 0]
            data['points'] = points

        if self.part_surface_points_filename:
            part_surface_points_filepath = os.path.join(
                model_dir, self.part_surface_points_filename)
            part_surface_points_data = np.load(part_surface_points_filepath)
            part_surface_points = part_surface_points_data['points'].astype(
                np.float32)
            part_surface_points = part_surface_points[:, [2, 1, 0]]
            part_surface_points[:, 2] = -part_surface_points[:, 2]
            part_surface_labels = part_surface_points_data['labels'].astype(
                np.float32)
            if self.part_surface_points_subsample_num > 0:
                part_surface_points, selected = common.subsample_points(
                    part_surface_points,
                    self.part_surface_points_subsample_num,
                    return_index=True,
                    replace=self.part_surface_points_subsample_replace)
                part_surface_labels = common.subsample_points(
                    part_surface_labels,
                    self.part_surface_points_subsample_num,
                    index=selected)

            merged_surface_labels = -np.ones_like(part_surface_labels)
            for original_id, merged_id in self.class_id_to_part_ids[
                    class_id].items():
                merged_surface_labels[part_surface_labels ==
                                      original_id] = merged_id + 1
            assert np.all(merged_surface_labels >= 0)
            if self.randomly_rot_x and binary_flag:
                part_surface_points = np.matmul(
                    self.rotx_matrix, part_surface_points[...,
                                                          np.newaxis])[..., 0]
            data['part_surface_points'] = part_surface_points
            data['part_surface_labels'] = merged_surface_labels

        if self.param_filename:
            param_filepath = os.path.join(model_dir, self.param_filename)
            raw_param = np.load(param_filepath)

            if 'rotation_primitives_num' in raw_param:
                rotation_primitives_num = raw_param[
                    'rotation_primitives_num'].item()
            else:
                rotation_primitives_num = 0
            if 'translation_primitives_num' in raw_param:
                translation_primitives_num = raw_param[
                    'translation_primitives_num'].item()
            else:
                translation_primitives_num = 0

            pad_rotation_primitives_num = self.primitive_num - rotation_primitives_num - translation_primitives_num

            rotation_anchor_points = []
            rotation_directions = []
            rotation_amount_maxs = []
            rotation_amount_mins = []
            rotation_amounts = []
            for idx in range(rotation_primitives_num):
                rotation_anchor_points.append(
                    raw_param['rotation_anchor_point_{}'.format(idx)])
                rotation_directions.append(
                    raw_param['rotation_direction_{}'.format(idx)])
                rotation_amount_maxs.append(
                    raw_param['rotation_amount_max_{}'.format(idx)])
                rotation_amount_mins.append(
                    raw_param['rotation_amount_min_{}'.format(idx)])
                rotation_amounts.append(
                    raw_param['rotation_amount_{}'.format(idx)])
            for idx in range(translation_primitives_num):
                rotation_anchor_points.append(
                    raw_param['translation_anchor_point_{}'.format(idx)])
                rotation_directions.append(
                    raw_param['translation_direction_{}'.format(idx)])
                rotation_amount_maxs.append(
                    raw_param['translation_amount_max_{}'.format(idx)])
                rotation_amount_mins.append(
                    raw_param['translation_amount_min_{}'.format(idx)])
                rotation_amounts.append(
                    raw_param['translation_amount_{}'.format(idx)])
            for idx in range(pad_rotation_primitives_num):
                rotation_anchor_points.append(
                    np.zeros_like(rotation_anchor_points[0]))
                rotation_directions.append(
                    np.zeros_like(rotation_directions[0]))
                rotation_amount_maxs.append(
                    np.zeros_like(rotation_amount_maxs[0]))
                rotation_amount_mins.append(
                    np.zeros_like(rotation_amount_mins[0]))
                rotation_amounts.append(np.zeros_like(rotation_amounts[0]))

            primitive_type = [0]
            primitive_type.extend([1] * rotation_primitives_num)
            primitive_type.extend([2] * translation_primitives_num)
            primitive_type.extend(
                [-1] * (self.primitive_num - 1 - rotation_primitives_num -
                        translation_primitives_num))

            param_rotation_anchor_point = np.stack(
                rotation_anchor_points).astype(np.float32)
            if self.randomly_rot_x and binary_flag:
                param_rotation_anchor_point = np.matmul(
                    self.rotx_matrix,
                    param_rotation_anchor_point[..., np.newaxis])[..., 0]
            data['param_anchor_point'] = param_rotation_anchor_point

            param_rotation_direction = np.stack(
                rotation_directions).astype(np.float32)
            if self.randomly_rot_x and binary_flag:
                param_rotation_direction = np.matmul(
                    self.rotx_matrix,
                    param_rotation_direction[..., np.newaxis])[..., 0]
            data['param_direction'] = param_rotation_direction

            data['param_amount_max'] = np.stack(
                rotation_amount_maxs).astype(np.float32)
            data['param_amount_min'] = np.stack(
                rotation_amount_mins).astype(np.float32)
            data['param_amount'] = np.stack(rotation_amounts).astype(
                np.float32)
            data['param_primitive_type_offset'] = np.array([1]).astype(
                np.float32)
            data['param_primitive_type'] = np.array(primitive_type).astype(
                np.float32)

        if self.surface_points_filename:
            surface_points_filepath = os.path.join(
                model_dir, self.surface_points_filename)
            surface_points_data = np.load(surface_points_filepath)
            surface_points = surface_points_data['points'].astype(np.float32)
            surface_normals = surface_points_data['normals'].astype(np.float32)
            if self.surface_points_subsample_num > 0:
                surface_points, selected = common.subsample_points(
                    surface_points,
                    self.surface_points_subsample_num,
                    return_index=True)
                surface_normals = common.subsample_points(
                    surface_normals,
                    self.surface_points_subsample_num,
                    index=selected)

            if self.randomly_rot_x and binary_flag:
                surface_points = np.matmul(self.rotx_matrix,
                                           surface_points[...,
                                                          np.newaxis])[..., 0]
            data['surface_points'] = surface_points
            if self.randomly_rot_x and binary_flag:
                surface_normals = np.matmul(self.rotx_matrix,
                                            surface_normals[...,
                                                            np.newaxis])[...,
                                                                         0]
            data['surface_normals'] = surface_normals

        if self.points_tsdf_filename:
            points_tsdf_filepath = os.path.join(model_dir,
                                                self.points_tsdf_filename)
            points_tsdf_data = np.load(points_tsdf_filepath)

            is_voxelized_tsdf_file = points_tsdf_data['tsdf'].shape == (64, 64,
                                                                        64, 1)
            if is_voxelized_tsdf_file:
                div = int(64 / self.downsample_tsdf_voxel_resolution)
                tsdf = -torch.nn.functional.max_pool3d(torch.from_numpy(
                    -points_tsdf_data['tsdf'].astype(np.float32)).view(
                        1, 64, 64, 64),
                                                       kernel_size=div).view(
                                                           -1, 1).numpy()
                n_parts = 0
                part_tsdfs = []
                while True:
                    if 'tsdf_{}'.format(n_parts) in points_tsdf_data:

                        part_tsdfs.append(points_tsdf_data['tsdf_{}'.format(
                            n_parts)].reshape([1, 1, 64, 64, 64]))

                        n_parts += 1
                    else:
                        break
                part_tsdfs = np.concatenate(part_tsdfs, axis=1)
                part_tsdfs = torch.from_numpy(part_tsdfs.astype(
                    np.float32)).view(1, n_parts, 64, 64, 64)
                tsdf_values, values = torch.nn.functional.max_pool3d(
                    -part_tsdfs, kernel_size=div).max(1)
                values += 1
                tsdf_values = -tsdf_values
                values *= (tsdf_values <= 0)
                values = values.view(-1, 1).numpy()
                points = self.sample_grid

            else:
                points = (points_tsdf_data['points'].astype(np.float32) +
                          0.5) / 256 - 0.5

                values = points_tsdf_data['values'].astype(np.float32)
                tsdf = points_tsdf_data['tsdf'].astype(np.float32)

            if self.is_imnet_res64_sampling_tsdf:
                point_batch_num = 4
                point_batch_size = 4096
                assert points.shape[0] == point_batch_num * point_batch_size
                which_batch = np.random.randint(point_batch_num)
                points = points[which_batch *
                                point_batch_size:(which_batch + 1) *
                                point_batch_size, :]
                values = values[which_batch *
                                point_batch_size:(which_batch + 1) *
                                point_batch_size, :]
                tsdf = tsdf[which_batch * point_batch_size:(which_batch + 1) *
                            point_batch_size, :]
            if class_id == 'oven' and modelname.startswith('0015'):
                shape = values.shape
                tmp = values.reshape(-1)
                tmp[tmp == 3] = 1
                values = tmp.reshape(*shape)
            if self.points_tsdf_subsample_num > 0:
                points, selected = common.subsample_points(
                    points, self.points_tsdf_subsample_num, return_index=True)
                values = common.subsample_points(
                    values, self.points_tsdf_subsample_num, index=selected)
                tsdf = common.subsample_points(tsdf,
                                               self.points_tsdf_subsample_num,
                                               index=selected)
            data['values'] = values[:, 0]
            if self.randomly_rot_x and binary_flag:
                points = np.matmul(self.rotx_matrix,
                                   points[..., np.newaxis])[..., 0]
            data['points'] = points
            if self.use_tsdf_opposite_sign:
                tsdf = -tsdf
            data['tsdf'] = tsdf[:, 0]

        return data
