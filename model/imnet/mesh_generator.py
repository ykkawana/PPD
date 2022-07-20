from collections import defaultdict
#from external.libmise import MISE
import mcubes
import numpy as np
import trimesh
import torch
import math
from utils import eval_utils
from utils.occ_datagen import occ_datagen_util


class MeshGenerator:
    def __init__(self,
                 model,
                 device,
                 trainer_ob,
                 gen_type='default',
                 resolution=64,
                 sampling_max_points=4096,
                 skip_mesh_gen=False,
                 threshold=0.5):
        self.trainer_ob = trainer_ob
        self.device = device
        self.model = model
        self.threshold = threshold
        self.gen_type = gen_type
        self.resolution = resolution
        self.sampling_max_points = sampling_max_points
        assert self.gen_type in ['default', 'imnet']
        self.skip_mesh_gen = skip_mesh_gen
        if self.skip_mesh_gen:
            assert self.gen_type == 'default'

        assert not self.model.is_expand_rotation_for_euler_angle

        self.prepare_grid_coords()

    def generate_mesh(self, data):
        ''' Generates the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        if self.gen_type == 'default':
            meshes = self.generate_mesh_default(data)
        else:
            raise NotImplementedError
        # elif self.gen_type == 'imnet':
        #     meshes = self.generate_mesh_imnet(inputs)
        return meshes

    def generate_latent(self, data):
        if self.trainer_ob.model_input_type == 'voxel':
            inputs = data['voxels'].to(self.device)
        elif self.trainer_ob.model_input_type == 'surface_points':
            inputs = data['surface_points'].to(self.device)
        else:
            raise NotImplementedError

        batch_size = inputs.shape[0]
        model_floats = np.zeros([
            batch_size, self.resolution, self.resolution, self.resolution,
            self.model.primitive_num
        ],
                                dtype=np.float32).reshape(
                                    [batch_size, -1, self.model.primitive_num])
        coord_sample_num = self.coord.shape[0]
        split_num = math.ceil(coord_sample_num / self.sampling_max_points)

        with torch.no_grad():
            ret = self.model(inputs,
                             None,
                             return_paramnet_kwargs=True,
                             return_occupancy=False)
        retcs = []
        for bidx in range(batch_size):
            ret_c = {}
            for key in [
                    'latent', 'motion_latent', 'generator_kwargs',
                    'paramnet_kwargs'
            ]:
                if key == 'generator_kwargs':
                    ret_c['shape_template_latent'] = ret[key][
                        'shape_template_latent'][bidx,
                                                 ...].detach().cpu().numpy()
                elif key == 'paramnet_kwargs':
                    ret_c['motion_template_latent'] = ret[key][
                        'motion_template_latent'][bidx,
                                                  ...].detach().cpu().numpy()
                else:
                    ret_c[key] = ret[key][bidx, ...].detach().cpu().numpy()
            ret_dict = {'latent.npz': ret_c}
            retcs.append(ret_dict)
        return retcs

    def generate_mesh_default(self, data):
        if self.trainer_ob.model_input_type == 'voxel':
            inputs = data['voxels'].to(self.device)
        elif self.trainer_ob.model_input_type == 'surface_points':
            inputs = data['surface_points'].to(self.device)
        else:
            raise NotImplementedError

        batch_size = inputs.shape[0]
        model_floats = np.zeros([
            batch_size, self.resolution, self.resolution, self.resolution,
            self.model.primitive_num
        ],
                                dtype=np.float32).reshape(
                                    [batch_size, -1, self.model.primitive_num])
        coord_sample_num = self.coord.shape[0]
        split_num = math.ceil(coord_sample_num / self.sampling_max_points)

        with torch.no_grad():
            ret = self.model(inputs, None, return_occupancy=False)
        if self.trainer_ob.disable_induction_by_moving:
            pred_params = None
        else:
            pred_params = ret['param']

        meshes = []
        for bidx in range(batch_size):
            meshess = defaultdict(lambda: {})
            meshess['gt'] = {
                key: value[bidx, ...].detach().cpu().numpy()
                for key, value in data.items() if key.startswith('param')
            }
            for key in ['param', 'raw_param']:
                if key not in ret:
                    continue
                meshess['pred'][key] = eval_utils.nested_batched_tensor2tensor(
                    ret[key], bidx, batch_size, to_np=True)
            param_all = {'pred_params.pkl': dict(meshess)}
            meshes.append(param_all)
        if self.skip_mesh_gen:
            return meshes

        pred_latent = ret['latent']
        pred_generator_kwargs = ret['generator_kwargs']

        with torch.no_grad():
            for idx in range(split_num):
                start_idx = idx * self.sampling_max_points
                end_idx = min((idx + 1) * self.sampling_max_points,
                              coord_sample_num)
                sliced_coord = self.coord[start_idx:end_idx, :].unsqueeze(
                    0).expand(batch_size, -1, -1)

                ret = self.model(inputs, sliced_coord)
                pred_values = ret['occupancy']

                self.trainer_ob.preprocess_transformation(
                    ret, sliced_coord, sliced_coord, pred_values, pred_params,
                    pred_latent, pred_generator_kwargs)
                moved_values = self.trainer_ob.get_moved_occupancy_value(
                    pred_latent,
                    sliced_coord, {'occupancy': pred_values},
                    pred_params,
                    generator_kwargs=pred_generator_kwargs)['occupancy']
                model_floats[:, start_idx:end_idx, :] = torch.sigmoid(
                    moved_values).cpu().numpy()
        model_floats = model_floats.reshape([
            batch_size, self.resolution, self.resolution, self.resolution,
            self.model.primitive_num
        ])
        for bidx in range(model_floats.shape[0]):
            meshess = {}
            whole_occ = model_floats[bidx, ...].max(-1)
            whole_mesh = occ_datagen_util.occ2mesh(whole_occ, self.threshold,
                                                   self.resolution)
            if whole_mesh is not None:
                meshess['whole_mesh.obj'] = whole_mesh

            for pidx in range(model_floats.shape[-1]):
                part_occ = model_floats[bidx, :, :, :, pidx]
                part_mask = model_floats[bidx, ...].argmax(-1) == pidx
                part_occ = part_occ * part_mask + (~part_mask) * (-1e6)
                part_mesh = occ_datagen_util.occ2mesh(part_occ, self.threshold,
                                                      self.resolution)
                if part_mesh is not None:
                    meshess['part_mesh_{}.obj'.format(pidx)] = part_mesh
            try:
                meshes[bidx].update(meshess)
            except IndexError:
                meshes.append(meshess)
        return meshes

    def generate_mesh_imnet(self, inputs):
        model_floats = self.z2voxel(inputs)

        batch_size = inputs.shape[0]
        meshes = []
        for bidx in range(model_floats.shape[0]):
            scene = trimesh.Scene()
            for pidx in range(model_floats.shape[-1]):
                vertices, triangles = mcubes.marching_cubes(
                    model_floats[bidx, :, :, :, pidx], self.threshold)
                vertices = (vertices.astype(np.float32) +
                            0.5) / self.real_size - 0.5
                if vertices.shape[0] < 3 or triangles.shape[0] < 1:
                    continue
                part_mesh = trimesh.Trimesh(vertices, triangles)
                scene.add_geometry(part_mesh)
            meshes.append(scene)
        return meshes

    def z2voxel(self, inputs):

        batch_size = inputs.shape[0]
        dimc = self.cell_grid_size
        dimf = self.frame_grid_size

        frame_flag = np.zeros([
            batch_size, dimf + 2, dimf + 2, dimf + 2, self.model.primitive_num
        ], np.uint8)

        frame_batch_num = int(dimf**3 / self.test_point_batch_size)
        assert frame_batch_num > 0

        #get frame grid values
        for i in range(frame_batch_num):
            point_coord = self.frame_coords[i *
                                            self.test_point_batch_size:(i +
                                                                        1) *
                                            self.test_point_batch_size]
            point_coord = np.expand_dims(point_coord, axis=0)
            point_coord = torch.from_numpy(point_coord)
            point_coord = point_coord.to(self.device).expand(
                batch_size, -1, -1)
            with torch.no_grad():
                ret = self.model(inputs, point_coord)
                if self.trainer_ob.disable_induction_by_moving:
                    pred_params = None
                else:
                    pred_params = ret['param']

                pred_latent = ret['latent']
                pred_values = ret['occupancy']
                pred_generator_kwargs = ret['generator_kwargs']

                self.trainer_ob.preprocess_transformation(
                    ret, point_coord, point_coord, pred_values, pred_params,
                    pred_latent, pred_generator_kwargs)
                model_out_ = self.trainer_ob.get_moved_occupancy_value(
                    pred_latent,
                    point_coord, {'occupancy': pred_values},
                    pred_params,
                    generator_kwargs=pred_generator_kwargs)['occupancy']

            model_out = model_out_.detach().cpu().numpy()
            x_coords = self.frame_x[i * self.test_point_batch_size:(i + 1) *
                                    self.test_point_batch_size]
            y_coords = self.frame_y[i * self.test_point_batch_size:(i + 1) *
                                    self.test_point_batch_size]
            z_coords = self.frame_z[i * self.test_point_batch_size:(i + 1) *
                                    self.test_point_batch_size]
            for pidx in range(self.model.primitive_num):
                frame_flag[:, x_coords + 1, y_coords + 1, z_coords + 1,
                           pidx] = np.reshape(
                               ((model_out[..., pidx] > self.threshold) &
                                (np.argmax(model_out, axis=-1)
                                 == pidx)).astype(np.uint8),
                               [batch_size, self.test_point_batch_size])

        model_float_batch = []
        for bidx in range(batch_size):
            if bidx == 2:
                break
            model_float_pm = []
            for pidx in range(self.model.primitive_num):
                print('{}th batch, {}th primitive'.format(bidx, pidx))
                #get queue and fill up ones
                model_float = np.zeros([
                    self.real_size + 2, self.real_size + 2, self.real_size + 2
                ], np.float32)
                queue = []
                for i in range(1, dimf + 1):
                    for j in range(1, dimf + 1):
                        for k in range(1, dimf + 1):
                            maxv = np.max(frame_flag[bidx, i - 1:i + 2,
                                                     j - 1:j + 2, k - 1:k + 2,
                                                     pidx])
                            minv = np.min(frame_flag[bidx, i - 1:i + 2,
                                                     j - 1:j + 2, k - 1:k + 2,
                                                     pidx])
                            if maxv != minv:
                                queue.append((i, j, k))
                            elif maxv == 1:
                                x_coords = self.cell_x + (i - 1) * dimc
                                y_coords = self.cell_y + (j - 1) * dimc
                                z_coords = self.cell_z + (k - 1) * dimc
                                model_float[x_coords + 1, y_coords + 1,
                                            z_coords + 1] = 1.0

                cell_batch_size = dimc**3
                cell_batch_num = int(self.test_point_batch_size /
                                     cell_batch_size)
                assert cell_batch_num > 0
                #run queue
                while len(queue) > 0:
                    batch_num = min(len(queue), cell_batch_num)
                    point_list = []
                    cell_coords = []
                    for i in range(batch_num):
                        point = queue.pop(0)
                        point_list.append(point)
                        cell_coords.append(self.cell_coords[point[0] - 1,
                                                            point[1] - 1,
                                                            point[2] - 1])
                    cell_coords = np.concatenate(cell_coords, axis=0)
                    cell_coords = np.expand_dims(cell_coords, axis=0)
                    cell_coords = torch.from_numpy(cell_coords)
                    cell_coords = cell_coords.to(self.device).expand(
                        batch_size, -1, -1)
                    with torch.no_grad():
                        ret = self.model(inputs, cell_coords)
                        if self.trainer_ob.disable_induction_by_moving:
                            pred_params = None
                        else:
                            pred_params = ret['param']

                        pred_latent = ret['latent']
                        pred_values = ret['occupancy']
                        pred_generator_kwargs = ret['generator_kwargs']

                        self.trainer_ob.preprocess_transformation(
                            ret, cell_coords, cell_coords, pred_values,
                            pred_params, pred_latent, pred_generator_kwargs)
                        model_out_batch_ = self.trainer_ob.get_moved_occupancy_value(
                            pred_latent,
                            cell_coords, {'occupancy': pred_values},
                            pred_params,
                            generator_kwargs=pred_generator_kwargs
                        )['occupancy']

                    model_out_batch = model_out_batch_.detach().cpu().numpy(
                    )[0]
                    for i in range(batch_num):
                        point = point_list[i]
                        model_out = model_out_batch[i *
                                                    cell_batch_size:(i + 1) *
                                                    cell_batch_size, 0]
                        x_coords = self.cell_x + (point[0] - 1) * dimc
                        y_coords = self.cell_y + (point[1] - 1) * dimc
                        z_coords = self.cell_z + (point[2] - 1) * dimc
                        model_float[x_coords + 1, y_coords + 1,
                                    z_coords + 1] = model_out

                        if np.max(model_out) > self.threshold:
                            for i in range(-1, 2):
                                pi = point[0] + i
                                if pi <= 0 or pi > dimf: continue
                                for j in range(-1, 2):
                                    pj = point[1] + j
                                    if pj <= 0 or pj > dimf: continue
                                    for k in range(-1, 2):
                                        pk = point[2] + k
                                        if pk <= 0 or pk > dimf: continue
                                        if (frame_flag[bidx, pi, pj, pk,
                                                       pidx] == 0):
                                            frame_flag[bidx, pi, pj, pk,
                                                       pidx] = 1
                                            queue.append((pi, pj, pk))
                model_float_pm.append(model_float)
            model_float_batch.append(np.stack(model_float_pm, axis=-1))
        return np.stack(model_float_batch, axis=0)

    def prepare_grid_coords(self):
        if self.gen_type == 'default':
            range_0 = range(self.resolution)
            range_1 = range(self.resolution)
            range_2 = range(self.resolution)

            Y, X, Z = np.meshgrid(range_0, range_1, range_2, indexing='ij')
            self.coord = np.stack(
                [Y.flatten(), X.flatten(),
                 Z.flatten()], axis=-1)

            self.coord = (self.coord.astype(np.float32) +
                          0.5) / self.resolution - 0.5
            self.coord = torch.from_numpy(self.coord).to(self.device)
        elif self.gen_type == 'imnet':
            #keep everything a power of 2
            self.cell_grid_size = 4
            self.frame_grid_size = 64
            self.real_size = self.cell_grid_size * self.frame_grid_size  #=256, output point-value voxel grid size in testing
            self.test_size = 32  #related to testing batch_size, adjust according to gpu memory size
            self.test_point_batch_size = self.test_size * self.test_size * self.test_size  #do not change

            #get coords for training
            dima = self.test_size
            dim = self.frame_grid_size
            self.aux_x = np.zeros([dima, dima, dima], np.uint8)
            self.aux_y = np.zeros([dima, dima, dima], np.uint8)
            self.aux_z = np.zeros([dima, dima, dima], np.uint8)
            multiplier = int(dim / dima)
            multiplier2 = multiplier * multiplier
            multiplier3 = multiplier * multiplier * multiplier
            for i in range(dima):
                for j in range(dima):
                    for k in range(dima):
                        self.aux_x[i, j, k] = i * multiplier
                        self.aux_y[i, j, k] = j * multiplier
                        self.aux_z[i, j, k] = k * multiplier
            self.coords = np.zeros([multiplier3, dima, dima, dima, 3],
                                   np.float32)
            for i in range(multiplier):
                for j in range(multiplier):
                    for k in range(multiplier):
                        self.coords[i * multiplier2 + j * multiplier +
                                    k, :, :, :, 0] = self.aux_x + i
                        self.coords[i * multiplier2 + j * multiplier +
                                    k, :, :, :, 1] = self.aux_y + j
                        self.coords[i * multiplier2 + j * multiplier +
                                    k, :, :, :, 2] = self.aux_z + k
            self.coords = (self.coords.astype(np.float32) + 0.5) / dim - 0.5
            self.coords = np.reshape(
                self.coords, [multiplier3, self.test_point_batch_size, 3])
            self.coords = torch.from_numpy(self.coords)
            self.coords = self.coords.to(self.device)

            #get coords for testing
            dimc = self.cell_grid_size
            dimf = self.frame_grid_size
            self.cell_x = np.zeros([dimc, dimc, dimc], np.int32)
            self.cell_y = np.zeros([dimc, dimc, dimc], np.int32)
            self.cell_z = np.zeros([dimc, dimc, dimc], np.int32)
            self.cell_coords = np.zeros(
                [dimf, dimf, dimf, dimc, dimc, dimc, 3], np.float32)
            self.frame_coords = np.zeros([dimf, dimf, dimf, 3], np.float32)
            self.frame_x = np.zeros([dimf, dimf, dimf], np.int32)
            self.frame_y = np.zeros([dimf, dimf, dimf], np.int32)
            self.frame_z = np.zeros([dimf, dimf, dimf], np.int32)
            for i in range(dimc):
                for j in range(dimc):
                    for k in range(dimc):
                        self.cell_x[i, j, k] = i
                        self.cell_y[i, j, k] = j
                        self.cell_z[i, j, k] = k
            for i in range(dimf):
                for j in range(dimf):
                    for k in range(dimf):
                        self.cell_coords[i, j, k, :, :, :,
                                         0] = self.cell_x + i * dimc
                        self.cell_coords[i, j, k, :, :, :,
                                         1] = self.cell_y + j * dimc
                        self.cell_coords[i, j, k, :, :, :,
                                         2] = self.cell_z + k * dimc
                        self.frame_coords[i, j, k, 0] = i
                        self.frame_coords[i, j, k, 1] = j
                        self.frame_coords[i, j, k, 2] = k
                        self.frame_x[i, j, k] = i
                        self.frame_y[i, j, k] = j
                        self.frame_z[i, j, k] = k
            self.cell_coords = (self.cell_coords.astype(np.float32) +
                                0.5) / self.real_size - 0.5
            self.cell_coords = np.reshape(
                self.cell_coords, [dimf, dimf, dimf, dimc * dimc * dimc, 3])
            self.cell_x = np.reshape(self.cell_x, [dimc * dimc * dimc])
            self.cell_y = np.reshape(self.cell_y, [dimc * dimc * dimc])
            self.cell_z = np.reshape(self.cell_z, [dimc * dimc * dimc])
            self.frame_x = np.reshape(self.frame_x, [dimf * dimf * dimf])
            self.frame_y = np.reshape(self.frame_y, [dimf * dimf * dimf])
            self.frame_z = np.reshape(self.frame_z, [dimf * dimf * dimf])
            self.frame_coords = (self.frame_coords.astype(np.float32) +
                                 0.5) / dimf - 0.5
            self.frame_coords = np.reshape(self.frame_coords,
                                           [dimf * dimf * dimf, 3])
