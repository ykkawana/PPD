# %%
from collections import defaultdict
import os
import wandb
import argparse
import dotenv
import uuid
import numpy as np
import trimesh
import sys
import pickle
import yaml
from tqdm import tqdm
dotenv.load_dotenv()
sys.path.insert(0, '.')
from utils import config_util
from evaluation import eval_utils
from datetime import datetime
import torch
import torch.nn as nn

import pandas as pd
import traceback

# %%

K100 = 100000
sampling_num = 1000
device = 'cuda'

class ParamNet(nn.Module):
    def __init__(self, c_dim=8, depth=2, pmnum=8, out_dim=3, direct=False, init_scale=1e-2):
        super().__init__()
        if direct:
            self.const = nn.Parameter(torch.randn(pmnum, out_dim))
            self.net = nn.Identity()
        else:
            self.const = nn.Parameter(torch.randn(pmnum, c_dim))
            self.net = []
            for _ in range(depth):
                self.net.append(nn.Linear(c_dim, c_dim))
                self.net.append(nn.LeakyReLU(inplace=True))
            self.net.append(nn.Linear(c_dim, out_dim))
            self.net[-1].weight.data = self.net[-1].weight.data * init_scale 
            self.net[-1].bias.data = self.net[-1].bias.data * init_scale
            self.net = nn.Sequential(*self.net)
    
    @property
    def param(self):
        return self.forward()

    def forward(self):
        return self.net(self.const)

part_nums = {
    'drawer': 4,
    'oven': 2,
    'washing_machine': 2,
    'eyeglasses': 3,
    'laptop': 2
}

STEP = 500
def get_job_id():
    return uuid.uuid4().hex


def eval_pose(cfg, args, job_id=None, parent_id=None):
    if not job_id:
        job_id = get_job_id()
    kwargs = dict(job_type='test', id=job_id)
    if parent_id:
        kwargs['group'] = parent_id

    if args.debug:
        os.environ['WANDB_MODE'] = 'dryrun'
    if not args.without_wandb or True:
        wandb.init(**kwargs)
        wandb.config.update(args)
        wandb.config.update({'job_name': os.path.basename(__file__)})

    # data_list_path = cfg['test']['list_path']
    # data_dir = cfg['test']['data_dir']
    train_data_list_path = cfg['data']['train']['kwargs']['list_path'].replace(
        '../../', '')
    train_data_dir = cfg['data']['common']['kwargs']['data_dir'].replace(
        '../../', '')
    target_part_mesh_filename_template = cfg['test']['filenames'][
        'part_mesh_filename_template']
    source_part_mesh_filename_template = cfg['test']['filenames'][
        'source_part_mesh_filename_template']
    train_points_values_filename = cfg['data']['common']['kwargs'][
        'points_values_filename']

    data_dir = cfg['test']['data_dir'].replace('../../', '')
    target_surface_points_filename = cfg['test']['filenames'][
        'surface_points_filename'].replace('../../', '')
    primitive_num = cfg['model']['kwargs']['primitive_num']
    rotation_primitive_num = cfg['model']['kwargs']['rotation_primitive_num']
    translation_primitive_num = primitive_num - rotation_primitive_num - 1
    motion_param_filename = 'pred_params.pkl'
    mesh_dir = os.path.join(cfg['mesh_generation']['outdir'],
                            'mesh').replace('../../', '')
    trained_class = cfg['data']['common']['kwargs']['classes']

    # Eval
    test_data_list_path = cfg['test']['list_path'].replace('../../', '')
    test_samples = [
        l.strip().split('/') for l in open(test_data_list_path).readlines()
    ]
    test_samples = [s for s in test_samples if s[0] in trained_class]
    non_canonical_test_samples = [s for s in test_samples if s[2] != '0000']
    canonical_test_samples = [s for s in test_samples if s[2] == '0000']

    seg_eval_dir = os.path.dirname(args.seg_config_path)
    fixed_assignment_table = np.loadtxt(os.path.join(
        seg_eval_dir, 'seg_assignment_table.csv'),
                                        dtype=np.int32)
    df = []
    canonical_samples = {}
    for idx, (class_id, modelname,
              in_class_id) in enumerate(canonical_test_samples):
        if (class_id, modelname) not in canonical_samples:
            target_points_values_path = os.path.join(
                train_data_dir, class_id, modelname, in_class_id,
                train_points_values_filename)
            target_points_values = np.load(
                target_points_values_path)['values'].reshape([-1])
            target_points_values = target_points_values[
                target_points_values > 0] - 1
            target_primitive_indices = np.unique(target_points_values).tolist()

            target_part_meshes = {}
            for pidx in target_primitive_indices:
                target_part_mesh_path = os.path.join(
                    train_data_dir, class_id, modelname, in_class_id,
                    target_part_mesh_filename_template).format(pidx)
                target_part_meshes[pidx] = trimesh.load(target_part_mesh_path)

            # points_per_part = (np.array(
            #     [1. / len(target_part_meshes)] * len(target_part_meshes)) *
            #                    K100).astype(np.uint32)
            # points_per_part[0] += (K100 - points_per_part.sum())

            target_label = []
            target_points = []
            target_points_set = {}
            for idx, (pidx,
                      part_mesh) in enumerate(target_part_meshes.items()):
                #sampling_num = points_per_part[idx]

                target_label.append(np.ones([sampling_num]) * pidx)

                sampled_points = part_mesh.sample(sampling_num)
                target_points.append(sampled_points)
                target_points_set[pidx] = sampled_points

            target_label = np.concatenate(target_label)
            target_points = np.concatenate(target_points)

            motion_param_path = os.path.join(mesh_dir, class_id,
                                             modelname + '_' + in_class_id,
                                             motion_param_filename)
            motion_param = pickle.load(open(motion_param_path, 'rb'))

            canonical_samples[(class_id, modelname)] = {
                'target_points': target_points,
                'target_label': target_label,
                'target_points_set': target_points_set,
                'target_primitive_indices': target_primitive_indices,
                'target_motion_param': motion_param
            }

    if args.debug:
        non_canonical_test_samples = tqdm(non_canonical_test_samples)
    for idx, (class_id, modelname,
              in_class_id) in enumerate(non_canonical_test_samples):
        eval_result = {
            'class_id': class_id,
            'modelname': modelname,
            'in_class_id': in_class_id
        }

        if idx > 5 and args.debug:
            break

        part_meshes = {}
        for pidx in range(primitive_num):
            source_part_mesh_path = os.path.join(
                mesh_dir, class_id, modelname + '_' + in_class_id,
                source_part_mesh_filename_template).format(pidx)
            if os.path.exists(source_part_mesh_path):
                part_meshes[pidx] = trimesh.load(source_part_mesh_path)

        if len(part_meshes) == 0:
            if args.debug:
                print('no part meshes')
            continue

        # points_per_part = (
        #     np.array([1. / len(part_meshes)] * len(part_meshes)) *
        #     K100).astype(np.uint32)
        # points_per_part[0] += (K100 - points_per_part.sum())

        source_label = []
        source_points = []
        source_points_set = {}
        for idx, (pidx, part_mesh) in enumerate(part_meshes.items()):
            #sampling_num = points_per_part[idx]

            source_label.append(np.ones([sampling_num]) * pidx)

            sampled_points = part_mesh.sample(sampling_num)
            source_points.append(sampled_points)
            source_points_set[pidx] = sampled_points

        source_label = np.concatenate(source_label)
        source_points = np.concatenate(source_points)

        canonical_sample = canonical_samples[(class_id, modelname)]

        # Use points from gt surface
        source_corresponding_points_set = defaultdict(lambda: {})
        for tpidx, target_part_points in canonical_sample[
                'target_points_set'].items():
            source_corresponding_indices = np.where(
                fixed_assignment_table == tpidx)[0].tolist()
            source_corresponding_points = []
            source_corresponding_labels = []
            for pidx in source_corresponding_indices:
                if pidx not in source_points_set:
                    continue
                source_corresponding_points.append(source_points_set[pidx])
                source_corresponding_labels.append(
                    (np.ones([source_corresponding_points[-1].shape[0]],
                             dtype=np.int32) * pidx).astype(np.int32))
            try:
                if len(source_corresponding_points) == 0:
                    source_corresponding_points_set[tpidx] = None
                    continue
                else:
                    source_corresponding_points = np.concatenate(
                        source_corresponding_points)
                    source_corresponding_labels = np.concatenate(
                        source_corresponding_labels)
            except BaseException as e:
                print(list(source_points_set.keys()))
                print(source_corresponding_indices)
                raise e

            _, selected_source_corresnponding_indices = eval_utils.one_sided_chamfer_distance_with_index(
                target_part_points, source_corresponding_points)
            selected_source_corresnponding_labels = source_corresponding_labels[
                selected_source_corresnponding_indices.flatten().astype(
                    np.int64)].astype(np.int64)

            for pidx in source_corresponding_indices:
                if pidx not in source_points_set:
                    continue
                source_corresponding_points_set[tpidx][
                    pidx] = target_part_points[
                        selected_source_corresnponding_labels == pidx, :]

        # Use points from recon surface
        source_corresponding_points_set_recon = defaultdict(lambda: {})
        for tpidx, target_part_points in canonical_sample[
                'target_points_set'].items():
            source_corresponding_indices = np.where(
                fixed_assignment_table == tpidx)[0].tolist()
            source_corresponding_points = []
            for pidx in source_corresponding_indices:
                if pidx not in source_points_set:
                    continue
                source_corresponding_points_set_recon[tpidx][
                    pidx] = source_points_set[pidx]
            if len(source_corresponding_points_set_recon[tpidx]) == 0:
                source_corresponding_points_set_recon[tpidx] = None

        # gt deformation
        motion_param_path = os.path.join(mesh_dir, class_id,
                                         modelname + '_' + in_class_id,
                                         motion_param_filename)
        motion_param = pickle.load(open(motion_param_path, 'rb'))
        gt_motion_param = motion_param['gt']
        gt_per_part_deformation_matrices = {}
        gt_pm_type = gt_motion_param['param_primitive_type']
        static_pm_offset = int(
            gt_motion_param['param_primitive_type_offset'][0].item())
        for tpidx in canonical_sample['target_primitive_indices']:
            tpidx_offset = tpidx - static_pm_offset
            pm_type = gt_pm_type[tpidx]
            # print('pm_type', pm_type)
            # print('tpidx', tpidx)
            # print('gt pm type', gt_pm_type)
            # print('target primitive indices',
            #       canonical_sample['target_primitive_indices'])
            if pm_type == 0:  # static
                gt_per_part_deformation_matrices[tpidx] = np.eye(4)
            elif pm_type == 1:  # revolute
                assert tpidx_offset >= 0
                anchor_point = gt_motion_param['param_anchor_point'][
                    tpidx_offset, :]
                tr1 = trimesh.transformations.translation_matrix(-anchor_point)
                rot_amount = gt_motion_param['param_amount'][
                    tpidx_offset].item()
                rot_direction = gt_motion_param['param_direction'][
                    tpidx_offset, :]
                rot = trimesh.transformations.rotation_matrix(
                    rot_amount, rot_direction)
                tr2 = trimesh.transformations.translation_matrix(anchor_point)
                T = trimesh.transformations.concatenate_matrices(tr2, rot, tr1)
                gt_per_part_deformation_matrices[tpidx] = T
            elif pm_type == 2:  # revolute
                assert tpidx_offset >= 0
                tra_amount = gt_motion_param['param_amount'][
                    tpidx_offset].item()
                tra_direction = gt_motion_param['param_direction'][
                    tpidx_offset, :]
                T = trimesh.transformations.translation_matrix(tra_amount *
                                                               tra_direction)
                gt_per_part_deformation_matrices[tpidx] = T
            else:
                raise ValueError(
                    'PM type exceeds expected value range [0, 1, 2], sth is wrong.'
                )

        # pred deformation, joint aware
        source_per_part_deformation_matrices_joint_aware = {0: np.eye(4)}
        can_motion_param = canonical_sample['target_motion_param']
        pcnt = 1
        for pidx in range(rotation_primitive_num):
            anchor_point = can_motion_param['pred']['param'][
                'canonical_location'][pidx + 1, :]
            tr1 = trimesh.transformations.translation_matrix(-anchor_point)
            rot_amount = motion_param['pred']['param']['rotation_amount'][
                pidx, 0].item()
            can_rot_amount = can_motion_param['pred']['param'][
                'rotation_amount'][pidx, 0].item()
            rot_amount = rot_amount - can_rot_amount
            rot_direction = can_motion_param['pred']['param'][
                'rotation_direction'][pidx, :]
            rot = trimesh.transformations.rotation_matrix(
                rot_amount, rot_direction)

            tr2 = trimesh.transformations.translation_matrix(anchor_point)
            T = trimesh.transformations.concatenate_matrices(tr2, rot, tr1)
            source_per_part_deformation_matrices_joint_aware[pcnt] = T
            pcnt += 1
        for pidx in range(translation_primitive_num):
            tra_amount = motion_param['pred']['param']['translation_amount'][
                pidx, 0].item()
            can_tra_amount = can_motion_param['pred']['param'][
                'translation_amount'][pidx, 0].item()
            tra_direction = can_motion_param['pred']['param'][
                'translation_direction'][pidx, :]
            tra_amount = tra_amount - can_tra_amount
            T = trimesh.transformations.translation_matrix(tra_amount *
                                                           tra_direction)
            source_per_part_deformation_matrices_joint_aware[pcnt] = T
            pcnt += 1

        # pred deformation, joint un-aware
        source_per_part_deformation_matrices_joint_unaware_can = {}
        pcnt = 1
        for idx in range(rotation_primitive_num):
            anchor_point = can_motion_param['pred']['param'][
                'canonical_location'][idx + 1, :]
            tr1 = trimesh.transformations.translation_matrix(-anchor_point)
            rot_amount = can_motion_param['pred']['param']['rotation_amount'][
                idx, 0].item()
            rot_direction = can_motion_param['pred']['param'][
                'rotation_direction'][idx, :]
            rot = trimesh.transformations.rotation_matrix(
                rot_amount, rot_direction)
            tr2 = trimesh.transformations.translation_matrix(anchor_point)
            T = trimesh.transformations.concatenate_matrices(tr2, rot, tr1)
            source_per_part_deformation_matrices_joint_unaware_can[
                pcnt] = np.linalg.inv(T)
            pcnt += 1
        for idx in range(translation_primitive_num):
            tra_amount = can_motion_param['pred']['param'][
                'translation_amount'][idx, 0].item()
            tra_direction = can_motion_param['pred']['param'][
                'translation_direction'][idx, :]
            T = trimesh.transformations.translation_matrix(tra_amount *
                                                           tra_direction)
            #Ts.append(-T)
            #Ts.append(T)
            source_per_part_deformation_matrices_joint_unaware_can[
                pcnt] = np.linalg.inv(T)
            #Ts.append(np.eye(4))
            pcnt += 1
        source_per_part_deformation_matrices_joint_unaware = {0: np.eye(4)}
        pcnt = 1
        for idx in range(rotation_primitive_num):
            anchor_point = motion_param['pred']['param']['canonical_location'][
                idx + 1, :]
            tr1 = trimesh.transformations.translation_matrix(-anchor_point)
            rot_amount = motion_param['pred']['param']['rotation_amount'][
                idx, 0].item()
            rot_direction = motion_param['pred']['param'][
                'rotation_direction'][idx, :]
            rot = trimesh.transformations.rotation_matrix(
                rot_amount, rot_direction)

            tr2 = trimesh.transformations.translation_matrix(anchor_point)
            can_anchor_point = can_motion_param['pred']['param'][
                'canonical_location'][idx + 1, :]
            tr_anchor = trimesh.transformations.translation_matrix(anchor_point - can_anchor_point)
            T = trimesh.transformations.concatenate_matrices(
                tr2, rot, tr1,
                # tr_anchor, tr2, rot, tr1,
                source_per_part_deformation_matrices_joint_unaware_can[pcnt])
            source_per_part_deformation_matrices_joint_unaware[pcnt] = T
            pcnt += 1
        for idx in range(translation_primitive_num):
            tra_amount = motion_param['pred']['param']['translation_amount'][
                idx, 0].item()
            tra_direction = motion_param['pred']['param'][
                'translation_direction'][idx, :]
            T = trimesh.transformations.translation_matrix(tra_amount *
                                                           tra_direction)
            T = trimesh.transformations.concatenate_matrices(
                T,
                source_per_part_deformation_matrices_joint_unaware_can[pcnt])
            source_per_part_deformation_matrices_joint_unaware[pcnt] = T
            pcnt += 1

        # EPE
        # points from gt surface
        epe_results = {}
        epe_sample_points_sets = {
            'gt_surface': source_corresponding_points_set,
        }
        source_deformation_sets = {
            'joint_aware': source_per_part_deformation_matrices_joint_aware,
            'joint_unaware': source_per_part_deformation_matrices_joint_unaware
        }

        if args.debug:
            debug_deform_results = {}
        for deformation_type, deformation_matrices in source_deformation_sets.items(
        ):
            for sampling_type, points_set in epe_sample_points_sets.items():
                cnt = 0
                all_epe = 0
                for tpidx, points_subset in points_set.items():
                    target_mt = gt_per_part_deformation_matrices[tpidx]
                    if sampling_type == 'recon_surface':
                        target_mt = np.linalg.inv(target_mt)
                    source_per_part_deformed_points = []
                    target_per_part_deformed_points = []
                    if points_subset is None:
                        #cnt += 1
                        continue
                    for pidx, points in points_subset.items():
                        # joint aware
                        src_mt = deformation_matrices[pidx]
                        if sampling_type == 'recon_surface':
                            src_mt = np.linalg.inv(src_mt)
                        source_deformed_points = trimesh.transform_points(
                            points, src_mt)
                        source_per_part_deformed_points.append(
                            source_deformed_points)

                        target_deformed_points = trimesh.transform_points(
                            points, target_mt)
                        target_per_part_deformed_points.append(
                            target_deformed_points)

                    source_per_part_deformed_points = np.concatenate(
                        source_per_part_deformed_points)
                    target_per_part_deformed_points = np.concatenate(
                        target_per_part_deformed_points)


                    if args.debug:
                        debug_deform_results['_'.join([
                            'target', deformation_type, sampling_type,
                            'part_{}'.format(tpidx)
                        ])] = target_per_part_deformed_points
                        debug_deform_results['_'.join([
                            'source', deformation_type, sampling_type,
                            'part_{}'.format(tpidx)
                        ])] = source_per_part_deformed_points

                    epe = (((source_per_part_deformed_points -
                             target_per_part_deformed_points)**2).sum(-1)**(
                                 0.5)).mean()
                    all_epe += epe
                    epe_results['_'.join([
                        deformation_type, sampling_type,
                        'part_{}'.format(tpidx)
                    ])] = epe
                    cnt += 1
                all_epe_avg = all_epe / cnt
                epe_results['_'.join([deformation_type, sampling_type,
                                      'all'])] = all_epe_avg
            if args.debug:
                pickle.dump(
                    debug_deform_results,
                    open('_'.join([class_id, modelname, in_class_id]) + '.pkl',
                         'wb'))
        eval_result.update(epe_results)
        df.append(eval_result)
        if args.debug:
            print('len df', len(df))
            print(df[-1])

    date_str = datetime.now().strftime(('%Y%m%d_%H%M%S'))
    eval_dir = os.path.join(cfg['mesh_generation']['outdir'], 'eval',
                            ((args.prefix + '_') if args.prefix is not None else '') + 'pose_eval_' + date_str)
    df = pd.DataFrame.from_dict(df)
    os.makedirs(eval_dir)
    df.to_pickle(os.path.join(eval_dir, 'pose_eval_df.pkl'))
    alls = [c for c in df.columns if c.endswith('_all')]
    summary = df.groupby('class_id').mean()[alls]
    summary.to_csv(os.path.join(eval_dir, 'pose_eval_summary.csv'))
    yaml.dump(cfg, open(os.path.join(eval_dir, 'pose_eval_config.yaml'), 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('seg_config_path', type=str, help='foooo')
    parser.add_argument('--job_id', type=str, default=None, help='foooo')
    parser.add_argument('--parent_id', type=str, default=None, help='foooo')
    parser.add_argument('--debug', action='store_true', help='foooo')
    parser.add_argument('--without_wandb', action='store_true', help='foooo')
    parser.add_argument('--prefix', type=str, default=None)
    args, unknown_args = parser.parse_known_args(sys.argv[1:])
    cfg = config_util.load_config(args.seg_config_path)
    unknown_args_wo_eq = []
    for ua in unknown_args:
        unknown_args_wo_eq.extend(ua.split('='))

    cfg = config_util.update_dict_with_options(cfg, unknown_args_wo_eq)

    eval_pose(cfg, args, args.job_id, args.parent_id)
