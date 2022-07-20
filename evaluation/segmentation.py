# %%
import os
import wandb
import argparse
import dotenv
import uuid
import numpy as np
import trimesh
import sys
import yaml
from tqdm import tqdm
dotenv.load_dotenv()
sys.path.insert(0, '.')
from utils import config_util
from evaluation import eval_utils
from datetime import datetime

import pandas as pd
import traceback

# %%

K100 = 100000
sampling_num = 1000


def get_job_id():
    return uuid.uuid4().hex


def eval_segmentation(cfg, args, job_id=None, parent_id=None):
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
    primitive_num = cfg['model']['kwargs']['primitive_num']
    mesh_dir = os.path.join(cfg['mesh_generation']['outdir'],
                            'mesh').replace('../../', '')
    trained_class = cfg['data']['common']['kwargs']['classes']
    train_samples = [
        l.strip().split('/') for l in open(train_data_list_path).readlines()
    ]
    train_samples = [s for s in train_samples if s[0] in trained_class]
    df = []
    label_table = np.zeros([primitive_num,
                            10])  # 10 is max gt primitive num, larget enough
    if args.debug:
        train_samples = tqdm(train_samples)
    for idx, (class_id, modelname, in_class_id) in enumerate(train_samples):
        if args.only_canonical_pose_as_gt_label and in_class_id != '0000':
            continue
        if idx > 5 and args.debug:
            break

        eval_result = {
            'class_id': class_id,
            'modelname': modelname,
            'in_class_id': in_class_id
        }
        part_meshes = {}
        for pidx in range(primitive_num):
            source_part_mesh_path = os.path.join(
                mesh_dir, class_id, modelname + '_' + in_class_id,
                source_part_mesh_filename_template).format(pidx)
            if os.path.exists(source_part_mesh_path):
                part_meshes[pidx] = trimesh.load(source_part_mesh_path)

        if len(part_meshes) == 0:
            continue

        # areas = np.array([m.area for m in part_meshes.values()])
        # areas_normalized = areas / areas.sum()
        # points_per_part = (K100 * areas_normalized).astype(np.uint32)
        points_per_part = (
            np.array([1. / len(part_meshes)] * len(part_meshes)) *
            K100).astype(np.uint32)
        points_per_part[0] += (K100 - points_per_part.sum())

        source_label = []
        source_points = []
        for idx, (pidx, part_mesh) in enumerate(part_meshes.items()):
            #sampling_num = points_per_part[idx]

            source_label.append(np.ones([sampling_num]) * pidx)

            sampled_points = part_mesh.sample(sampling_num)
            source_points.append(sampled_points)

        source_label = np.concatenate(source_label)
        source_points = np.concatenate(source_points)

        target_points_values_path = os.path.join(train_data_dir, class_id,
                                                 modelname, in_class_id,
                                                 train_points_values_filename)
        # target_primitive_indices = (np.clip(
        #     np.unique(np.load(target_points_values_path)['values']), 1, 100) -
        #                             1).tolist()
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

        # areas = np.array([m.area for m in target_part_meshes.values()])
        # areas_normalized = areas / areas.sum()
        # points_per_part = (K100 * areas_normalized).astype(np.uint32)
        points_per_part = (np.array(
            [1. / len(target_part_meshes)] * len(target_part_meshes)) *
                           K100).astype(np.uint32)
        points_per_part[0] += (K100 - points_per_part.sum())

        target_label = []
        target_points = []
        for idx, (pidx, part_mesh) in enumerate(target_part_meshes.items()):
            #sampling_num = points_per_part[idx]

            target_label.append(np.ones([sampling_num]) * pidx)

            sampled_points = part_mesh.sample(sampling_num)
            target_points.append(sampled_points)

        target_label = np.concatenate(target_label)
        target_points = np.concatenate(target_points)

        # target to pred idx
        _, selected_source_label = eval_utils.one_sided_chamfer_distance_with_index(
            target_points, source_points)
        for lidx in range(len(selected_source_label.flatten())):
            sidx = source_label[selected_source_label.flatten().astype(
                np.int64)].astype(np.int64)[lidx]
            tidx = target_label.flatten().astype(np.int64)[lidx]
            label_table[sidx, tidx] += 1. / sampling_num

    assignment_table = label_table.argmax(1)
    invalid_pm_indices = label_table.sum(1) == 0
    fixed_assignment_table = np.where(invalid_pm_indices,
                                      -np.ones_like(assignment_table),
                                      assignment_table)

    date_str = datetime.now().strftime(('%Y%m%d_%H%M%S'))
    if args.only_canonical_pose_as_gt_label:
        eval_dir = os.path.join(cfg['mesh_generation']['outdir'], 'eval',
                                ((args.prefix + '_') if args.prefix is not None else '') + 'can_pose_seg_eval_' + date_str)
    else:
        eval_dir = os.path.join(cfg['mesh_generation']['outdir'], 'eval',
                                ((args.prefix + '_') if args.prefix is not None else '') + 'seg_eval_' + date_str)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    ret = {
        'assignment_table': fixed_assignment_table,
        'label_table': label_table
    }
    np.savez(os.path.join(eval_dir, 'seg_assignment.npz'), **ret)
    np.savetxt(os.path.join(eval_dir, 'seg_assignment_table.csv'),
               fixed_assignment_table,
               delimiter=',')

    # Eval
    test_data_list_path = cfg['test']['list_path'].replace('../../', '')
    test_samples = [
        l.strip().split('/') for l in open(test_data_list_path).readlines()
    ]
    test_samples = [s for s in test_samples if s[0] in trained_class]
    df = []
    if args.debug:
        test_samples = tqdm(test_samples)
    for idx, (class_id, modelname, in_class_id) in enumerate(test_samples):
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

        # areas = np.array([m.area for m in part_meshes.values()])
        # areas_normalized = areas / areas.sum()
        # points_per_part = (K100 * areas_normalized).astype(np.uint32)
        points_per_part = (
            np.array([1. / len(part_meshes)] * len(part_meshes)) *
            K100).astype(np.uint32)
        points_per_part[0] += (K100 - points_per_part.sum())

        source_label = []
        source_points = []
        for idx, (pidx, part_mesh) in enumerate(part_meshes.items()):
            #sampling_num = points_per_part[idx]

            source_label.append(np.ones([sampling_num]) * pidx)

            sampled_points = part_mesh.sample(sampling_num)
            source_points.append(sampled_points)

        source_label = np.concatenate(source_label)
        source_points = np.concatenate(source_points)

        target_points_values_path = os.path.join(train_data_dir, class_id,
                                                 modelname, in_class_id,
                                                 train_points_values_filename)
        # target_primitive_indices = (np.clip(
        #     np.unique(np.load(target_points_values_path)['values']), 1, 100) -
        #                             1).tolist()

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

        # areas = np.array([m.area for m in target_part_meshes.values()])
        # areas_normalized = areas / areas.sum()
        # points_per_part = (K100 * areas_normalized).astype(np.uint32)
        points_per_part = (np.array(
            [1. / len(target_part_meshes)] * len(target_part_meshes)) *
                           K100).astype(np.uint32)
        points_per_part[0] += (K100 - points_per_part.sum())

        target_label = []
        target_points = []
        for idx, (pidx, part_mesh) in enumerate(target_part_meshes.items()):
            #sampling_num = points_per_part[idx]

            target_label.append(np.ones([sampling_num]) * pidx)

            sampled_points = part_mesh.sample(sampling_num)
            target_points.append(sampled_points)

        target_label = np.concatenate(target_label)
        target_points = np.concatenate(target_points)

        # target to pred idx
        _, selected_source_label = eval_utils.one_sided_chamfer_distance_with_index(
            target_points, source_points)
        pred_label = fixed_assignment_table[source_label.flatten().astype(
            np.int64)[selected_source_label.flatten().astype(
                np.int64)]].astype(np.int64)

        # tps = defaultdict(lambda: 0)
        # fps = defaultdict(lambda: 0)
        # fns = defaultdict(lambda: 0)
        # for tpidx in target_primitive_indices:
        #     tp = np.logical_and(pred_label == tpidx,
        #                         target_label == tpidx).sum()
        #     tps[tpidx] += tp

        #     fp = np.logical_and(pred_label == tpidx,
        #                         target_label != tpidx).sum()
        #     fps[tpidx] += fp

        #     fn = np.logical_and(pred_label != tpidx, pred_label == tpidx).sum()
        #     fns[tpidx] += fn

        # tmp = {}
        # all_iou = 0
        # for tpidx in target_primitive_indices:
        #     iou = tps[tpidx] / (tps[tpidx] + fps[tpidx] + fns[tpidx])
        #     tmp['part_{}_label_iou'.format(tpidx)] = iou
        #     all_iou += iou
        all_iou = 0
        tmp = {}
        for tpidx in target_primitive_indices:
            intersect = np.logical_and(pred_label == tpidx,
                                       target_label == tpidx).sum()
            union = np.logical_or(pred_label == tpidx,
                                  target_label == tpidx).sum()
            iou = float(intersect) / float(union)
            tmp['part_{}_label_iou'.format(tpidx)] = iou
            all_iou += iou

        tmp['label_iou'] = all_iou / len(target_primitive_indices)
        eval_result.update(tmp)
        df.append(eval_result)

        if args.debug:
            print('len df', len(df))
            print(df[-1])

    df = pd.DataFrame.from_dict(df)
    df.to_pickle(os.path.join(eval_dir, 'seg_eval_df.pkl'))
    summary = df.groupby('class_id').mean()
    summary.to_csv(os.path.join(eval_dir, 'seg_eval_summary.csv'))
    yaml.dump(cfg, open(os.path.join(eval_dir, 'seg_eval_config.yaml'), 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='foooo')
    parser.add_argument('--job_id', type=str, default=None, help='foooo')
    parser.add_argument('--parent_id', type=str, default=None, help='foooo')
    parser.add_argument('--debug', action='store_true', help='foooo')
    parser.add_argument('--without_wandb', action='store_true', help='foooo')
    parser.add_argument('--only_canonical_pose_as_gt_label',
                        action='store_true',
                        help='foooo')
    parser.add_argument('--prefix', type=str, default=None)

    args, unknown_args = parser.parse_known_args(sys.argv[1:])
    cfg = config_util.load_config(args.config_path)
    unknown_args_wo_eq = []
    for ua in unknown_args:
        unknown_args_wo_eq.extend(ua.split('='))

    cfg = config_util.update_dict_with_options(cfg, unknown_args_wo_eq)

    eval_segmentation(cfg, args, args.job_id, args.parent_id)
