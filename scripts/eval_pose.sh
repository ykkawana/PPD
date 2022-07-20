#!/bin/bash

set -eux
CONFIG_PATH="artifacts/experiments/mesh_out/pmd_drawer_main_paper/eval/can_pose_seg_eval/seg_eval_config.yaml"

python3 evaluation/pose_evaluation.py \
    $CONFIG_PATH \
    --test.list_path artifacts/dataset/shape2motion_v3/list/test.lst