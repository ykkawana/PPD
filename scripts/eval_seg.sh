#!/bin/bash

set -eux
CONFIG_PATH="artifacts/experiments/mesh_out/pmd_drawer_main_paper/mesh/config.yaml"

python3 evaluation/segmentation.py \
    $CONFIG_PATH \
    --test.list_path artifacts/dataset/shape2motion_v3/list/test.lst \
    --only_canonical_pose_as_gt_label