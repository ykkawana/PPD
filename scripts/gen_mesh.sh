#!/bin/bash

set -eux
MESH_OUT_DIR="artifacts/experiments/mesh_out"
LIST_PATH="artifacts/dataset/shape2motion_v3/list/all.lst"
CHECKPOINT_PATH="weights/main-paper/oven-res32.pth"
# RUN_ID="" # For using W&B

python3 trains/imex_train.py a \
    --checkpoint $CHECKPOINT_PATH \
    --mode gen_mesh \
    --mesh_out_dir $MESH_OUT_DIR \
    --data.test.kwargs.list_path $LIST_PATH \
    --mesh_gen_threshold 0.5

# For using W&B
# python3 trains/imex_train.py a \
#     --run_id $RUN_ID \
#     --mode gen_mesh \
#     --mesh_out_dir $MESH_OUT_DIR \
#     --data.test.kwargs.list_path $LIST_PATH \
#     --mesh_gen_threshold 0.5

