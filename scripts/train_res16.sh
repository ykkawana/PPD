#!/bin/bash

# Although we did not fix and try different random seeds for dataloader during the trainings of the models reported in the main paper,
# for reproducbility purpose, we provide the random seed for dataloader that produces qualitatively similar results with the reported model.
# Note that changing number of workers of the dataloader, changing batch size may result in different results.
set -eux

BASE_CONFIG_PATH="configs/base_res16.yaml"

CLASS="eyeglasses"
SEED=3
DATA_SEED=4
MAX_ITERS=99000

python3 trains/imex_train.py \
    $BASE_CONFIG_PATH \
    --advanced_reproducibility \
    --data_seed $DATA_SEED
    --seed $SEED \
    --data.common.kwargs.classes "[$CLASS]" \
    --training.terminate_iters $MAX_ITERS

# Other categories coming soon