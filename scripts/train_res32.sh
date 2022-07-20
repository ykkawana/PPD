#!/bin/bash

set -eux

RUN_ID= # Set W&B run id here
CLASS="eyeglasses"
PTHV="v99"
SEED=3
DATA_SEED=4
MAX_ITERS=148000

python3 trains/imex_train.py a \
    --advanced_reproducibility \
    --seed $SEED \
    --data_seed $DATA_SEED \
    --run_id "$RUN_ID:$PTHV" \
    --resume_from_run_id \
    --data.common.kwargs.points_values_filename points_values_whole_shape_sample_wise_normalize_32.npz \
    --trainer.kwargs.use_minimize_raw_canonical_direction_to_offset_loss true \
    --trainer.kwargs.minimize_raw_canonical_direction_to_offset_loss_weight 10 \
    --model.kwargs.param_type only_amount_as_matrix_loc_offset_canonical_motion_euler_angle_plus_direction_offset \
    --model.pretrained.ignore_artifact_keys "[optimizer]" \
    --training.terminate_iters $MAX_ITERS

# Other categories coming soon