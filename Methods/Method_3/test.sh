#!/bin/bash

DATASET_NAME="Water-3D"


DATA_PATH="./datasets/${DATASET_NAME}"
MODEL_PATH="./models/${DATASET_NAME}"
ROLLOUT_PATH="./rollouts/${DATASET_NAME}"


# Plot the first rollout.
python -m render_rollout --rollout_path="${ROLLOUT_PATH}/rollout_test_0.pkl" --block_on_show=True

