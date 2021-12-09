#!/bin/bash


#python -m model_demo

DATASET_NAME="Water-3D"

# Train for a few steps.
DATA_PATH="./datasets/${DATASET_NAME}"
MODEL_PATH="./models/${DATASET_NAME}"
python -m train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --num_steps=1000000

# Evaluate on validation split.
#python -m train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --mode="eval" --eval_split="valid"

# Generate test rollouts.
#ROLLOUT_PATH="./rollouts/${DATASET_NAME}"
#mkdir -p ${ROLLOUT_PATH}
#python -m train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --mode="eval_rollout" --output_path=${ROLLOUT_PATH}


#Crash 1
python -m train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --num_steps=1000000

#Crash 2
python -m train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --num_steps=1000000

#Crash 3
python -m train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --num_steps=1000000

#Crash 4
python -m train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --num_steps=1000000

#Crash 5
python -m train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --num_steps=1000000


#Validate
python -m train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --mode="eval" --eval_split="valid"


#Rollout
# Generate test rollouts.
ROLLOUT_PATH="./rollouts/${DATASET_NAME}"
mkdir -p ${ROLLOUT_PATH}
python -m train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --mode="eval_rollout" --output_path=${ROLLOUT_PATH}

