#!/bin/bash
#!/usr/bin/env bash
CURDIR=$(cd $(dirname $0); pwd)
cd $CURDIR
export PYTHONPATH="$PYTHONPATH:$CURDIR"
echo 'The work dir is: ' $CURDIR
# NODE_RANK=$1

DATA_PATH=ImageNet:split=TRAIN:root=/fp/projects01/ec35/data/IN2012/:extra=/fp/projects01/ec35/homes/ec-thallesss/representation_learning/src/methods/dinov2/dinov2/data/extra
# RESUME_FROM_DIR=/fp/projects01/ec35/homes/ec-thallesss/representation_learning/src/methods/dinov2/experiments/run0

torchrun --nproc-per-node=4 dinov2/run/train/train.py \
    --config-file dinov2/configs/train/vitb16_dinov2_short.yaml \
    --output-dir ./experiments/vitb16_dinov2_short \
    --no-resume false \
    train.dataset_path=${DATA_PATH} 


# torchrun --nproc-per-node=4 dinov2/run/train/train.py \
#     --config-file dinov2/configs/train/vitb16_dinov2_short.yaml \
#     --output-dir ./experiments/run2 \
#     --no-resume true \
#     train.dataset_path=${DATA_PATH} 

# torchrun --nproc_per_node=4 --nnodes=2 --node_rank=${NODE_RANK} --rdzv_id=64515 --rdzv_backend=c10d --rdzv_endpoint=10.110.0.72:29606 dinov2/run/train/train.py \
#     --config-file dinov2/configs/train/vitb16_short.yaml \
#     --output-dir ./experiments/run1 \
#     train.batch_size_per_gpu=64 \
#     train.dataset_path=${DATA_PATH} \
#     MODEL.WEIGHTS=${RESUME_FROM_DIR}
