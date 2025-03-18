#!/bin/bash
#!/usr/bin/env bash
CURDIR=$(cd $(dirname $0); pwd)
cd $CURDIR
export PYTHONPATH="$PYTHONPATH:$CURDIR"
echo 'The work dir is: ' $CURDIR

MODELS_ROOT_PATH='/fp/projects01/ec35/homes/ec-thallesss/representation_learning/src/methods/dinov2/experiments/vitb16_dinov2_short'
MODEL_VERSION='training_124999'

torchrun --nproc-per-node=4 dinov2/run/eval/knn.py \
    --config-file "${MODELS_ROOT_PATH}/config.yaml" \
    --pretrained-weights "${MODELS_ROOT_PATH}/eval/${MODEL_VERSION}/teacher_checkpoint.pth" \
    --output-dir "${MODELS_ROOT_PATH}/eval/${MODEL_VERSION}/knn" \
    --train-dataset ImageNet:split=TRAIN:root=/fp/projects01/ec35/data/IN2012/:extra=/fp/projects01/ec35/homes/ec-thallesss/representation_learning/src/methods/dinov2/dinov2/data/extra \
    --val-dataset ImageNet:split=VAL:root=/fp/projects01/ec35/data/IN2012/:extra=/fp/projects01/ec35/homes/ec-thallesss/representation_learning/src/methods/dinov2/dinov2/data/extra
