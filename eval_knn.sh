#!/bin/bash
#!/usr/bin/env bash
CURDIR=$(cd $(dirname $0); pwd)
cd $CURDIR
export PYTHONPATH="$PYTHONPATH:$CURDIR"
echo 'The work dir is: ' $CURDIR

torchrun --nproc-per-node=4 dinov2/run/eval/knn.py \
    --config-file /fp/projects01/ec35/homes/ec-thallesss/representation_learning/src/methods/dinov2/config.yaml \
    --pretrained-weights /fp/projects01/ec35/homes/ec-thallesss/representation_learning/src/methods/dinov2/eval/training_37499/teacher_checkpoint.pth \
    --output-dir /fp/projects01/ec35/homes/ec-thallesss/representation_learning/src/methods/dinov2/eval/training_37499/knn \
    --train-dataset ImageNet:split=TRAIN:root=/fp/projects01/ec35/data/IN2012/:extra=/fp/projects01/ec35/homes/ec-thallesss/representation_learning/src/methods/dinov2/dinov2/data/extra \
    --val-dataset ImageNet:split=VAL:root=/fp/projects01/ec35/data/IN2012/:extra=/fp/projects01/ec35/homes/ec-thallesss/representation_learning/src/methods/dinov2/dinov2/data/extra
