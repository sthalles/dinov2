#!/bin/bash
#!/usr/bin/env bash
CURDIR=$(cd $(dirname $0); pwd)
cd $CURDIR
export PYTHONPATH="$PYTHONPATH:$CURDIR"
echo 'The work dir is: ' $CURDIR

python dinov2/run/train/train.py \
    --nodes 4 \
    --config-file dinov2/configs/train/vitb16_short.yaml \
    --output-dir ./ \
    train.dataset_path=ImageNet:split=TRAIN:root=/fp/projects01/ec35/data/IN2012/:extra=/fp/projects01/ec35/homes/ec-thallesss/representation_learning/src/methods/dinov2/dinov2/data/extra