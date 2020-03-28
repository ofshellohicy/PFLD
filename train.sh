#!/usr/bin/env bash
save_model=./models/model0
#pre_model=./models2/model3
pre_model=''
logs=./models/log0.txt
#lr=0.000000001
lr=0.0001

CUDA_VISIBLE_DEVICES='' \
nohup python -u train_model.py --model_dir=${save_model} \
                               --pretrained_model=${pre_model} \
                               --learning_rate=${lr} \
                               --level=L1 \
                               --debug=False \
                               --image_size=112 \
                               --batch_size=128 \
                               > ${logs} 2>&1 &
tail -f ${logs}
