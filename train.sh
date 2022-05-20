#!/bin/bash

DATASET="cafecup"
DATASET_MODEL="cafecup"
MODEL_CKPT="./exp_result/checkpoint1/checkpoint1test_lr_001_SGD_class4_freeze34-2_new_ver2.pt"
CKPT_PATH="/home/jjunhee98/바탕화면/ksh/exp_result/checkpoint1/"
CKPT_NAME="checkpoint1test_lr_001_SGD_class4_freeze34-2_new_ver2.pt"
CUDA_VISIBLE_DEVICES=0 /home/jjunhee98/anaconda3/envs/py38/bin/python infer.py $DATASET $DATASET_MODEL --model_ckpt $MODEL_CKPT $CKPT_PATH $CKPT_NAME