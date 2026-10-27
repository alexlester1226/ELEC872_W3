#!/usr/bin/env bash
# Partner 1: Parts 0, A, B, C  (7 runs, ~15 min each on a Colab T4)
set -e
T="--task gestures --seed 0"
python train.py $T --name ref_s0
python train.py $T --name A_cnn_s0     --model cnn
python train.py $T --name A_gru_s0     --rnn gru
python train.py $T --name B_conv1_s0   --n-conv 1
python train.py $T --name B_conv2_s0   --n-conv 2
python train.py $T --name C_w20_s0     --window 20
python train.py $T --name C_w60_s0     --window 60
python summarize.py
