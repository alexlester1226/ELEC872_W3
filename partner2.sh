#!/usr/bin/env bash
# Partner 2: Parts D, E, F  (7 runs, ~15 min each on a Colab T4)
set -e
T="--task gestures --seed 0"
# D: the reference config already has weight decay 0 and no augmentation, so start
# from a "fully regularized" config and remove one regularizer at a time.
FULL="--dropout 0.5 --weight-decay 1e-4 --augment"
python train.py $T --name D_full_s0    $FULL
python train.py $T --name D_nodrop_s0  --dropout 0.0 --weight-decay 1e-4 --augment
python train.py $T --name D_nowd_s0    --dropout 0.5 --weight-decay 0   --augment
python train.py $T --name D_noaug_s0   --dropout 0.5 --weight-decay 1e-4
# E: high learning rate without and with a 5-epoch linear warm-up
python train.py $T --name E_lr3e-3_s0         --lr 3e-3
python train.py $T --name E_lr3e-3_warmup_s0  --lr 3e-3 --warmup 5
# F: focal loss (compare with ref_s0 from partner 1, especially macro F1)
python train.py $T --name F_focal_s0   --loss focal --focal-gamma 2
python summarize.py
