#!/bin/bash
cd /home/ylindq/Code/seg_trick/2d_unet/baseline_refactor
python train.py --fold 0 --train-gpus 5
python train.py --fold 1 --train-gpus 5
python train.py --fold 2 --train-gpus 5
python train.py --fold 3 --train-gpus 5
python train.py --fold 4 --train-gpus 5

python test.py --fold 0 --test-test-epoch 0 --train-gpus 5
python test.py --fold 1 --test-test-epoch 0 --train-gpus 5
python test.py --fold 2 --test-test-epoch 0 --train-gpus 5
python test.py --fold 3 --test-test-epoch 0 --train-gpus 5
python test.py --fold 4 --test-test-epoch 0 --train-gpus 5

python get_result.py