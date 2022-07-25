cd /newdata/ianlin/CODE/seg_trick/3d_unet/lossdice

python train.py --task lossdice --fold 0 --train-loss dice
python test.py --task lossdice --fold 0 --test-test-epoch 0