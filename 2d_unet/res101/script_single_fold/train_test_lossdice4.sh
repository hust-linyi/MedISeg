cd /newdata/ianlin/CODE/seg_trick/2d_unet/lossdice

python train.py --task lossdice --fold 4 --train-loss dice --name res101
python test.py --task lossdice --fold 4 --test-test-epoch 0 --name res101