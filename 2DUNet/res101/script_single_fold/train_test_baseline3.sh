cd /newdata/ianlin/CODE/seg_trick/2d_unet/baseline

python train.py --task baseline --fold 3 --name res101

python test.py --task baseline --fold 3 --test-test-epoch 0 --name res101