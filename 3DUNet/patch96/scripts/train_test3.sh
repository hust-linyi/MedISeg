cd /newdata/ianlin/CODE/seg_trick/3d_unet/patch96

python train.py --task patch96 --fold 3 --patch-size 96
python test.py --task patch96 --fold 3 --test-test-epoch 0 --patch-size 96