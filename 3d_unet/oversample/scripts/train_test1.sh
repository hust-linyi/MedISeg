cd /newdata/ianlin/CODE/seg_trick/3d_unet/oversample

python train.py --task oversample --fold 1 --patch-size 96
python test.py --task oversample --fold 1 --test-test-epoch 0 --patch-size 96