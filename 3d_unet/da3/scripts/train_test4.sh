cd /newdata/ianlin/CODE/seg_trick/3d_unet/da3

python train.py --task da3 --fold 4
python test.py --task da3 --fold 4 --test-test-epoch 0