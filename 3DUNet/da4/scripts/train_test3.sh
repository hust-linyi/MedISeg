cd /newdata/ianlin/CODE/seg_trick/3d_unet/da4

python train.py --task da4 --fold 3
python test.py --task da4 --fold 3 --test-test-epoch 0 --dataset yeung