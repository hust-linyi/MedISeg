cd /newdata/ianlin/CODE/seg_trick/3d_unet/da2rot

python train.py --task da2rot --fold 2
python test.py --task da2rot --fold 2