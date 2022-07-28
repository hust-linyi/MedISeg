cd /newdata/ianlin/CODE/seg_trick/2d_unet/da4

python train.py --task da4 --fold 2 --name res101
python test.py --task da4 --fold 2 --name res101
