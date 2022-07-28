cd /newdata/ianlin/CODE/seg_trick/2d_unet/da1

python train.py --task da1 --fold 0 --name res101
python test.py --task da1 --fold 0 --name res101
