cd /newdata/ianlin/CODE/seg_trick/3d_unet/da2flip

python train.py --task da2flip --fold 3
python test.py --task da2flip --fold 3