cd /newdata/ianlin/CODE/seg_trick/3d_unet/lossfocal

python train.py --task lossfocal --fold 3 --train-loss focal
python test.py --task lossfocal --fold 3 --test-test-epoch 0