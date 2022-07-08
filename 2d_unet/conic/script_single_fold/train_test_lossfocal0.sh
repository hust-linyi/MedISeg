cd /newdata/ianlin/CODE/seg_trick/2d_unet/lossfocal

python train.py --task lossfocal --fold 0 --train-loss focal --name res101
python test.py --task lossfocal --fold 0 --test-test-epoch 0 --name res101
