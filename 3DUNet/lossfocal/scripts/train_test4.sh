cd /newdata/ianlin/CODE/seg_trick/3DUNet/lossfocal

python train.py --task lossfocal --fold 4 --train-loss focal
python test.py --task lossfocal --fold 4 --test-test-epoch 0