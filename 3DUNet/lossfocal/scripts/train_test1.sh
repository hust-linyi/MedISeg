cd /newdata/ianlin/CODE/seg_trick/3DUNet/NetworkTrainer

python train.py --task lossfocal --fold 1 --train-loss focal
python test.py --task lossfocal --fold 1 --test-test-epoch 0