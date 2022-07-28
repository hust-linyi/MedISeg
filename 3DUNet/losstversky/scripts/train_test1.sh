cd /newdata/ianlin/CODE/seg_trick/3DUNet/NetworkTrainer

python train.py --task losstversky --fold 1 --train-loss tversky
python test.py --task losstversky --fold 1 --test-test-epoch 0