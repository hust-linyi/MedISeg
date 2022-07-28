cd /newdata/ianlin/CODE/seg_trick/3DUNet/NetworkTrainer

python train.py --task losswce --fold 0 --train-loss wce
python test.py --task losswce --fold 0 --test-test-epoch 0