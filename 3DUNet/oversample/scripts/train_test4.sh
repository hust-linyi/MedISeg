cd /newdata/ianlin/CODE/seg_trick/3DUNet/NetworkTrainer

python train.py --task oversample --fold 4
python test.py --task oversample --fold 4 --test-test-epoch 0