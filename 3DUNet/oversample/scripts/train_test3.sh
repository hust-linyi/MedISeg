cd /newdata/ianlin/CODE/seg_trick/3DUNet/NetworkTrainer

python train.py --task oversample --fold 3
python test.py --task oversample --fold 3 --test-test-epoch 0