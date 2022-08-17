cd /newdata/ianlin/CODE/seg_trick/3DUNet/NetworkTrainer

python train.py --task oversample --fold 2
python test.py --task oversample --fold 2 --test-test-epoch 0