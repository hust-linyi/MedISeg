cd /newdata/ianlin/CODE/seg_trick/3DUNet/NetworkTrainer

python train.py --task oversample --fold 1
python test.py --task oversample --fold 1 --test-test-epoch 0