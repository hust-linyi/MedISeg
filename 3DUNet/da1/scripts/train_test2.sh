cd /newdata/ianlin/CODE/seg_trick/3DUNet/NetworkTrainer

python train.py --task da1 --fold 2
python test.py --task da1 --fold 2 --test-test-epoch 0