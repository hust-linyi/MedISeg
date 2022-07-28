cd /newdata/ianlin/CODE/seg_trick/3DUNet/NetworkTrainer

python train.py --task da1 --fold 1
python test.py --task da1 --fold 1 --test-test-epoch 0