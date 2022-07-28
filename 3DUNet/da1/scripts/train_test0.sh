cd /newdata/ianlin/CODE/seg_trick/3DUNet/da1

python train.py --task da1 --fold 0
python test.py --task da1 --fold 0 --test-test-epoch 0