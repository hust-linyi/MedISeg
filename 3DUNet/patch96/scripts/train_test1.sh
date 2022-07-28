cd /newdata/ianlin/CODE/seg_trick/3DUNet/patch96

python train.py --task patch96 --fold 1 --patch-size 96
python test.py --task patch96 --fold 1 --test-test-epoch 0 --patch-size 96