cd /newdata/ianlin/CODE/seg_trick/3DUNet/oversample

python train.py --task oversample --fold 2 --patch-size 96
python test.py --task oversample --fold 2 --test-test-epoch 0 --patch-size 96