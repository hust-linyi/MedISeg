cd /newdata/ianlin/CODE/seg_trick/3DUNet/oversample

python train.py --task oversample --fold 3 --patch-size 96
python test.py --task oversample --fold 3 --test-test-epoch 0 --patch-size 96