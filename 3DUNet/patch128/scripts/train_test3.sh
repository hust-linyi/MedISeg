cd /newdata/ianlin/CODE/seg_trick/3DUNet/patch128

python train.py --task patch128 --fold 3 --train-gpus 0 --patch-size 128
python test.py --task patch128 --fold 3 --test-gpus 0 --test-test-epoch 0 --patch-size 128