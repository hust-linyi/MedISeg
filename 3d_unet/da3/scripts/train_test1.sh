cd /newdata/ianlin/CODE/seg_trick/3d_unet/da3

python train.py --task da3 --fold 1 --train-gpus 0
python test.py --task da3 --fold 1 --test-gpus 0 --test-test-epoch 0