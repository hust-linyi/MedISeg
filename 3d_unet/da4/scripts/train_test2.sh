cd /newdata/ianlin/CODE/seg_trick/3d_unet/da4

python train.py --task da4 --fold 2 --train-gpus 0
python test.py --task da4 --fold 2 --test-gpus 0 --test-test-epoch 0