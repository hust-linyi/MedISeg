cd /newdata/ianlin/CODE/seg_trick/3d_unet/da1

python train.py --task da1 --fold 4 --train-gpus 0
python test.py --task da1 --fold 4 --test-gpus 0 --test-test-epoch 0