cd /newdata/ianlin/CODE/seg_trick/3d_unet/baseline

python train.py --task baseline --fold 4 --train-gpus 0
python test.py --task baseline --fold 4 --test-gpus 0 --test-test-epoch 0