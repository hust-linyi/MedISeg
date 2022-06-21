cd /newdata/ianlin/CODE/seg_trick/3d_unet/seed2019

python train.py --task seed2019 --fold 0 --train-seed 2019
python test.py --task seed2019 --fold 0 --test-test-epoch 0