cd /newdata/ianlin/CODE/seg_trick/3d_unet/seed2019

python train.py --task seed2019 --fold 3 --train-seed 2019
python test.py --task seed2019 --fold 3 --test-test-epoch 0