cd /newdata/ianlin/CODE/seg_trick/2d_unet/pt

python train.py --task pt --fold 3 --pretrained True --name res101
python test.py --task pt --fold 3 --test-test-epoch 0 --name res101
