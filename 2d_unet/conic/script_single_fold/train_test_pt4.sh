cd /newdata/ianlin/CODE/seg_trick/2d_unet/pt

python train.py --task pt --fold 4 --pretrained True --name res50 --dataset conic
python test.py --task pt --fold 4 --test-test-epoch 0 --name res50 --dataset conic
