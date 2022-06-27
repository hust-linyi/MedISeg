cd /newdata/ianlin/CODE/seg_trick/2d_unet/pt

# python train.py --task pt --fold 0 --pretrained True --name res50
# python train.py --task pt --fold 1 --pretrained True --name res50
# python train.py --task pt --fold 2 --pretrained True --name res50
# python train.py --task pt --fold 3 --pretrained True --name res50
# python train.py --task pt --fold 4 --pretrained True --name res50

python test.py --task pt --fold 0 --test-test-epoch 0
python test.py --task pt --fold 1 --test-test-epoch 0
python test.py --task pt --fold 2 --test-test-epoch 0
python test.py --task pt --fold 3 --test-test-epoch 0
python test.py --task pt --fold 4 --test-test-epoch 0
