cd /newdata/ianlin/CODE/seg_trick/3d_unet/sortimglist

python train.py --task sortimglist --fold 0 --train-gpus 0
python test.py --task sortimglist --fold 0 --test-gpus 0 --test-test-epoch 0