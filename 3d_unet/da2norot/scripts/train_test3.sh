cd /newdata/ianlin/CODE/seg_trick/3d_unet/da2norot

python train.py --task da2norot --fold 3 --train-gpus 0
python test.py --task da2norot --fold 3 --test-gpus 0 --test-test-epoch 0