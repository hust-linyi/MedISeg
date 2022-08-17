cd /newdata/ianlin/CODE/seg_trick/3DUNet/NetworkTrainer

python train.py --task patch192 --fold 0 --patch-size 192
python test.py --task patch192 --fold 0 --patch-size 192