cd /newdata/ianlin/CODE/seg_trick/3DUNet/NetworkTrainer

python train.py --task patch128 --fold 3 --patch-size 128
python test.py --task patch128 --fold 3 --patch-size 128