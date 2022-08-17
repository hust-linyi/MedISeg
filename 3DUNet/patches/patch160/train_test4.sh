cd /newdata/ianlin/CODE/seg_trick/3DUNet/NetworkTrainer

python train.py --task patch160 --fold 4 --patch-size 160
python test.py --task patch160 --fold 4 --patch-size 160