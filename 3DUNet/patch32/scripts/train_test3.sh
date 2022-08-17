cd /newdata/ianlin/CODE/seg_trick/3DUNet/NetworkTrainer

python train.py --task patch32 --fold 3 --patch-size 32
python test.py --task patch32 --fold 3 --patch-size 32