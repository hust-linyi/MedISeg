cd /newdata/ianlin/CODE/seg_trick/3DUNet/NetworkTrainer

python train.py --task patch64 --fold 1 --patch-size 64
python test.py --task patch64 --fold 1 --patch-size 64