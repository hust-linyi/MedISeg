cd /newdata/ianlin/CODE/seg_trick/3DUNet/da3
python train.py --task da3 --fold 4

cd /newdata/ianlin/CODE/seg_trick/3DUNet/NetworkTrainer
python test.py --task da3 --fold 4