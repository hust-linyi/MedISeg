cd /newdata/ianlin/CODE/seg_trick/3DUNet/losswce

python train.py --task losswce --fold 2 --train-loss wce
python test.py --task losswce --fold 2 --test-test-epoch 0