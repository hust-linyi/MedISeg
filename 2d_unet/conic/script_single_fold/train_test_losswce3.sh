cd /newdata/ianlin/CODE/seg_trick/2d_unet/losswce

python train.py --task losswce --fold 3 --train-loss wce --name res101
python test.py --task losswce --fold 3 --test-test-epoch 0 --name res101
