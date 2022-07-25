cd /newdata/ianlin/CODE/seg_trick/3d_unet/deeps

python train.py --task deeps --fold 2 --train-deeps True
python test.py --task deeps --fold 2 --test-test-epoch 0 --dataset yeung