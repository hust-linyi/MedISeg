cd /newdata/ianlin/CODE/seg_trick/2d_unet/deeps

python train.py --task deeps --fold 0 --train-deeps True --name res101
python test.py --task deeps --fold 0 --test-test-epoch 0 --name res101 --train-deeps True
