cd /newdata/ianlin/CODE/seg_trick/2d_unet/lossohem

python train.py --task lossohem --fold 1 --train-loss ohem --name res101
python test.py --task lossohem --fold 1 --test-test-epoch 0 --name res101
