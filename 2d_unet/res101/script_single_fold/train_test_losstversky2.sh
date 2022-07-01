cd /newdata/ianlin/CODE/seg_trick/2d_unet/losstversky

python train.py --task losstversky --fold 2 --train-loss tversky --name res101
python test.py --task losstversky --fold 2 --test-test-epoch 0 --name res101
