cd /newdata/ianlin/CODE/seg_trick/3d_unet/losstversky

python train.py --task losstversky --fold 0 --train-loss tversky
python test.py --task losstversky --fold 0 --test-test-epoch 0