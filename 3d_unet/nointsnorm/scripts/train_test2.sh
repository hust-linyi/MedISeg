cd /newdata/ianlin/CODE/seg_trick/3d_unet/nointsnorm

python train.py --task nointsnorm --fold 2 --dataset nointsnorm
python test.py --task nointsnorm --fold 2 --test-test-epoch 0 --dataset yeung