cd /newdata/ianlin/CODE/seg_trick/3DUNet/nointsnorm

python train.py --task nointsnorm --fold 3 --dataset nointsnorm
python test.py --task nointsnorm --fold 3 --test-test-epoch 0 --dataset yeung