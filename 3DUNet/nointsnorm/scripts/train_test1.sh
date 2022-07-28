cd /newdata/ianlin/CODE/seg_trick/3DUNet/NetworkTrainer

python train.py --task nointsnorm --fold 1 --dataset nointsnorm
python test.py --task nointsnorm --fold 1 --test-test-epoch 0 --dataset yeung