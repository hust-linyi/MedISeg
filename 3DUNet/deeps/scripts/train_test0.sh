cd /newdata/ianlin/CODE/seg_trick/3DUNet/deeps

python train.py --task deeps --fold 0 --train-deeps True
python test.py --task deeps --fold 0 --test-test-epoch 0 --dataset yeung