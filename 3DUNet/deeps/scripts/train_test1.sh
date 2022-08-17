cd /newdata/ianlin/CODE/seg_trick/3DUNet/NetworkTrainer

python train.py --task deeps --fold 1 --train-deeps True
python test.py --task deeps --fold 1 --test-test-epoch 0  --train-deeps True