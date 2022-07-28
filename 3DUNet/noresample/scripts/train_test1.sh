cd /newdata/ianlin/CODE/seg_trick/3DUNet/NetworkTrainer

python train.py --task noresample --fold 1 --dataset noresample
python test.py --task noresample --fold 1 --test-test-epoch 0 --dataset noresample