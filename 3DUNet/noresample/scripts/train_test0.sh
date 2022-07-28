cd /newdata/ianlin/CODE/seg_trick/3DUNet/NetworkTrainer

python train.py --task noresample --fold 0 --dataset noresample
python test.py --task noresample --fold 0 --test-test-epoch 0 --dataset noresample