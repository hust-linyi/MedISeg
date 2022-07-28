cd /newdata/ianlin/CODE/seg_trick/3DUNet/noresample

python train.py --task noresample --fold 2 --dataset noresample
python test.py --task noresample --fold 2 --test-test-epoch 0 --dataset noresample