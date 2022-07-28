cd /newdata/ianlin/CODE/seg_trick/3DUNet/baseline

python train.py --task baseline --fold 4 
python test.py --task baseline --fold 4 --test-test-epoch 0 