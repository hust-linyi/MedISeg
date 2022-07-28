cd /newdata/ianlin/CODE/seg_trick/3DUNet/NetworkTrainer

python train.py --task baseline --fold 0 
python test.py --task baseline --fold 0 --test-test-epoch 0 