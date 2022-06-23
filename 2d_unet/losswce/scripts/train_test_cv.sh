cd /newdata/ianlin/CODE/seg_trick/2d_unet/losswce

python train.py --task losswce --fold 0 --train-loss wce
python train.py --task losswce --fold 1 --train-loss wce
python train.py --task losswce --fold 2 --train-loss wce
python train.py --task losswce --fold 3 --train-loss wce
python train.py --task losswce --fold 4 --train-loss wce

python test.py --task losswce --fold 0 --test-test-epoch 0
python test.py --task losswce --fold 1 --test-test-epoch 0
python test.py --task losswce --fold 2 --test-test-epoch 0
python test.py --task losswce --fold 3 --test-test-epoch 0
python test.py --task losswce --fold 4 --test-test-epoch 0
