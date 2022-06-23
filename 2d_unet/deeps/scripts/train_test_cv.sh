cd /newdata/ianlin/CODE/seg_trick/2d_unet/deeps

python train.py --task deeps --fold 0 --train-deeps True
python train.py --task deeps --fold 1
python train.py --task deeps --fold 2
python train.py --task deeps --fold 3
python train.py --task deeps --fold 4

python test.py --task deeps --fold 0 --test-test-epoch 0
python test.py --task deeps --fold 1 --test-test-epoch 0
python test.py --task deeps --fold 2 --test-test-epoch 0
python test.py --task deeps --fold 3 --test-test-epoch 0
python test.py --task deeps --fold 4 --test-test-epoch 0
