cd /newdata/ianlin/CODE/seg_trick/2d_unet/da3

python train.py --task da3 --fold 0 
python train.py --task da3 --fold 1
python train.py --task da3 --fold 2
python train.py --task da3 --fold 3
python train.py --task da3 --fold 4

python test.py --task da3 --fold 0 --test-test-epoch 0
python test.py --task da3 --fold 1 --test-test-epoch 0
python test.py --task da3 --fold 2 --test-test-epoch 0
python test.py --task da3 --fold 3 --test-test-epoch 0
python test.py --task da3 --fold 4 --test-test-epoch 0
