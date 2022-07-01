cd /newdata/ianlin/CODE/seg_trick/2d_unet/lossohem

python train.py --task lossohem --fold 0 --train-loss ohem --name res101
python train.py --task lossohem --fold 1 --train-loss ohem --name res101
python train.py --task lossohem --fold 2 --train-loss ohem --name res101
python train.py --task lossohem --fold 3 --train-loss ohem --name res101
python train.py --task lossohem --fold 4 --train-loss ohem --name res101

python test.py --task lossohem --fold 0 --test-test-epoch 0 --name res101
python test.py --task lossohem --fold 1 --test-test-epoch 0 --name res101
python test.py --task lossohem --fold 2 --test-test-epoch 0 --name res101
python test.py --task lossohem --fold 3 --test-test-epoch 0 --name res101
python test.py --task lossohem --fold 4 --test-test-epoch 0 --name res101
