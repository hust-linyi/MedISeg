cd /newdata/ianlin/CODE/seg_trick/2d_unet/lossfocal

python train.py --task lossfocal --fold 0 --train-loss focal --name res101
python train.py --task lossfocal --fold 1 --train-loss focal --name res101
python train.py --task lossfocal --fold 2 --train-loss focal --name res101
python train.py --task lossfocal --fold 3 --train-loss focal --name res101
python train.py --task lossfocal --fold 4 --train-loss focal --name res101

python test.py --task lossfocal --fold 0 --test-test-epoch 0 --name res101
python test.py --task lossfocal --fold 1 --test-test-epoch 0 --name res101
python test.py --task lossfocal --fold 2 --test-test-epoch 0 --name res101
python test.py --task lossfocal --fold 3 --test-test-epoch 0 --name res101
python test.py --task lossfocal --fold 4 --test-test-epoch 0 --name res101
