cd /newdata/ianlin/CODE/seg_trick/2d_unet/lossohem

python train.py --task lossohem --fold 0 --train-loss ohem --dataset conic
python train.py --task lossohem --fold 1 --train-loss ohem --dataset conic
python train.py --task lossohem --fold 2 --train-loss ohem --dataset conic
python train.py --task lossohem --fold 3 --train-loss ohem --dataset conic
python train.py --task lossohem --fold 4 --train-loss ohem --dataset conic

python test.py --task lossohem --fold 0  --dataset conic
python test.py --task lossohem --fold 1  --dataset conic
python test.py --task lossohem --fold 2  --dataset conic
python test.py --task lossohem --fold 3  --dataset conic
python test.py --task lossohem --fold 4  --dataset conic
