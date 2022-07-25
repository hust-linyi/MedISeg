cd /newdata/ianlin/CODE/seg_trick/2d_unet/lossfocal

# python train\.py --task lossfocal --fold 0 --train-loss focal --dataset conic
# python train\.py --task lossfocal --fold 1 --train-loss focal --dataset conic
# python train\.py --task lossfocal --fold 2 --train-loss focal --dataset conic
# python train\.py --task lossfocal --fold 3 --train-loss focal --dataset conic
# python train\.py --task lossfocal --fold 4 --train-loss focal --dataset conic

python test.py --task lossfocal --fold 0  --dataset conic
python test.py --task lossfocal --fold 1  --dataset conic
python test.py --task lossfocal --fold 2  --dataset conic
python test.py --task lossfocal --fold 3  --dataset conic
python test.py --task lossfocal --fold 4  --dataset conic
