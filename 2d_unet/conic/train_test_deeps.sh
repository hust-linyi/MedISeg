cd /newdata/ianlin/CODE/seg_trick/2d_unet/deeps

# python train\.py --task deeps --fold 0 --train-deeps True --dataset conic
# python train\.py --task deeps --fold 1 --train-deeps True --dataset conic
# python train\.py --task deeps --fold 2 --train-deeps True --dataset conic
# python train\.py --task deeps --fold 3 --train-deeps True --dataset conic
# python train\.py --task deeps --fold 4 --train-deeps True --dataset conic

python test.py --task deeps --fold 0  --dataset conic --train-deeps True
python test.py --task deeps --fold 1  --dataset conic --train-deeps True
python test.py --task deeps --fold 2  --dataset conic --train-deeps True
python test.py --task deeps --fold 3  --dataset conic --train-deeps True
python test.py --task deeps --fold 4  --dataset conic --train-deeps True
