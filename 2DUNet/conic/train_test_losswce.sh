cd /newdata/ianlin/CODE/seg_trick/2d_unet/losswce

# python train\.py --task losswce --fold 0 --train-loss wce --dataset conic
# python train\.py --task losswce --fold 1 --train-loss wce --dataset conic
# python train\.py --task losswce --fold 2 --train-loss wce --dataset conic
# python train\.py --task losswce --fold 3 --train-loss wce --dataset conic
# python train\.py --task losswce --fold 4 --train-loss wce --dataset conic

python test.py --task losswce --fold 0  --dataset conic
python test.py --task losswce --fold 1  --dataset conic
python test.py --task losswce --fold 2  --dataset conic
python test.py --task losswce --fold 3  --dataset conic
python test.py --task losswce --fold 4  --dataset conic
