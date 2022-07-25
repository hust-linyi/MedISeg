cd /newdata/ianlin/CODE/seg_trick/2d_unet/losstversky

# python train\.py --task losstversky --fold 0 --train-loss tversky --dataset conic
# python train\.py --task losstversky --fold 1 --train-loss tversky --dataset conic
# python train\.py --task losstversky --fold 2 --train-loss tversky --dataset conic
# python train\.py --task losstversky --fold 3 --train-loss tversky --dataset conic
# python train\.py --task losstversky --fold 4 --train-loss tversky --dataset conic

python test.py --task losstversky --fold 0  --dataset conic
python test.py --task losstversky --fold 1  --dataset conic
python test.py --task losstversky --fold 2  --dataset conic
python test.py --task losstversky --fold 3  --dataset conic
python test.py --task losstversky --fold 4  --dataset conic
