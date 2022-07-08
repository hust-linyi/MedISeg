cd /newdata/ianlin/CODE/seg_trick/2d_unet/lossdice

python train.py --task lossdice --fold 0 --train-loss dice --dataset conic
python train.py --task lossdice --fold 1 --train-loss dice --dataset conic
python train.py --task lossdice --fold 2 --train-loss dice --dataset conic
python train.py --task lossdice --fold 3 --train-loss dice --dataset conic
python train.py --task lossdice --fold 4 --train-loss dice --dataset conic

python test.py --task lossdice --fold 0  --dataset conic
python test.py --task lossdice --fold 1  --dataset conic
python test.py --task lossdice --fold 2  --dataset conic
python test.py --task lossdice --fold 3  --dataset conic
python test.py --task lossdice --fold 4  --dataset conic
