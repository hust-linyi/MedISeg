cd /newdata/ianlin/CODE/seg_trick/2d_unet/da4

python train.py --task da4 --fold 0 --dataset conic
python train.py --task da4 --fold 1 --dataset conic
python train.py --task da4 --fold 2 --dataset conic
python train.py --task da4 --fold 3 --dataset conic
python train.py --task da4 --fold 4 --dataset conic

python test.py --task da4 --fold 0 --dataset conic
python test.py --task da4 --fold 1 --dataset conic
python test.py --task da4 --fold 2 --dataset conic
python test.py --task da4 --fold 3 --dataset conic
python test.py --task da4 --fold 4 --dataset conic
