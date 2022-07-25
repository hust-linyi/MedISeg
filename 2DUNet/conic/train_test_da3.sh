cd /newdata/ianlin/CODE/seg_trick/2d_unet/da3

python train.py --task da3 --fold 0 --dataset conic
python train.py --task da3 --fold 1 --dataset conic
python train.py --task da3 --fold 2 --dataset conic
python train.py --task da3 --fold 3 --dataset conic
python train.py --task da3 --fold 4 --dataset conic

python test.py --task da3 --fold 0 --dataset conic
python test.py --task da3 --fold 1 --dataset conic
python test.py --task da3 --fold 2 --dataset conic
python test.py --task da3 --fold 3 --dataset conic
python test.py --task da3 --fold 4 --dataset conic
