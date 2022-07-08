cd /newdata/ianlin/CODE/seg_trick/2d_unet/pt

python train.py --task pt --fold 0 --pretrained True --dataset conic
python train.py --task pt --fold 1 --pretrained True --dataset conic
python train.py --task pt --fold 2 --pretrained True --dataset conic
python train.py --task pt --fold 3 --pretrained True --dataset conic
python train.py --task pt --fold 4 --pretrained True --dataset conic

python test.py --task pt --fold 0  --dataset conic
python test.py --task pt --fold 1  --dataset conic
python test.py --task pt --fold 2  --dataset conic
python test.py --task pt --fold 3  --dataset conic
python test.py --task pt --fold 4  --dataset conic
