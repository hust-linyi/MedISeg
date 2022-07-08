cd /newdata/ianlin/CODE/seg_trick/2d_unet/pt

python train.py --task pt1k --fold 0 --pretrained True --dataset conic_1k
python train.py --task pt1k --fold 1 --pretrained True --dataset conic_1k
python train.py --task pt1k --fold 2 --pretrained True --dataset conic_1k
python train.py --task pt1k --fold 3 --pretrained True --dataset conic_1k
python train.py --task pt1k --fold 4 --pretrained True --dataset conic_1k

python test.py --task pt1k --fold 0  --dataset conic_1k
python test.py --task pt1k --fold 1  --dataset conic_1k
python test.py --task pt1k --fold 2  --dataset conic_1k
python test.py --task pt1k --fold 3  --dataset conic_1k
python test.py --task pt1k --fold 4  --dataset conic_1k
