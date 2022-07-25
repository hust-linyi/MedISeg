cd /newdata/ianlin/CODE/seg_trick/2d_unet/pt

python train.py --task pt1k --pretrained True --name res50_1k  --dataset conic --fold 2
python test.py --task pt1k --name res50_1k  --dataset conic --fold 2