cd /newdata/ianlin/CODE/seg_trick/2d_unet/pt

python train.py --task pt21k --pretrained True --name res50_21k  --dataset conic --fold 1
python test.py --task pt21k --name res50_21k  --dataset conic --fold 1