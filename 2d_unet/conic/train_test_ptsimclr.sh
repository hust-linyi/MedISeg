cd /newdata/ianlin/CODE/seg_trick/2d_unet/pt

python train.py --task ptsimclr --fold 0 --pretrained True --name res50_simclr --dataset conic
python train.py --task ptsimclr --fold 1 --pretrained True --name res50_simclr --dataset conic
python train.py --task ptsimclr --fold 2 --pretrained True --name res50_simclr --dataset conic
python train.py --task ptsimclr --fold 3 --pretrained True --name res50_simclr --dataset conic
python train.py --task ptsimclr --fold 4 --pretrained True --name res50_simclr --dataset conic

python test.py --task ptsimclr --fold 0  --name res50_simclr --dataset conic
python test.py --task ptsimclr --fold 1  --name res50_simclr --dataset conic
python test.py --task ptsimclr --fold 2  --name res50_simclr --dataset conic
python test.py --task ptsimclr --fold 3  --name res50_simclr --dataset conic
python test.py --task ptsimclr --fold 4  --name res50_simclr --dataset conic
