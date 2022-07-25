cd /newdata/ianlin/CODE/seg_trick/2d_unet/arch

python train.py --task arch --fold 0  --name dense121
python train.py --task arch --fold 1  --name dense121
python train.py --task arch --fold 2  --name dense121
python train.py --task arch --fold 3  --name dense121
python train.py --task arch --fold 4  --name dense121

python test.py --task arch --fold 0  --name dense121
python test.py --task arch --fold 1  --name dense121
python test.py --task arch --fold 2  --name dense121
python test.py --task arch --fold 3  --name dense121
python test.py --task arch --fold 4  --name dense121
