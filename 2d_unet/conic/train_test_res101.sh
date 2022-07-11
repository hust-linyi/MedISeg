cd /newdata/ianlin/CODE/seg_trick/2d_unet/arch

python train.py --task arch --fold 0  --name res101 --dataset conic
python train.py --task arch --fold 1  --name res101 --dataset conic
python train.py --task arch --fold 2  --name res101 --dataset conic
python train.py --task arch --fold 3  --name res101 --dataset conic
python train.py --task arch --fold 4  --name res101 --dataset conic

python test.py --task arch --fold 0  --name res101 --dataset conic
python test.py --task arch --fold 1  --name res101 --dataset conic
python test.py --task arch --fold 2  --name res101 --dataset conic
python test.py --task arch --fold 3  --name res101 --dataset conic
python test.py --task arch --fold 4  --name res101 --dataset conic
