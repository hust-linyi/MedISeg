cd /newdata/ianlin/CODE/seg_trick/2d_unet/da3

python train.py --task da3 --fold 0 --name res101
python train.py --task da3 --fold 1 --name res101
python train.py --task da3 --fold 2 --name res101
python train.py --task da3 --fold 3 --name res101
python train.py --task da3 --fold 4 --name res101

python test.py --task da3 --fold 0 --name res101
python test.py --task da3 --fold 1 --name res101
python test.py --task da3 --fold 2 --name res101
python test.py --task da3 --fold 3 --name res101
python test.py --task da3 --fold 4 --name res101
