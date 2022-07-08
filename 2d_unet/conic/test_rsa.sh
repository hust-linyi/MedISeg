cd /newdata/ianlin/CODE/seg_trick/2d_unet/post

python test.py --task rsa --fold 0  --post-rsa True --dataset conic
python test.py --task rsa --fold 1  --post-rsa True --dataset conic
python test.py --task rsa --fold 2  --post-rsa True --dataset conic
python test.py --task rsa --fold 3  --post-rsa True --dataset conic
python test.py --task rsa --fold 4  --post-rsa True --dataset conic
