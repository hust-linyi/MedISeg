cd /newdata/ianlin/CODE/seg_trick/2d_unet/post

python test.py --task abl --fold 0  --post-abl True --dataset conic
python test.py --task abl --fold 1  --post-abl True --dataset conic
python test.py --task abl --fold 2  --post-abl True --dataset conic
python test.py --task abl --fold 3  --post-abl True --dataset conic
python test.py --task abl --fold 4  --post-abl True --dataset conic
