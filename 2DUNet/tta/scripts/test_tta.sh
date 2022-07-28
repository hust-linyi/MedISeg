cd /newdata/ianlin/CODE/seg_trick/2d_unet/tta

python test.py --task tta --fold 0  --test-flip True --test-rotate True
python test.py --task tta --fold 1  --test-flip True --test-rotate True
python test.py --task tta --fold 2  --test-flip True --test-rotate True
python test.py --task tta --fold 3  --test-flip True --test-rotate True
python test.py --task tta --fold 4  --test-flip True --test-rotate True
