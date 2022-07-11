cd /newdata/ianlin/CODE/seg_trick/2d_unet/tta

python test_da1.py --task ttada1 --fold 0  --test-flip True --test-rotate True
python test_da1.py --task ttada1 --fold 1  --test-flip True --test-rotate True
python test_da1.py --task ttada1 --fold 2  --test-flip True --test-rotate True
python test_da1.py --task ttada1 --fold 3  --test-flip True --test-rotate True
python test_da1.py --task ttada1 --fold 4  --test-flip True --test-rotate True
