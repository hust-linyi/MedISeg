cd /newdata/ianlin/CODE/seg_trick/2d_unet/tta

python test_da2.py --task ttada2 --fold 0  --test-flip True --test-rotate True
python test_da2.py --task ttada2 --fold 1  --test-flip True --test-rotate True
python test_da2.py --task ttada2 --fold 2  --test-flip True --test-rotate True
python test_da2.py --task ttada2 --fold 3  --test-flip True --test-rotate True
python test_da2.py --task ttada2 --fold 4  --test-flip True --test-rotate True
