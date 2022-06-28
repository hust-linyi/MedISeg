cd /newdata/ianlin/CODE/seg_trick/3d_unet/tta

python test_da2.py --task tta_da2 --fold 0 --test-test-epoch 0 --test-flip True --test-rotate True
python test_da2.py --task tta_da2 --fold 1 --test-test-epoch 0 --test-flip True --test-rotate True
python test_da2.py --task tta_da2 --fold 2 --test-test-epoch 0 --test-flip True --test-rotate True
python test_da2.py --task tta_da2 --fold 3 --test-test-epoch 0 --test-flip True --test-rotate True
python test_da2.py --task tta_da2 --fold 4 --test-test-epoch 0 --test-flip True --test-rotate True
