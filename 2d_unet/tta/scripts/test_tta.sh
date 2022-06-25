cd /newdata/ianlin/CODE/seg_trick/2d_unet/tta

python test.py --task tta --fold 0 --test-test-epoch 0 --test-if_tta True
python test.py --task tta --fold 1 --test-test-epoch 0 --test-if_tta True
python test.py --task tta --fold 2 --test-test-epoch 0 --test-if_tta True
python test.py --task tta --fold 3 --test-test-epoch 0 --test-if_tta True
python test.py --task tta --fold 4 --test-test-epoch 0 --test-if_tta True
