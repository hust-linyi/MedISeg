cd /newdata/ianlin/CODE/seg_trick/3d_unet/tta

python test_baseline.py --task tta_flip --fold 0 --test-test-epoch 0 --test-flip True
python test_baseline.py --task tta_flip --fold 1 --test-test-epoch 0 --test-flip True
python test_baseline.py --task tta_flip --fold 2 --test-test-epoch 0 --test-flip True
python test_baseline.py --task tta_flip --fold 3 --test-test-epoch 0 --test-flip True
python test_baseline.py --task tta_flip --fold 4 --test-test-epoch 0 --test-flip True