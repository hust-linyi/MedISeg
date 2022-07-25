cd /newdata/ianlin/CODE/seg_trick/2d_unet/tta

python test.py --task tta --fold 0 --test-test-epoch 0 --test-flip True --test-rotate True --name res101
python test.py --task tta --fold 1 --test-test-epoch 0 --test-flip True --test-rotate True --name res101
python test.py --task tta --fold 2 --test-test-epoch 0 --test-flip True --test-rotate True --name res101
python test.py --task tta --fold 3 --test-test-epoch 0 --test-flip True --test-rotate True --name res101
python test.py --task tta --fold 4 --test-test-epoch 0 --test-flip True --test-rotate True --name res101
