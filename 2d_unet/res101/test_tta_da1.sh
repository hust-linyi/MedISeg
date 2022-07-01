cd /newdata/ianlin/CODE/seg_trick/2d_unet/tta

python test_da1.py --task ttada1 --fold 0 --test-test-epoch 0 --test-flip True --test-rotate True --name res101
python test_da1.py --task ttada1 --fold 1 --test-test-epoch 0 --test-flip True --test-rotate True --name res101
python test_da1.py --task ttada1 --fold 2 --test-test-epoch 0 --test-flip True --test-rotate True --name res101
python test_da1.py --task ttada1 --fold 3 --test-test-epoch 0 --test-flip True --test-rotate True --name res101
python test_da1.py --task ttada1 --fold 4 --test-test-epoch 0 --test-flip True --test-rotate True --name res101
