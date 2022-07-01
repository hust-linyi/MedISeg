cd /newdata/ianlin/CODE/seg_trick/2d_unet/tta

python test_da2.py --task ttada2 --fold 0 --test-test-epoch 0 --test-flip True --test-rotate True --name res101
python test_da2.py --task ttada2 --fold 1 --test-test-epoch 0 --test-flip True --test-rotate True --name res101
python test_da2.py --task ttada2 --fold 2 --test-test-epoch 0 --test-flip True --test-rotate True --name res101
python test_da2.py --task ttada2 --fold 3 --test-test-epoch 0 --test-flip True --test-rotate True --name res101
python test_da2.py --task ttada2 --fold 4 --test-test-epoch 0 --test-flip True --test-rotate True --name res101
