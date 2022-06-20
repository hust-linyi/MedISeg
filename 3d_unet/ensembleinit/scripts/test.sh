cd /newdata/ianlin/CODE/seg_trick/3d_unet/ensembleinit

python test.py --fold 0 --train-seed 2023 --test-test-epoch 0
python test.py --fold 0 --train-seed 2024 --test-test-epoch 0
python test.py --fold 0 --train-seed 2025 --test-test-epoch 0
python test.py --fold 0 --train-seed 2026 --test-test-epoch 0

python test.py --fold 1 --train-seed 2023 --test-test-epoch 0
python test.py --fold 1 --train-seed 2024 --test-test-epoch 0
python test.py --fold 1 --train-seed 2025 --test-test-epoch 0
python test.py --fold 1 --train-seed 2026 --test-test-epoch 0

python test.py --fold 2 --train-seed 2023 --test-test-epoch 0
python test.py --fold 2 --train-seed 2024 --test-test-epoch 0
python test.py --fold 2 --train-seed 2025 --test-test-epoch 0
python test.py --fold 2 --train-seed 2026 --test-test-epoch 0
