cd ..

python train.py --fold 0 --train-seed 2023 --train-gpus 0
python train.py --fold 0 --train-seed 2024 --train-gpus 0
python train.py --fold 0 --train-seed 2025 --train-gpus 0
python train.py --fold 0 --train-seed 2026 --train-gpus 0

python test.py --fold 0 --train-seed 2023 --test-gpus 0
python test.py --fold 0 --train-seed 2024 --test-gpus 0
python test.py --fold 0 --train-seed 2025 --test-gpus 0
python test.py --fold 0 --train-seed 2026 --test-gpus 0


