cd /home/ylindq/Code/seg_trick/2d_unet/ensembleinit

python train.py --fold 3 --train-seed 2023 --train-gpus 3
python train.py --fold 3 --train-seed 2024 --train-gpus 3
python train.py --fold 3 --train-seed 2025 --train-gpus 3
python train.py --fold 3 --train-seed 2026 --train-gpus 3 

python test.py --fold 3 --train-seed 2023 --test-gpus 3
python test.py --fold 3 --train-seed 2024 --test-gpus 3
python test.py --fold 3 --train-seed 2025 --test-gpus 3
python test.py --fold 3 --train-seed 2026 --test-gpus 3 
