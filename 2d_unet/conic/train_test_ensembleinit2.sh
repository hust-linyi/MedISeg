cd /newdata/ianlin/CODE/seg_trick/2d_unet/ensembleinit

python train.py --task ensembleinit --train-seed 2023 --fold 2 --dataset conic
python train.py --task ensembleinit --train-seed 2024 --fold 2 --dataset conic
python train.py --task ensembleinit --train-seed 2025 --fold 2 --dataset conic 
python train.py --task ensembleinit --train-seed 2026 --fold 2 --dataset conic 

python test.py --task ensembleinit --train-seed 2023 --fold 2 --test-save-flag True --dataset conic
python test.py --task ensembleinit --train-seed 2024 --fold 2 --test-save-flag True --dataset conic 
python test.py --task ensembleinit --train-seed 2025 --fold 2 --test-save-flag True --dataset conic 
python test.py --task ensembleinit --train-seed 2026 --fold 2 --test-save-flag True --dataset conic 
