cd /newdata/ianlin/CODE/seg_trick/2d_unet/ensembleinit

python train.py --task ensembleinit --train-seed 2023 --fold 1 --name res101
python train.py --task ensembleinit --train-seed 2024 --fold 1 --name res101
python train.py --task ensembleinit --train-seed 2025 --fold 1 --name res101 
python train.py --task ensembleinit --train-seed 2026 --fold 1 --name res101 

python test.py --task ensembleinit --train-seed 2023 --fold 1 --test-save-flag True --name res101
python test.py --task ensembleinit --train-seed 2024 --fold 1 --test-save-flag True --name res101 
python test.py --task ensembleinit --train-seed 2025 --fold 1 --test-save-flag True --name res101 
python test.py --task ensembleinit --train-seed 2026 --fold 1 --test-save-flag True --name res101 