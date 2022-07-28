cd /newdata/ianlin/CODE/seg_trick/3d_unet/ttaensemble

python test.py --task ttaensemble --fold 0 --train-seed 2024 --test-flip True --test-save-flag True
python test.py --task ttaensemble --fold 1 --train-seed 2024 --test-flip True --test-save-flag True
python test.py --task ttaensemble --fold 2 --train-seed 2024 --test-flip True --test-save-flag True
python test.py --task ttaensemble --fold 3 --train-seed 2024 --test-flip True --test-save-flag True
python test.py --task ttaensemble --fold 4 --train-seed 2024 --test-flip True --test-save-flag True
