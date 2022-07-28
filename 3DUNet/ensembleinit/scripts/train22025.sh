cd /newdata/ianlin/CODE/seg_trick/3DUNet/ensembleinit

python train.py --task ensembleinit --train-seed 2025 --fold 2 
python test.py --task ensembleinit --train-seed 2025 --fold 2 --test-save-flag True
