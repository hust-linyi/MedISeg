cd /newdata/ianlin/CODE/seg_trick/3DUNet/ensembleinit

python train.py --task ensembleinit --train-seed 2026 --fold 1 
python test.py --task ensembleinit --train-seed 2026 --fold 1 --test-save-flag True
