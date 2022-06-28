cd /newdata/ianlin/CODE/seg_trick/2d_unet/ensembleinit

python train.py --task ensembleinit --train-seed 2025 --fold 1 
python test.py --task ensembleinit --train-seed 2025 --fold 1 --test-save-flag True
