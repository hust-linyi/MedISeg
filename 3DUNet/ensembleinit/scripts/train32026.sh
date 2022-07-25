cd /newdata/ianlin/CODE/seg_trick/2d_unet/ensembleinit

python train.py --task ensembleinit --train-seed 2026 --fold 3 
python test.py --task ensembleinit --train-seed 2026 --fold 3 --test-save-flag True
