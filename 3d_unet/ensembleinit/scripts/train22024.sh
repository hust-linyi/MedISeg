cd /newdata/ianlin/CODE/seg_trick/2d_unet/ensembleinit

python train.py --task ensembleinit --train-seed 2024 --fold 2 
python test.py --task ensembleinit --train-seed 2024 --fold 2 --test-save-flag True
