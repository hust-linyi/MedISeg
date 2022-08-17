cd /newdata/ianlin/CODE/seg_trick/3DUNet/ensembleinit

python train.py --task ensembleinit --train-seed 2022 --fold 3 
python test.py --task ensembleinit --train-seed 2022 --fold 3 --test-save-flag True
