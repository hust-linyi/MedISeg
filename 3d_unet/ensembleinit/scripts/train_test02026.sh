cd /newdata/ianlin/CODE/seg_trick/3d_unet/ensembleinit

python train.py --fold 0 --train-seed 2025
python test.py --fold 0 --train-seed 2025

