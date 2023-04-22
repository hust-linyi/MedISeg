
GPU=0
MAX_FOLD=4
SEED=2022 # In our exepriment, we use different seed for enesmble, i.e, from 2022 to 2026.
cd ../3DUNet/ensembleinit

# FOLD 0-4
for fold_id in $(seq 0 1 $MAX_FOLD)
do  
python train.py --task ensembleinit --train-seed SEED --fold $fold_id --train-gpus $GPU 
python test.py --task ensembleinit  --train-seed SEED --test-save-flag True --fold $fold_id --test-test-epoch 0 --test-gpus $GPU
done
