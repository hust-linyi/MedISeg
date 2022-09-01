
GPU=2
MAX_FOLD=4
PATCH_SIZE=32 # you may choose 32, 64, 96, 128, 160, 192, etc.
cd ../NetworkTrainer


# FOLD 0-4
for fold_id in $(seq 0 1 $MAX_FOLD)
do  
python train.py --task patch${PATCH_SIZE} --fold $fold_id --train-gpus $GPU  --patch-size ${PATCH_SIZE}
python test.py --task patch${PATCH_SIZE} --fold $fold_id --test-test-epoch 0 --test-gpus $GPU  --patch-size ${PATCH_SIZE}
done