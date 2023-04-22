
GPU=1
MAX_FOLD=4
cd ../NetworkTrainer

# FOLD 0-4
for fold_id in $(seq 0 1 $MAX_FOLD)
do  
python train.py --task lossdice --fold $fold_id --train-gpus $GPU --train-loss dice
python test.py --task lossdice --fold $fold_id --test-test-epoch 0 --test-gpus $GPU
done