GPU=0
MAX_FOLD=4
cd ../NetworkTrainer

# FOLD 0-4
for fold_id in $(seq 0 1 $MAX_FOLD)
do  
python test.py --task abl --fold $fold_id --test-test-epoch 0 --test-gpus $GPU --post-abl True
done