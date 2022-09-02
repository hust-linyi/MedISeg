# change the pretrained model with the parameters 'name', e.g., res50, res50_1k, res50_21k, res50_moco, res50_simclr.
GPU=0
MAX_FOLD=4
cd ../NetworkTrainer

# FOLD 0-4
for fold_id in $(seq 0 1 $MAX_FOLD)
do  
python train.py --task pretrain --fold $fold_id --train-gpus $GPU --pretrained True --name res50
python test.py --task pretrain --fold $fold_id --test-test-epoch 0 --test-gpus $GPU
done