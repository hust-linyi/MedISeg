# change the pretrained model with the parameters 'name', e.g., res50, res50_1k, res50_21k, res50_moco, res50_simclr.
# add 'name' for ViT: ViT-B_16_mae, ViT-B_32_mae, ViT-L_16_mae, ViT-L_32_mae, ViT-H_14_mae, ViT-B_16_mocov3, ViT-B_32_mocov3
GPU=5
MAX_FOLD=4
name=ViT-B_16_mae
cd ../NetworkTrainer

# FOLD 0-4
for fold_id in $(seq 0 1 $MAX_FOLD)
do  
python train.py --task pretrain --fold $fold_id --train-gpus $GPU --pretrained True --name $name
python test.py --task pretrain --fold $fold_id --test-test-epoch 0 --test-gpus $GPU --name $name
done