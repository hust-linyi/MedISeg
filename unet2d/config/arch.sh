# we provide various architectures for 2DUNet, including:
# ResNet ==> res18, res34, res50, res101, res152;
# DenseNet ==> dense121, dense161, dense169, dense201;
# Vit ==> R50-ViT-B_16, R50-ViT-L_16, ViT-B_16, ViT-B_32, ViT-H_14, ViT-L_16, ViT-L_32;
GPU=4
MAX_FOLD=4
name=ViT-B_16
cd ../NetworkTrainer

# FOLD 0-4
for fold_id in $(seq 0 1 $MAX_FOLD)
do  
python train.py --task debug --name $name --fold $fold_id --train-gpus $GPU 
python test.py --task debug --name $name --fold $fold_id --test-test-epoch 0 --test-gpus $GPU
done