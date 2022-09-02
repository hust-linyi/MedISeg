# If you want to inference with the trained model, you need to manually modify the path of the trained model.
# For example, in NetworkTrainer/options/options_liver.py, line 87: self.test['checkpoint_dir'] 
GPU=0
MAX_FOLD=4
cd /home/ylindq/CODE/seg_trick/3DUNet/tta

# FOLD 0-4
for fold_id in $(seq 0 1 $MAX_FOLD)
do
python test.py --task tta --fold $fold_id --train-gpus $GPU --test-flip True --test-rotate True  
done