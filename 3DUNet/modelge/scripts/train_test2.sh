cd /newdata/ianlin/CODE/seg_trick/3DUNet/NetworkTrainer

python train.py --task modelge --fold 2 --pretrained True
python test.py --task modelge --fold 2 --test-test-epoch 0