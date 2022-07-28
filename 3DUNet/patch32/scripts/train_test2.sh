cd /newdata/ianlin/CODE/seg_trick/3DUNet/NetworkTrainer

python train.py --task patch32 --fold 2 --train-gpus 0 --patch-size 32
python test.py --task patch32 --fold 2 --test-gpus 0 --test-test-epoch 0 --patch-size 32