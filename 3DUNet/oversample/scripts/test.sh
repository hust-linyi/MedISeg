cd /newdata/ianlin/CODE/seg_trick/3DUNet/NetworkTrainer

python test.py --task oversample --fold 0 --test-gpus 0 --test-test-epoch 0 --patch-size 32
python test.py --task oversample --fold 1 --test-gpus 0 --test-test-epoch 0 --patch-size 32
python test.py --task oversample --fold 2 --test-gpus 0 --test-test-epoch 0 --patch-size 32
python test.py --task oversample --fold 3 --test-gpus 0 --test-test-epoch 0 --patch-size 32
python test.py --task oversample --fold 4 --test-gpus 0 --test-test-epoch 0 --patch-size 32
