cd /newdata/ianlin/CODE/seg_trick/3DUNet/insnorm

python test.py --task insnorm --fold 0 --test-gpus 0 --test-test-epoch 0 --train-norm in
python test.py --task insnorm --fold 1 --test-gpus 0 --test-test-epoch 0 --train-norm in
python test.py --task insnorm --fold 2 --test-gpus 0 --test-test-epoch 0 --train-norm in
python test.py --task insnorm --fold 3 --test-gpus 0 --test-test-epoch 0 --train-norm in
python test.py --task insnorm --fold 4 --test-gpus 0 --test-test-epoch 0 --train-norm in
