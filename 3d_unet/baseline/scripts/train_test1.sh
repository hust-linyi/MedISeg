cd ..

python train.py --fold 1 --train-gpus 1
python test.py --fold 1 --test-gpus 1