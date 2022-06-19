cd ..

python train.py --fold 2 --train-gpus 2
python test.py --fold 2 --test-gpus 2