cd ..

python train.py --fold 3 --train-gpus 3
python test.py --fold 3 --test-gpus 3