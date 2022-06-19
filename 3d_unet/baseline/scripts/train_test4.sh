cd ..

python train.py --fold 4 --train-gpus 4
python test.py --fold 4 --test-gpus 4