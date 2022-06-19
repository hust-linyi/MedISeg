cd ..

python train.py --task da2 --fold 0 --train-gpus 0
python test.py --task da2 --fold 0 --test-gpus 0