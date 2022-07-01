cd /newdata/ianlin/CODE/seg_trick/2d_unet/post

python test.py --task rsa --fold 0 --test-test-epoch 0 --post-rsa True --name res101
python test.py --task rsa --fold 1 --test-test-epoch 0 --post-rsa True --name res101
python test.py --task rsa --fold 2 --test-test-epoch 0 --post-rsa True --name res101
python test.py --task rsa --fold 3 --test-test-epoch 0 --post-rsa True --name res101
python test.py --task rsa --fold 4 --test-test-epoch 0 --post-rsa True --name res101
