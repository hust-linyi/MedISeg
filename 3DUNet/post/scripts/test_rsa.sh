cd /newdata/ianlin/CODE/seg_trick/3DUNet/post

python test.py --task rsa --fold 0 --test-test-epoch 0 --post-rsa True
python test.py --task rsa --fold 1 --test-test-epoch 0 --post-rsa True
python test.py --task rsa --fold 2 --test-test-epoch 0 --post-rsa True
python test.py --task rsa --fold 3 --test-test-epoch 0 --post-rsa True
python test.py --task rsa --fold 4 --test-test-epoch 0 --post-rsa True
