cd /newdata/ianlin/CODE/seg_trick/2d_unet/baseline

# python train.py --task baseline --fold 0 --dataset conic
# python train.py --task baseline --fold 1 --dataset conic
# python train.py --task baseline --fold 2 --dataset conic
# python train.py --task baseline --fold 3 --dataset conic
# python train.py --task baseline --fold 4 --dataset conic

python test.py --task baseline --fold 0  --dataset conic --test-save-flag True
python test.py --task baseline --fold 1  --dataset conic --test-save-flag True
python test.py --task baseline --fold 2  --dataset conic --test-save-flag True
python test.py --task baseline --fold 3  --dataset conic --test-save-flag True
python test.py --task baseline --fold 4  --dataset conic --test-save-flag True
