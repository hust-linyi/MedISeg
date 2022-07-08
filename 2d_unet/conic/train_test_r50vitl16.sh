cd /newdata/ianlin/CODE/seg_trick/2d_unet/arch

python train.py --task arch --fold 0  --name R50-ViT-L_16 --dataset conic
python train.py --task arch --fold 1  --name R50-ViT-L_16 --dataset conic
python train.py --task arch --fold 2  --name R50-ViT-L_16 --dataset conic
python train.py --task arch --fold 3  --name R50-ViT-L_16 --dataset conic
python train.py --task arch --fold 4  --name R50-ViT-L_16 --dataset conic

python test.py --task arch --fold 0  --name R50-ViT-L_16 --dataset conic
python test.py --task arch --fold 1  --name R50-ViT-L_16 --dataset conic
python test.py --task arch --fold 2  --name R50-ViT-L_16 --dataset conic
python test.py --task arch --fold 3  --name R50-ViT-L_16 --dataset conic
python test.py --task arch --fold 4  --name R50-ViT-L_16 --dataset conic
