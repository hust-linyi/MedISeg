[comment]: <> ()
![visualization](figures/logo.png)

<div align="center">

[🆕News](#News) |
[🛠️Installation](#Installation) |
[👀Model Zoo](#Model-Zoo) |
[🤔Reporting Issues](https://github.com/hust-linyi/seg_trick/issues)

</div>

## Introduction
<div align="justify">
The modern MedISeg publications usually focus on presentations of the major contributions while unwittingly ignoring some marginal implementation tricks, leading to a potential problem of the unfair experimental result comparisons. In this work, we collect a series of MedISeg tricks for different model implementation phases, and experimentally explore the effectiveness of these tricks on the consistent baselines. Compared to the paper-driven surveys that only blandly focus on the advantage and limitation analyses, our work provides a large number of solid experiments and is more technically operable. Witnessed by the extensive experimental results on both the representative 2D and 3D medical image datasets, we explicitly clarify the effect of these tricks.
</div>

[comment]: <> ()
![visualization](figures/fig1.png)
<div align="center">
The surveyed medical image segmentation tricks and their latent relations 
</div>

## Citation
🌻 If you use this toolbox or benchmark in your research, please cite:
```
@article{zhangmediseg2022,
  title={Deep Learning for Medical Image Segmentation: Tricks, Challenges and Future Directions},
  author={Zhang, Dong and Lin, Yi and Chen, Hao and Tian, Zhuotao and Yang, Xin and Tang, Jinhui and Cheng, Kwang-Ting},
  journal={arXiv},
  year={2022}
}
```

## News
🌻 1.1.0 was released in 01/09/2022

- **Supported Backbones:**
- [x] [ResNet (CVPR'2016)](https://arxiv.org/abs/1512.03385?context=cs)
- [x] [DenseNet (CVPR'2017)](https://arxiv.org/abs/1608.06993)
- [x] [ViT (ICLR'2021)](https://arxiv.org/abs/2010.11929)

- **Supported Methods:**
- [x] [2D-UNet (MICCAI'2016)](https://arxiv.org/abs/1505.04597)
- [x] [3D-UNet (MICCAI'2016)](https://arxiv.org/abs/1606.06650)

- **Supported Datasets:**
- [x] [ISIC 2018](https://challenge.isic-archive.com/landing/2018/) 
- [x] [CoNIC](https://conic-challenge.grand-challenge.org/)
- [x] [KiTS19](https://kits19.grand-challenge.org/data/)
- [x] [LiTS](https://competitions.codalab.org/competitions/17094)

## Installation
- **Option 1:**
```python
pip install -r requirements.txt
```
- **Option 2:** 
```python
pip install albumentations
pip install ml_collections
pip install numpy 
pip install opencv-python
pip install pandas
pip install rich
pip install SimpleITK
pip install timm
pip install torch
pip install tqdm
pip install nibabel
pip install medpy
```

## Data Preparation
Please download datasets from the official website:
- [x] ISIC 2018: [2D ISIC 2018 Lesion Boundary Segmentation Dataset](https://challenge.isic-archive.com/landing/2018/)
- [x] CoNIC: [2D Colon Nuclei Identification and Counting Challenge Dataset](https://conic-challenge.grand-challenge.org/)
- [x] KiTS19: [3D Kidney Tumor Segmentation 2019 Dataset](https://kits19.grand-challenge.org/data/)
- [x] LiTS17: [3D Liver Tumor Segmentation 2017 Dataset](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation)

## Inference with Pre-trained Models
Download the trained weights from [here](#Model-Zoo). 

Run the following command for 2DUNet:
```
python 2DUNet/NetworkTrainer/test.py --test-model-path $YOUR_MODEL_PATH
```

Run the following command for 3DUNet:
```
python 3DUNet/NetworkTrainer/test.py --test-model-path $YOUR_MODEL_PATH
```

## Training & Evaluation
We provide the shell scripts for training and evaluation by 5-fold cross-validation. 

Run the following command for 2DUNet:
```
sh 2DUNet/config/baseline.sh
```

Run the following command for 3DUNet:
```
sh 3DUNet/config/baseline.sh
```
And the commands train/test with various tricks are also provided in  */config/. For the details of the segmentation tricks, please refer to the paper.

## Visualization
From top to bottom: raw image, ground truth, prediction.

[comment]: <> ()
![visualization](figures/res_isic.png)
<div align="center">
ISIC 2018 
</div>

[comment]: <> ()
![visualization](figures/res_conic.png)
<div align="center">
CoNIC
</div>

[comment]: <> ()
![visualization](figures/res_kits19.gif)
<div align="center">
KiTS19
</div>

[comment]: <> ()
![visualization](figures/res_lits17.gif)
<div align="center">
LiTS17
</div>

## Model Zoo

- Since our Google space is very limited, here we only provide a part of the weight links. 

- In each 5-fold cross-validation, here we only release a weight with a higher performance.

- The full weights can be downloaded [Baidu Netdisk](https://github.com/hust-linyi/MedISeg)

Training weights on ISIC 2018:

Dataset  | Baseline | Method | Recall (%) | Percision (%) |  Dice (%) |  IoU (%) | Weight
------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
[ISIC 2018](https://challenge.isic-archive.com/landing/2018/)  | 2DUNet | PyTorch | 88.18%  | 89.88%  | 86.89% | 85.80%  | [weight](https://drive.google.com/drive/folders/1cwvroWLmjQCvRU9qP_kMnlAVnds5wA9u?usp=sharing)
[ISIC 2018](https://challenge.isic-archive.com/landing/2018/)  | 2DUNet | + Image-21K | 90.21%  | 91.48%  | 89.38% | 88.00%  | [weight] 
[ISIC 2018](https://challenge.isic-archive.com/landing/2018/)  | 2DUNet | + GTAug-B | 88.32%  | 91.11%  | 88.07% | 86.98%  | [weight] 
[ISIC 2018](https://challenge.isic-archive.com/landing/2018/)  | 2DUNet | + CBL(Tvers) | 89.40%  | 90.19%  | 87.87% | 86.42%  | [weight] 
[ISIC 2018](https://challenge.isic-archive.com/landing/2018/)  | 2DUNet | + TTAGTAug-B | 90.21%  | 90.94%  | 88.94% | 87.59%  | [weight] 
[ISIC 2018](https://challenge.isic-archive.com/landing/2018/)  | 2DUNet | + EnsAvg | 91.08%  | 89.50%  | 88.52% | 87.21%  | [weight] 


Training weights on CoNIC:

Dataset  | Baseline | Method | Recall (%) | Percision (%) |  Dice (%) |  IoU (%) | Weight
------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
[CoNIC](https://conic-challenge.grand-challenge.org/)  | 2DUNet | PyTorch | 78.12%  | 77.25%   | 77.23% | 77.58% | [weight](https://drive.google.com/drive/folders/1Opk7fSRRj9Llxi5XhU61RIIFS30ip5HI?usp=sharing)
[CoNIC](https://conic-challenge.grand-challenge.org/)  | 2DUNet | + Image-21K | 78.79%  | 79.66%   | 78.75% | 78.91% | [weight]
[CoNIC](https://conic-challenge.grand-challenge.org/)  | 2DUNet | + GTAug-B | 79.28%  | 82.53%   | 80.33% | 80.35% | [weight]
[CoNIC](https://conic-challenge.grand-challenge.org/)  | 2DUNet | + TTAGTAug-A | 80.19%  | 80.57%   | 80.00% | 79.86% | [weight]


Training weights on KiTS19:

Dataset  | Baseline | Method | Recall (%) | Percision (%) |  Dice (%) |  IoU (%) | Weight
------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
[KiTS19](https://kits19.grand-challenge.org/data/) | 3DUNet | PyTorch  | 91.01% | 95.20%  | 92.50% | 87.35% | [weight](https://drive.google.com/drive/folders/1JjdN0peTGWAWjbjKRUvkGreakdykzlmU?usp=sharing)
[KiTS19](https://kits19.grand-challenge.org/data/) | 3DUNet | + EnsAvg  | 93.00% | 96.69%  | 94.39% | 90.02% | [weight]

Training weights on LiTS17:

Dataset  | Baseline | Method | Recall (%) | Percision (%) |  Dice (%) |  IoU (%) | Weight
------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
[LiTS17](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation) | 3DUNet | PyTorch  | 89.33% | 84.03%  | 86.11% | 76.44% | [weight](https://drive.google.com/drive/folders/1EfaXieZrX36DBnBUh8iIWvuBQSF_jd0o?usp=sharing)
[LiTS17](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation) | 3DUNet | + ModelGe  | 90.54% | 84.66%  | 86.99% | 77.67% | [weight]
[LiTS17](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation) | 3DUNet | Patching192  | 93.31% | 95.35%  | 94.08% | 89.18% | [weight]
[LiTS17](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation) | 3DUNet | + GTAug-A  | 90.28% | 84.24%  | 86.62% | 76.89% | [weight]
[LiTS17](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation) | 3DUNet | + OHEM  | 90.14% | 85.64%  | 87.35% | 78.24% | [weight]
[LiTS17](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation) | 3DUNet | + EnsAvg  | 90.21% | 88.39%  | 88.77% | 80.73% | [weight]
[LiTS17](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation) | 3DUNet | + ABL-CS  | 89.31% | 87.38%  | 87.79% | 79.13% | [weight]

## Todo list
- [ ] Experiments on more datasets 

- [ ] Experiments on other backbones 

- [ ] Experiments on more tricks

- [ ] Other interesting attempts


## Announcements
🌻 We welcome more like-minded friends to join in this project and continue to expand this storage

🌻 If you have any suggestions or comments please let us know

🌻 If you have any problems in using this code, please contact: dongz@ust.hk or yi.lin@connect.ust.hk
