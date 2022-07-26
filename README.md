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
The modern MIS publications usually focus on presentations of the major technologies while exorbitantly ignoring some marginal implementation details, leading to a potential problem of the unfair experimental result comparisons. In this work, we collect a series of MIS tricks for different implementation phases, and experimentally explore the effectiveness of these tricks on the consistent CNNs baselines. Witnessed by the extensive experimental results on both the representative 2D and 3D medical image datasets, we explicitly clarify the effect of these tricks on MIS.
</div>

[comment]: <> ()
![visualization](figures/fig1.png)
<div align="center">
The surveyed medical image segmentation tricks and their latent relations 
</div>

## Citation
🌻 If you use this toolbox or benchmark in your research, please cite:
```
@article{zhangbatmis2022,
  title={Bag of Tricks with Convolutional Neural Networks for Medical Image Segmentation},
  author={Zhang, Dong and Lin, Yi and Chen, Hao and Tian, Zhuotao and Yang, Xin and Tang, Jinhui and Cheng, Kwang-Ting},
  journal={arXiv},
  year={2022}
}
```

## News
🌻 1.1.0 was released in 01/08/2022

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

## Installation
- **Option 1:**
```python
pip install -r requirements.txt
```
- **Option 2:** 
```python
conda install ipython
pip install albumentations
pip install torch
pip install opencv-python
pip install imageio
pip install ftfy regex tqdm
pip install altair
pip install streamlit
pip install --upgrade protobuf
pip install timm
pip install tensorboardX
pip install matplotlib
pip install test-tube
pip install wandb
```

## Data Preparation
Please download datasets from the official website:
- [x] ISIC 2018: [2D ISIC 2018 Lesion Boundary Segmentation Dataset](https://challenge.isic-archive.com/landing/2018/)
- [x] CoNIC: [2D Colon Nuclei Identification and Counting Challenge Dataset](https://conic-challenge.grand-challenge.org/)
- [x] KiTS19: [3D Kidney Tumor Segmentation 2019 Dataset](https://kits19.grand-challenge.org/data/)

## Inference with Pre-trained Models


## Training & Evaluation


## Model Zoo

Dataset  | Baseline | Method | Recall (%) | Percision (%) |  Dice (%) |  IoU (%) | Weight
------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
[ISIC 2018](https://challenge.isic-archive.com/landing/2018/)  | 2D-UNet | + PyTorch | 85.80%  | 85.80%  | 85.80% | 85.80%  | weight
[CoNIC](https://conic-challenge.grand-challenge.org/)  | 2D-UNet | + PyTorch | 87.07%  | 85.80%   | 85.80% | 85.80% | weight 
[KiTS19](https://kits19.grand-challenge.org/data/) | 3D-UNet | + PyTorch  | 87.35% | 87.35%  | 87.35% | 87.35% | weight


## Todo list
- [ ] Experiments on more medical image datasets 

- [ ] Experiments on other backbone networks 

- [ ] Experiments on more MIS tricks

- [ ] Other interesting attempts


🌻 We welcome more like-minded friends to join in this project and continue to expand this storage

🌻 If you have any suggestions or comments please let us know

🌻 If you have any problems in using this code, please contact: dongz@ust.hk or yi.lin@connect.ust.hk
