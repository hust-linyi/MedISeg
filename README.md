[comment]: <> ()
![visualization](figures/logo.png)

<div align="center">

[🆕What's New](#News) |
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

## Authors
* [Dong Zhang](https://dongzhang89.github.io/)
* [Yi Lin](https://ianyilin.github.io/)
* [Hao Chen](https://cse.hkust.edu.hk/admin/people/faculty/profile/jhc)
* [Zhuotao Tian](https://scholar.google.com/citations?user=mEjhz-IAAAAJ&hl=zh-TW)
* [Xin Yang](https://scholar.google.com/citations?user=lsz8OOYAAAAJ&hl=zh-CN)
* [Jinhui Tang](https://scholar.google.com/citations?user=ByBLlEwAAAAJ&hl=zh-CN)
* [Kwang-Ting Cheng](https://seng.hkust.edu.hk/about/people/faculty/tim-kwang-ting-cheng)

## News
**1.1.0** was released in 01/08/2022:  
- A series of MIS tricks are collected and experimentally explored on [2D-UNet](https://arxiv.org/abs/1505.04597) and [3D-UNet](https://arxiv.org/abs/1606.06650).
- We explicitly clarify the effectiveness of tricks on [ISIC 2018](https://challenge.isic-archive.com/landing/2018/), [CoNIC](https://conic-challenge.grand-challenge.org/), [KiTS19](https://kits19.grand-challenge.org/data/) datasets. 
- BaT-MIS will provide guidance for a wide range of medical image processing challenges in the future.

## Installation
### 🌻 Option 1: 
```python
pip install -r requirements.txt
```
### 🌻 Option 2: 
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
🌻 Please download the dataset from the official website:
* ISIC 2018: [2D ISIC 2018 Lesion Boundary Segmentation Dataset](https://challenge.isic-archive.com/landing/2018/)
* CoNIC: [2D Colon Nuclei Identification and Counting Challenge Dataset](https://conic-challenge.grand-challenge.org/)
* KiTS19: [3D Kidney Tumor Segmentation 2019 Dataset](https://kits19.grand-challenge.org/data/)

## Inference with Pre-trained Models


## Training & Evaluation


## Model Zoo

Dataset  | Baseline | Method | Recall (%) | Percision (%) |  Dice (%) |  IoU (%) | Weight
------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
[ISIC 2018](https://challenge.isic-archive.com/landing/2018/)  | 2D-UNet | + PyTorch | 85.80%  | 85.80%  | 85.80% | 85.80%  | weight
[CoNIC](https://conic-challenge.grand-challenge.org/)  | 2D-UNet | + PyTorch | 87.07%  | 85.80%   | 85.80% | 85.80% | weight 
[KiTS19](https://kits19.grand-challenge.org/data/) | 3D-UNet | + PyTorch  | 87.35% | 87.35%  | 87.35% | 87.35% | weight


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

## Todo list
- [ ] Experiments on more medical image datasets 

- [ ] Experiments on other backbone networks 

- [ ] Experiments on more MIS tricks

- [ ] Other interesting attempts


🌻 We welcome more like-minded friends to join in this project and continue to expand this storage

🌻 If you have any suggestions or comments please let us know

🌻 If you have any problems in using this code, please contact: dongz@ust.hk or yi.lin@connect.ust.hk
