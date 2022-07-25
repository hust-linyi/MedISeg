# BaT-MIS

The repo contains the official PyTorch Implementation for paper:  
[Bag of Tricks with Convolutional Neural Networks for Medical Image Segmentation](https://arxiv.org/list/cs.CV/recent)


[comment]: <> ()
### An illustration of the surveyed MIS tricks and their latent relations  
We separate an medical image segmentation model into six implementation phases, which include pre-training model, data pre-processing, data augmentation, model implementation, model inference, and result post-processing. For each trick, we experimentally explore its effectiveness on the consistent CNNs segmentation baselines including 2D-UNet and 3D-UNet on three medical image segmentation datasets.

[comment]: <> ()
![visualization](figures/fig1.png)

###  Authors:
* [Dong Zhang](https://dongzhang89.github.io/)
* [Yi Lin](https://ianyilin.github.io/)
* [Hao Chen](https://cse.hkust.edu.hk/admin/people/faculty/profile/jhc)
* [Zhuotao Tian](https://scholar.google.com/citations?user=mEjhz-IAAAAJ&hl=zh-TW)
* [Xin Yang](https://scholar.google.com/citations?user=lsz8OOYAAAAJ&hl=zh-CN)
* [Jinhui Tang](https://scholar.google.com/citations?user=ByBLlEwAAAAJ&hl=zh-CN)
* [Kwang-Ting Cheng](https://seng.hkust.edu.hk/about/people/faculty/tim-kwang-ting-cheng)

### Installation
#### 🌻 Option 1: 
```python
pip install -r requirements.txt
```
#### 🌻 Option 2: 
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

### Data Preparation
🌻 Please download the dataset from the official website:
* [ISIC 2018: 2D ISIC 2018 Lesion Boundary Segmentation Dataset](https://challenge.isic-archive.com/landing/2018/)
* [CoNIC: 2D Colon Nuclei Identification and Counting Challenge Dataset](https://conic-challenge.grand-challenge.org/)
* [KiTS19: 3D Kidney Tumor Segmentation 2019 Dataset](https://kits19.grand-challenge.org/data/)

### Inference with Pre-trained Models


### Training & Evaluation


### Model Zoo

Dataset  | Baseline | Method | Recall (%) | Percision (%) |  Dice (%) |  IoU (%) | Weight
------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
[ISIC 2018](https://challenge.isic-archive.com/landing/2018/)  | 2D-UNet | + PyTorch | 85.80%  | 85.80%  | 85.80% | 85.80%  | weight
[CoNIC](https://conic-challenge.grand-challenge.org/)  | 2D-UNet | + PyTorch | 87.07%  | 85.80%   | 85.80% | 85.80% | weight 
[KiTS19](https://kits19.grand-challenge.org/data/) | 3D-UNet | + PyTorch  | 87.35% | 87.35%  | 87.35% | 87.35% | weight


### Citation
🌻 If you find this repo useful, please cite:
```
@article{zhangbatmis2022,
  title={Bag of Tricks with Convolutional Neural Networks for Medical Image Segmentation},
  author={Zhang, Dong and Lin, Yi and Chen, Hao and Tian, Zhuotao and Yang, Xin and Tang, Jinhui and Cheng, Kwang-Ting},
  journal={arXiv},
  year={2022}
}
```

🌻 If you have any problems in using this code, please contact: dongz@ust.hk or yi.lin@connect.ust.hk
