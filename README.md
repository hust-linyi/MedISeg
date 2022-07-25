# BaT-MIS

The repo contains the official PyTorch Implementation for paper:  
[Bag of Tricks with Convolutional Neural Networks for Medical Image Segmentation](https://arxiv.org/list/cs.CV/recent)


[comment]: <> (![fig1]&#40;figures/fig1.png&#41;)
#### 🌻 An illustration of the surveyed MIS tricks and their latent relations  
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
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
pip install git+https://github.com/zhanghang1989/PyTorch-Encoding/
pip install pytorch-lightning==1.3.5
pip install opencv-python
pip install imageio
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
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
By default, for training, testing and demo, we use [ISIC 2018](https://challenge.isic-archive.com/landing/2018/)



### Inference  with Pre-trained Models

### Training & Evaluation


### Model Zoo

Dataset  | Baseline | Method | Recall (%) | Percision (%) |  Dice (%) |  IoU (%) | Weight
------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
[ISIC 2018](https://challenge.isic-archive.com/landing/2018/)  | 2D-UNet | + PyTorch | 85.80%  | 85.80%  | 85.80% | 85.80%  | weight
[CoNIC](https://conic-challenge.grand-challenge.org/)  | 2D-UNet | + PyTorch | 87.07%  | 85.80%   | 85.80% | 85.80% | weight 
[KiTS19](https://kits19.grand-challenge.org/data/) | 3D-UNet | + PyTorch  | 87.35% | 87.35%  | 87.35% | 87.35% | weight



### Citation


If you have any problems in using this code, please contact: dongz@ust.hk or yi.lin@connect.ust.hk


