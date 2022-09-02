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
The modern MedISeg publications usually focus on presentations of the major contributions while unwittingly ignoring some marginal implementation tricks, leading to a potential problem of the unfair experimental result comparisons. In this paper, we collect a series of MedISeg tricks for different model implementation phases, and experimentally explore the effectiveness of these tricks on the consistent baseline models. Compared to some paper-driven surveys that only blandly focus on the advantage and limitation analyses of segmentation models, our work provides a large number of solid experiments and is more technically operable. Witnessed by the extensive experimental results on both the representative 2D and 3D medical image datasets, we explicitly clarify the effect of these tricks.
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
- [x] LiTS17: [3D Liver Tumor Segmentation 2017 Dataset](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation)

## Inference with Pre-trained Models
Download the trained weights from [here](youtube.com). Then run the following command for 2DUNet and 3DUNet:
```
python 2DUNet/NetworkTrainer/test.py --test-model-path $YOUR_MODEL_PATH
python 3DUNet/NetworkTrainer/test.py --test-model-path $YOUR_MODEL_PATH
```

## Training & Evaluation
We provide the shell scripts for training and evaluation by 5-fold cross-validation. Please run the following command for the baseline method:
```
sh 2DUNet/config/baseline.sh
sh 3DUNet/config/baseline.sh
```
And the commands train/test with various tricks are also provided in  */config/. For the details of the segmentation tricks, please refer to the paper.

## Model Zoo

Dataset  | Baseline | Method | Recall (%) | Percision (%) |  Dice (%) |  IoU (%) | Weight
------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
[ISIC 2018](https://challenge.isic-archive.com/landing/2018/)  | 2D-UNet | + PyTorch | 88.27%  | 89.86%  | 86.96% | 85.87%  | [weight](https://drive.google.com/drive/folders/1cwvroWLmjQCvRU9qP_kMnlAVnds5wA9u?usp=sharing)
[CoNIC](https://conic-challenge.grand-challenge.org/)  | 2D-UNet | + PyTorch | 78.12%  | 77.25%   | 77.23% | 77.58% | [weight](https://drive.google.com/drive/folders/1Opk7fSRRj9Llxi5XhU61RIIFS30ip5HI?usp=sharing)
[KiTS19](https://kits19.grand-challenge.org/data/) | 3D-UNet | + PyTorch  | 92.63% | 92.99%  | 92.42% | 86.43% | [weight](https://drive.google.com/drive/folders/1JjdN0peTGWAWjbjKRUvkGreakdykzlmU?usp=sharing)
[LiTS17](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation) | 3D-UNet | + PyTorch  | 89.33% | 84.03%  | 86.11% | 76.44% | [weight](https://drive.google.com/drive/folders/1EfaXieZrX36DBnBUh8iIWvuBQSF_jd0o?usp=sharing)
 	 	 	 

## Todo list
- [ ] Experiments on more medical image datasets 

- [ ] Experiments on other backbone networks 

- [ ] Experiments on more MIS tricks

- [ ] Other interesting attempts


## Announcements
🌻 We welcome more like-minded friends to join in this project and continue to expand this storage

🌻 If you have any suggestions or comments please let us know

🌻 If you have any problems in using this code, please contact: dongz@ust.hk or yi.lin@connect.ust.hk
