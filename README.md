# Geoseg - A Computer Vision Package for Automatic Building Segmentation and Outline extraction

## Table of Contents
- <a href='#organization'>Organization</a>
- <a href='#usage'>Usage</a>
- <a href='#performance'>Performance</a>
- <a href='#visualization'>Visualization</a>
- <a href='#todo'>TODO</a>
- <a href='#citation'>Citation</a>


### Organization of Geoseg

- Sub-directories
```
  ├── dataset/
  │   └── train, validate and test dataset
  ├── logs/
  │   ├── learning curve, statistic, etc.
  ├── models/
  │   ├── fcn, fpn, u-net, segnet, etc.
  ├── result/
  │   └── quantitative & qualitative result
  ├── utils/
  │   ├── datasets.py
  │   ├── metrics.py
  │   ├── preprocess.py
  │   ├── runner.py
  │   └── vision.py
```
- Files for training specific models
```
  ├── FCNs.py
  ├── FPN.py
  ├── UNet.py
  ├── MC-FCN.py
  ├── BR-Net.py
  ├── ResUNet.py
...
```
- Files for evaluation and visualization
```
├── visSingle.py
├── visSingleComparison.py
...
```
### Usage

- Download repo.
- Download dataset
- Download pre-trainded models

We provide an training dataset [LINK]
(https://drive.google.com/file/d/1boGcJz9TyK9XB4GUhjCHVu8XGtbgjjbi/view?usp=sharing)

The location, scale and resolution of the dataset please refer to paper:
```
@article{wu2018boundary,
  title={A boundary regulated network for accurate roof segmentation and outline extraction},
  author={Wu, Guangming and Guo, Zhiling and Shi, Xiaodan and Chen, Qi and Xu, Yongwei and Shibasaki, Ryosuke and Shao, Xiaowei},
  journal={Remote Sensing},
  volume={10},
  number={8},
  pages={1195},
  year={2018},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```
### Performance

1. Performance

![performance](./result/excel/performance.png)

2. Computational efficiency
![time](./result/excel/computational-efficiency.png)

### Visualization

- Learning Curve
![FCN8s training curve](./logs/curve/FCN8s_iter_5000.png)

- Segmentation and outline extraction
![FCN8s segmentation maps](./result/single/FCN8s_canny_segmap_edge_1.png)

### TODO
- Update to pytorch 0.4.0
- Add support for more dataset

### Citation

If it helps, please cite the paper.[LINK](https://arxiv.org/pdf/1809.03175.pdf)
```
@article{wu2018geoseg,
  title={Geoseg: A Computer Vision Package for Automatic Building Segmentation and Outline Extraction},
  author={Wu, Guangming and Guo, Zhiling},
  journal={arXiv preprint arXiv:1809.03175},
  year={2018}
}
```
