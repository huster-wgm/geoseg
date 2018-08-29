# Geoseg - An Package for Automatic Building Segmentation and Outline extraction via  Deep Learning

## Structure of directories
### Directory for dataset
```
├── dataset
│   └── NewZealand
│       ├── land
│       └── segmap

```
### Directory for saving logs
```
├── logs
│   ├── learning-curve
│   ├── statistic
│   └── training
```
### Directory for model files
```
├── models
│   ├── __init__.py
│   ├── blockunits.py
│   ├── fcn.py
│   ├── fpn.py
│   ├── lenet.py
│   ├── linknet.py
│   ├── mcfcn.py
│   ├── mtfcn.py
│   ├── resunet.py
│   └── unet.py
```
### Directory for result
```
├── result
│   ├── comparison
│   ├── excel
│   └── single
```
### Directory for utils
```
├── utils
│   ├── __init__.py
│   ├── datasets.py
│   ├── metrics.py
│   ├── preprocess.py
│   ├── runner.py
│   └── vision.py
```
### Files used for training & evaluating model
```
├── GAN.py
├── PatchGAN.py
├── cGAN.py
├── FCNs.py
├── FPN.py
├── UNet.py
├── mtFCN.py
├── LinkNet.py
├── MC-FCN.py
├── ResUNet.py
...
```
### Files for generate visualization
```
├── genArea.py
├── genComparison.py
├── genSingle.py
...
```

## Model Performance

### Accuracy Performance
* Overall-accuracy
![oa](./result/excel/overall-accuracy.png)
* Precision
![precision](./result/excel/precision.png)
* Recall
![recall](./result/excel/recall.png)
* Kappa
![kappa](./result/excel/kappa.png)
* Jaccard
![jaccard](./result/excel/jaccard.png)

### Computational Performance
* Time cost
![time](./result/excel/prediction%20time.png)
* FPS
![fps](./result/excel/prediction%20fps.png)

## Visualization Samples

### Learning Curve
* FCN8s
![FCN8s training curve](./logs/learning-curve/FCN8s_iter_10000.png)

### Segmentation and ouline extraction
* FCN8s
![FCN8s segmentation maps](./result/single/FCN8s_segmap_edge_0.png)

### Segmentation result comparison
* FCN8s, FCN16s, FCN32s
![FCN8s, FCN16s, FCN32s](./result/single-comparison/segmap_FCN8s_FCN16s_FCN32s_0.png)

### Edge extraction result comparison
* FCN8s, FCN16s, FCN32s
![FCN8s, FCN16s, FCN32s](./result/single-comparison/edge_FCN8s_FCN16s_FCN32s_0.png)
