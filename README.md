# Geoseg - An Package for Automatic Building Segmentation and Outline extraction via  Deep Learning

## Structure of directories
### sub directories
```
├── dataset/
│   └── NewZealand
├── logs/
│   ├── learning curve, loging, statistic etc.
├── models/
│   ├── fcn, fpn, u-nt, segnet, etc.
├── result/
│   ├── comparison
│   ├── excel
│   └── single
├── utils/
│   ├── datasets.py
│   ├── metrics.py
│   ├── preprocess.py
│   ├── runner.py
│   └── vision.py
```
### Files used for training model
```
├── FCNs.py
├── FPN.py
├── UNet.py
├── MC-FCN.py
├── BR-Net.py
├── ResUNet.py
...
```
### Files for generate visualization
```
├── visSingle.py
├── visSingleComparison.py
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
![FCN8s training curve](./logs/curve/FCN8s_iter_5000.png)

### Segmentation and outline extraction
* FCN8s
![FCN8s segmentation maps](./result/single/FCN8s_canny_segmap_edge_0.png)

### Segmentation result comparison
* FCN32s, FCN16s, FCN8s
![FCN8s, FCN16s, FCN32s](./result/single-comparison/segmap_FCN32s_FCN16s_FCN8s_0.png)

### Edge extraction result comparison
* FCN32s, FCN16s, FCN8s
![FCN8s, FCN16s, FCN32s](./result/single-comparison/edge_FCN32s_FCN16s_FCN8s_0.png)
