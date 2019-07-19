# Model Architecture
---------------------

# Table of Contents

- <a href='#Implemented models'>Implemented models</a>
    - <a href='#FCN'>FCN</a>
    - <a href='#U-Net'>U-Net</a>
    - <a href='#SegNet'>SegNet</a>
    - <a href='#FPN'>FPN</a>
    - <a href='#ResUNet'>ResUNet</a>
    - <a href='#MCFCN'>MCFCN</a>
    - <a href='#BR-Net'>BR-Net</a>
    - <a href='#SFCN'>SFCN</a>

- <a href='#Ongoing models'>Ongoing models</a>
    - <a href='#DeconvNet'>DeconvNet</a>
    - <a href='#RefineNet'>RefineNet</a>
    - <a href='#Deeplab'>Deeplab</a>


# Implemented models


## FCN
Fully Convolutional Networks for Semantic Segmentation
> ### a> Architecture
![architecture](http://deeplearning.net/tutorial/_images/fcn.png)
> ### b> Paper
[LINK](https://arxiv.org/pdf/1411.4038.pdf)
> ### c> Source code
[LINK](./fcn.py)


## U-Net

> ### a> Architecture
 ![architecture](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)
> ### b> Paper
 [LINK](https://arxiv.org/pdf/1505.04597.pdf)
> ### c> Source code
 [LINK](./unet.py)


## SegNet
SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation

> ### a> Architecture
 ![architecture](http://www.programmersought.com/images/959/3b7812c216f7a695acde8c9a9725b25f.png)
> ### b> Paper
 [LINK](https://arxiv.org/pdf/1511.00561.pdf)
> ### c> Source code
 [LINK](./segnet.py)

 
## FPN
Feature Pyramid Networks for Object Detection

> ### a> Architecture
 ![architecture](https://miro.medium.com/max/1400/1*UtfPTLB53cR8EathGBOT2Q.jpeg)
> ### b> Paper
 [LINK](https://arxiv.org/pdf/1612.03144.pdf)
> ### c> Source code
 [LINK](./fpn.py)


## ResUNet
Road Extraction by Deep Residual U-Net

> ### a> Architecture
 ![architecture](https://storage.googleapis.com/groundai-web-prod/media/users/user_34/project_199097/images/x2.png)
> ### b> Paper
 [LINK](https://arxiv.org/pdf/1711.10684.pdf)
> ### c> Source code
 [LINK](./resunet.py)


## MCFCN
Automatic building segmentation of aerial imagery using multi-constraint fully convolutional networks

> ### a> Architecture
 ![architecture](https://www.mdpi.com/remotesensing/remotesensing-10-00407/article_deploy/html/images/remotesensing-10-00407-ag-550.jpg)
> ### b> Paper
 [LINK](https://www.mdpi.com/2072-4292/10/3/407/pdf)
> ### c> Source code
 [LINK](./mcfcn.py)
 
 
 
## BR-Net
A Boundary Regulated Network for Accurate Roof Segmentation and Outline Extraction

> ### a> Architecture
 ![architecture](https://www.mdpi.com/remotesensing/remotesensing-10-01195/article_deploy/html/images/remotesensing-10-01195-g003.png)
> ### b> Paper
 [LINK](https://www.mdpi.com/2072-4292/10/8/1195/pdf)
> ### c> Source code
 [LINK](./brnet.py)
 
 
## SFCN
A Stacked Fully Convolutional Networks with Feature Alignment Framework for Multi-Label Land-cover Segmentation

> ### a> Architecture
 ![architecture](https://www.mdpi.com/remotesensing/remotesensing-11-01051/article_deploy/html/images/remotesensing-11-01051-g003.png)
> ### b> Paper
 [LINK](https://www.mdpi.com/2072-4292/11/9/1051/pdf)
> ### c> Source code
 [LINK](../estrain.py)


# Ongoing models

## DeconvNet

> ### a> Architecture
 ![architecture](https://miro.medium.com/max/1400/1*LW8Anre45o9nfamxIVTY8Q.png)
> ### b> Paper
 [LINK](https://arxiv.org/pdf/1505.04597.pdf)
> ### c> Source code
 [LINK](https://github.com/fabianbormann/Tensorflow-DeconvNet-Segmentation)
 

## RefineNet

> ### a> Architecture
 ![architecture](https://i1.wp.com/blog.negativemind.com/wp-content/uploads/2019/03/RefineNet_architecture.jpg?w=1545&ssl=1)
> ### b> Paper
 [LINK](https://arxiv.org/pdf/1611.06612.pdf)
> ### c> Source code
 [MATLAB](https://github.com/guosheng/refinenet)
 [Tensorflow](https://github.com/eragonruan/refinenet-image-segmentation)


## Deeplab v1,v2,v3,v3+

> ### a> Architecture
 ![architecture](http://liangchiehchen.com/fig/deeplab.png)
> ### b> Paper
 [V3+](https://arxiv.org/pdf/1802.02611.pdf)
> ### c> Source code
 [Tensorflow](https://github.com/tensorflow/models/tree/master/research/deeplab)






