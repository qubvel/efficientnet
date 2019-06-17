# EfficientNet-Keras

This repository contains Keras reimplementation of EfficientNet, the new convolutional neural network architecture from [EfficientNet](https://arxiv.org/abs/1905.11946) ([TensorFlow implementation](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)). 

### Table of content
 1. [About EfficientNets](#about)
 2. [Examples](#examples)
 3. [Models](#models) 
 4. [Installation](#installation)


### About EfficientNet Models <a name="about"></a>

If you're new to EfficientNets, here is an explanation straight from the official TensorFlow implementation: 

EfficientNets are a family of image classification models, which achieve state-of-the-art accuracy, yet being an order-of-magnitude smaller and faster than previous models. EfficientNets are based on AutoML and Compound Scaling. In particular, [AutoML Mobile framework](https://ai.googleblog.com/2018/08/mnasnet-towards-automating-design-of.html) have been used to develop a mobile-size baseline network, named as EfficientNet-B0; Then, the compound scaling method is used to scale up this baseline to obtain EfficientNet-B1 to B7.

<table border="0">
<tr>
    <td>
    <img src="https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/g3doc/params.png" width="100%" />
    </td>
    <td>
    <img src="https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/g3doc/flops.png", width="90%" />
    </td>
</tr>
</table>

EfficientNets achieve state-of-the-art accuracy on ImageNet with an order of magnitude better efficiency:


* In high-accuracy regime, EfficientNet-B7 achieves state-of-the-art 84.4% top-1 / 97.1% top-5 accuracy on ImageNet with 66M parameters and 37B FLOPS, being 8.4x smaller and 6.1x faster on CPU inference than previous best [Gpipe](https://arxiv.org/abs/1811.06965).

* In middle-accuracy regime, EfficientNet-B1 is 7.6x smaller and 5.7x faster on CPU inference than [ResNet-152](https://arxiv.org/abs/1512.03385), with similar ImageNet accuracy.

* Compared with the widely used [ResNet-50](https://arxiv.org/abs/1512.03385), EfficientNet-B4 improves the top-1 accuracy from 76.3% of ResNet-50 to 82.6% (+6.3%), under similar FLOPS constraint.

### Examples <a name="examples"></a>

 - Two lines to create model:

```python
from efficientnet import EfficientNetB0

model = EfficientNetB0(weights='imagenet')

```

 - Inference example:  
 [inference_example.ipynb](https://github.com/qubvel/efficientnet/blob/master/examples/inference_exmaple.ipynb)

 - Loading saved model:
 
```python
from efficientnet import load_model

model = load_model('path/to/model.h5')
```

### Models <a name="models"></a>

Available architectures and pretrained weights (converted from original repo):

| Architecture   | @top1*| @top5*| Weights |
|----------------|:-----:|:-----:|:-------:|
| EfficientNetB0 |0.7668 |0.9312 |    +    |
| EfficientNetB1 |0.7863 |0.9418 |    +    |
| EfficientNetB2 |0.7968 |0.9475 |    +    |
| EfficientNetB3 |0.8083 |0.9531 |    +    |
| EfficientNetB4 |0.8259 |0.9612 |    +    |
| EfficientNetB5 |0.8309 |0.9646 |    +    |
| EfficientNetB6 |   -   |  -    |    -    |
| EfficientNetB7 |   -   |  -    |    -    |

"*" - topK accuracy score for converted models (imagenet `val` set) 
 
Weights for B6-B7 are not released yet ([issue](https://github.com/tensorflow/tpu/issues/377)).

### Installation <a name="installation"></a>

Requirements:
 - keras >= 2.2.0 (tensorflow)
 - scikit-image

Source:

```bash
$ pip install -U git+https://github.com/qubvel/efficientnet
```

PyPI:

```bash
$ pip install -U efficientnet
```
