# EfficientNet Keras (and TensorFlow Keras)

[![PyPI version](https://badge.fury.io/py/efficientnet.svg)](https://badge.fury.io/py/efficientnet) [![Downloads](https://pepy.tech/badge/efficientnet/month)](https://pepy.tech/project/efficientnet/month)

This repository contains a Keras (and TensorFlow Keras) reimplementation of **EfficientNet**, a lightweight convolutional neural network architecture achieving the [state-of-the-art accuracy with an order of magnitude fewer parameters and FLOPS](https://arxiv.org/abs/1905.11946), on both ImageNet and
five other commonly used transfer learning datasets.

The codebase is heavily inspired by the [TensorFlow implementation](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet).

## Important!
There was a huge library update on **24th of July 2019**. Now efficientnet works with both frameworks: `keras` and `tensorflow.keras`.
If you have models trained before that date, please use efficientnet of version 0.0.4 to load them. You can roll back using `pip install -U efficientnet==0.0.4` or `pip install -U git+https://github.com/qubvel/efficientnet/tree/v0.0.4`.

## Table of Contents

 1. [About EfficientNet Models](#about-efficientnet-models)
 2. [Examples](#examples)
 3. [Models](#models)
 4. [Installation](#installation)
 5. [Frequently Asked Questions](#frequently-asked-questions)
 6. [Acknowledgements](#acknowledgements)

## About EfficientNet Models

EfficientNets rely on AutoML and compound scaling to achieve superior performance without compromising resource efficiency. The [AutoML Mobile framework](https://ai.googleblog.com/2018/08/mnasnet-towards-automating-design-of.html) has helped develop a mobile-size baseline network, **EfficientNet-B0**, which is then improved by the compound scaling method  to obtain EfficientNet-B1 to B7.

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

* In high-accuracy regime, EfficientNet-B7 achieves the state-of-the-art 84.4% top-1 / 97.1% top-5 accuracy on ImageNet with 66M parameters and 37B FLOPS. At the same time, the model is 8.4x smaller and 6.1x faster on CPU inference than the former leader, [Gpipe](https://arxiv.org/abs/1811.06965).

* In middle-accuracy regime, EfficientNet-B1 is 7.6x smaller and 5.7x faster on CPU inference than [ResNet-152](https://arxiv.org/abs/1512.03385), with similar ImageNet accuracy.

* Compared to the widely used [ResNet-50](https://arxiv.org/abs/1512.03385), EfficientNet-B4 improves the top-1 accuracy from 76.3% of ResNet-50 to 82.6% (+6.3%), under similar FLOPS constraints.

## Examples

* *Initializing the model*:

```python
# models can be build with Keras or Tensorflow frameworks
# use keras and tfkeras modules respectively
# efficientnet.keras / efficientnet.tfkeras
import efficientnet.keras as efn 

model = efn.EfficientNetB0(weights='imagenet')  # or weights='noisy-student'

```

* *Loading the pre-trained weights*:

```python
# model use some custom objects, so before loading saved model
# import module your network was build with
# e.g. import efficientnet.keras / import efficientnet.tfkeras
import efficientnet.tfkeras
from tensorflow.keras.models import load_model

model = load_model('path/to/model.h5')
```

See the complete example of loading the model and making an inference in the Jupyter notebook [here](https://github.com/qubvel/efficientnet/blob/master/examples/inference_example.ipynb).

## Models

The performance of each model variant using the pre-trained weights converted from checkpoints provided by the authors is as follows:

| Architecture   | @top1* Imagenet| @top1* Noisy-Student| 
| -------------- | :----: |:---:|
| EfficientNetB0 | 0.772  |0.788|
| EfficientNetB1 | 0.791  |0.815|
| EfficientNetB2 | 0.802  |0.824|
| EfficientNetB3 | 0.816  |0.841|
| EfficientNetB4 | 0.830  |0.853|
| EfficientNetB5 | 0.837  |0.861|
| EfficientNetB6 | 0.841  |0.864|
| EfficientNetB7 | 0.844  |0.869|

**\*** - topK accuracy score for converted models (imagenet `val` set)

## Installation

### Requirements

* `Keras >= 2.2.0` / `TensorFlow >= 1.12.0`
* `keras_applications >= 1.0.7`
* `scikit-image`

### Installing from the source

```bash
$ pip install -U git+https://github.com/qubvel/efficientnet
```

### Installing from PyPI

PyPI stable release

```bash
$ pip install -U efficientnet
```

PyPI latest release (with keras and tf.keras support)

```bash
$ pip install -U --pre efficientnet
```

## Frequently Asked Questions

* **How can I convert the original TensorFlow checkpoints to Keras HDF5?**

Pick the target directory (like `dist`) and run the [converter script](./scripts) from the repo directory as follows:

```bash
$ ./scripts/convert_efficientnet.sh --target_dir dist
```

You can also optionally create the virtual environment with all the dependencies installed by adding `--make_venv=true` and operate in a self-destructing temporary location instead of the target directory by setting `--tmp_working_dir=true`.

## Acknowledgements
I would like to thanks community members who actively contribute to this repository:

1) Sasha Illarionov ([@sdll](https://github.com/sdll)) for preparing automated script for weights conversion
2) Björn Barz ([@Callidior](https://github.com/Callidior)) for model code adaptation for keras and tensorflow.keras frameworks 
