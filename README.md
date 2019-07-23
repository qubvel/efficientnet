# EfficientNet-Keras

This repository contains a Keras reimplementation of **EfficientNet**, a lightweight convolutional neural network architecture achieving the [state-of-the-art accuracy with an order of magnitude fewer parameters and FLOPS](https://arxiv.org/abs/1905.11946), on both ImageNet and
five other commonly used transfer learning datasets.

The codebase is heavily inspired by the [TensorFlow implementation](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet).

## Table of Contents

 1. [About EfficientNet Models](#about-efficientnet-models)
 2. [Examples](#examples)
 3. [Models](#models)
 4. [Installation](#installation)
 5. [Frequently Asked Questions](#frequently-asked-questions)

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

model = efn.EfficientNetB0(weights='imagenet')

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

| Architecture   | @top1* | @top5* | Weights |
| -------------- | :----: | :----: | :-----: |
| EfficientNetB0 | 0.7668 | 0.9312 |    +    |
| EfficientNetB1 | 0.7863 | 0.9418 |    +    |
| EfficientNetB2 | 0.7968 | 0.9475 |    +    |
| EfficientNetB3 | 0.8083 | 0.9531 |    +    |
| EfficientNetB4 | 0.8259 | 0.9612 |    +    |
| EfficientNetB5 | 0.8309 | 0.9646 |    +    |
| EfficientNetB6 |   -    |   -    |    -    |
| EfficientNetB7 |   -    |   -    |    -    |

**\*** - topK accuracy score for converted models (imagenet `val` set)

## Installation

### Requirements

* `Keras >= 2.2.0` / `TensorFlow >= 1.12.0`
* `keras_applications >= 1.0.7`
* `scikit-image`

### Installing from the source

```bash
pip install -U git+https://github.com/qubvel/efficientnet
```

### Installing from PyPI

```bash
pip install -U efficientnet
```

## Frequently Asked Questions

* **How can I convert the original TensorFlow checkpoints to Keras HDF5?**

Pick the target directory (like `dist`) and run the [converter script](./scripts) from the repo directory as follows:

```bash
./scripts/convert_efficientnet.sh --target_dir dist
```

You can also optionally create the virtual environment with all the dependencies installed by adding `--make_venv=true` and operate in a self-destructing temporary location instead of the target directory by setting `--tmp_working_dir=true`.

* **Why are B6 and B7 model variants not yet supported?**

Weights for B6-B7 have not been made available yet, but might appear soon. Follow the [issue](https://github.com/tensorflow/tpu/issues/377) for updates.

## Acknowledgements
I would like to thanks community members who actively contribute to this repository:

1) Sasha Illarionov ([@sdll](https://github.com/sdll)) for preparing automated script for weights conversion
2) Björn Barz ([@Callidior](https://github.com/Callidior)) for model code adaptation for keras and tensorflow.keras frameworks 
