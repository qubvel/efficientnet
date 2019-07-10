# Copyright 2019 The TensorFlow Authors, Pavel Yakubovskiy. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for EfficientNet model.

[1] Mingxing Tan, Quoc V. Le
  EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
  ICML'19, https://arxiv.org/abs/1905.11946
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import keras.backend as K
import keras.models as KM
import keras.layers as KL
from keras.utils import get_file

from .layers import Swish, DropConnect
from .params import get_model_params, IMAGENET_WEIGHTS
from .initializers import conv_kernel_initializer, dense_kernel_initializer


__all__ = [
    "EfficientNet",
    "EfficientNetB0",
    "EfficientNetB1",
    "EfficientNetB2",
    "EfficientNetB3",
    "EfficientNetB4",
    "EfficientNetB5",
    "EfficientNetB6",
    "EfficientNetB7",
]


def round_filters(filters, global_params):
    """Round number of filters based on depth multiplier."""
    orig_f = filters
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    # print('round_filter input={} output={}'.format(orig_f, new_filters))
    return int(new_filters)


def round_repeats(repeats, global_params):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def SEBlock(block_args, global_params):
    num_reduced_filters = max(1, int(block_args.input_filters * block_args.se_ratio))
    filters = block_args.input_filters * block_args.expand_ratio
    if global_params.data_format == "channels_first":
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]

    def block(inputs):
        x = inputs
        x = KL.Lambda(lambda a: K.mean(a, axis=spatial_dims, keepdims=True))(x)
        x = KL.Conv2D(
            num_reduced_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding="same",
            use_bias=True,
        )(x)
        x = Swish()(x)
        # Excite
        x = KL.Conv2D(
            filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding="same",
            use_bias=True,
        )(x)
        x = KL.Activation("sigmoid")(x)
        out = KL.Multiply()([x, inputs])
        return out

    return block


def MBConvBlock(block_args, global_params, drop_connect_rate=None):
    batch_norm_momentum = global_params.batch_norm_momentum
    batch_norm_epsilon = global_params.batch_norm_epsilon

    if global_params.data_format == "channels_first":
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]

    has_se = (
        (block_args.se_ratio is not None)
        and (block_args.se_ratio > 0)
        and (block_args.se_ratio <= 1)
    )

    filters = block_args.input_filters * block_args.expand_ratio
    kernel_size = block_args.kernel_size

    def block(inputs):

        if block_args.expand_ratio != 1:
            x = KL.Conv2D(
                filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=conv_kernel_initializer,
                padding="same",
                use_bias=False,
            )(inputs)
            x = KL.BatchNormalization(
                axis=channel_axis,
                momentum=batch_norm_momentum,
                epsilon=batch_norm_epsilon,
            )(x)
            x = Swish()(x)
        else:
            x = inputs

        x = KL.DepthwiseConv2D(
            [kernel_size, kernel_size],
            strides=block_args.strides,
            depthwise_initializer=conv_kernel_initializer,
            padding="same",
            use_bias=False,
        )(x)
        x = KL.BatchNormalization(
            axis=channel_axis, momentum=batch_norm_momentum, epsilon=batch_norm_epsilon
        )(x)
        x = Swish()(x)

        if has_se:
            x = SEBlock(block_args, global_params)(x)

        # output phase

        x = KL.Conv2D(
            block_args.output_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding="same",
            use_bias=False,
        )(x)
        x = KL.BatchNormalization(
            axis=channel_axis, momentum=batch_norm_momentum, epsilon=batch_norm_epsilon
        )(x)

        if block_args.id_skip:
            if (
                all(s == 1 for s in block_args.strides)
                and block_args.input_filters == block_args.output_filters
            ):
                # only apply drop_connect if skip presents.
                if drop_connect_rate:
                    x = DropConnect(drop_connect_rate)(x)
                x = KL.Add()([x, inputs])
        return x

    return block


def EfficientNet(
    input_shape, block_args_list, global_params, input_tensor=None, include_top=True, pooling=None
):
    batch_norm_momentum = global_params.batch_norm_momentum
    batch_norm_epsilon = global_params.batch_norm_epsilon
    if global_params.data_format == "channels_first":
        channel_axis = 1
    else:
        channel_axis = -1

    # Stem part
    if input_tensor is None:
        inputs = KL.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            inputs = KL.Input(tensor=input_tensor, shape=input_shape)
        else:
            inputs = input_tensor
    x = inputs
    x = KL.Conv2D(
        filters=round_filters(32, global_params),
        kernel_size=[3, 3],
        strides=[2, 2],
        kernel_initializer=conv_kernel_initializer,
        padding="same",
        use_bias=False,
    )(x)
    x = KL.BatchNormalization(
        axis=channel_axis, momentum=batch_norm_momentum, epsilon=batch_norm_epsilon
    )(x)
    x = Swish()(x)

    # Blocks part
    block_idx = 1
    n_blocks = sum([block_args.num_repeat for block_args in block_args_list])
    drop_rate = global_params.drop_connect_rate or 0
    drop_rate_dx = drop_rate / n_blocks

    for block_args in block_args_list:
        assert block_args.num_repeat > 0
        # Update block input and output filters based on depth multiplier.
        block_args = block_args._replace(
            input_filters=round_filters(block_args.input_filters, global_params),
            output_filters=round_filters(block_args.output_filters, global_params),
            num_repeat=round_repeats(block_args.num_repeat, global_params),
        )

        # The first block needs to take care of stride and filter size increase.
        x = MBConvBlock(
            block_args, global_params, drop_connect_rate=drop_rate_dx * block_idx
        )(x)
        block_idx += 1

        if block_args.num_repeat > 1:
            block_args = block_args._replace(
                input_filters=block_args.output_filters, strides=[1, 1]
            )

        for _ in xrange(block_args.num_repeat - 1):
            x = MBConvBlock(
                block_args, global_params, drop_connect_rate=drop_rate_dx * block_idx
            )(x)
            block_idx += 1

    # Head part
    x = KL.Conv2D(
        filters=round_filters(1280, global_params),
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding="same",
        use_bias=False,
    )(x)
    x = KL.BatchNormalization(
        axis=channel_axis, momentum=batch_norm_momentum, epsilon=batch_norm_epsilon
    )(x)
    x = Swish()(x)

    if include_top:
        x = KL.GlobalAveragePooling2D(data_format=global_params.data_format)(x)
        if global_params.dropout_rate > 0:
            x = KL.Dropout(global_params.dropout_rate)(x)
        x = KL.Dense(
            global_params.num_classes, kernel_initializer=dense_kernel_initializer
        )(x)
        x = KL.Activation("softmax")(x)
    else:
        if pooling == "avg":
            x = KL.GlobalAveragePooling2D(data_format=global_params.data_format)(x)
        elif pooling == "max":
            x = KL.GlobalMaxPooling2D(data_format=global_params.data_format)(x)

    outputs = x
    model = KM.Model(inputs, outputs)

    return model


def _get_model_by_name(
    model_name, 
    input_shape=None, 
    input_tensor=None, 
    include_top=True, 
    weights=None, 
    classes=1000, 
    pooling=None
):
    """Re-Implementation of EfficientNet for Keras

    Reference:
        https://arxiv.org/abs/1807.11626

    Args:
        input_shape: optional, if ``None`` default_input_shape is used
            EfficientNetB0 - (224, 224, 3)
            EfficientNetB1 - (240, 240, 3)
            EfficientNetB2 - (260, 260, 3)
            EfficientNetB3 - (300, 300, 3)
            EfficientNetB4 - (380, 380, 3)
            EfficientNetB5 - (456, 456, 3)
            EfficientNetB6 - (528, 528, 3)
            EfficientNetB7 - (600, 600, 3)
        input_tensor: optional, if ``None`` default_input_tensor is used
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet).
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        pooling: optional [None, 'avg', 'max'], if ``include_top=False``
            add global pooling on top of the network
            - avg: GlobalAveragePooling2D
            - max: GlobalMaxPooling2D

    Returns:
        A Keras model instance.

    """
    if weights not in {None, "imagenet"}:
        raise ValueError('Parameter `weights` should be one of [None, "imagenet"]')

    if weights == "imagenet" and model_name not in IMAGENET_WEIGHTS:
        raise ValueError(
            "There are not pretrained weights for {} model.".format(model_name)
        )

    if weights == "imagenet" and include_top and classes != 1000:
        raise ValueError(
            "If using `weights` and `include_top`" " `classes` should be 1000"
        )

    block_agrs_list, global_params, default_input_shape = get_model_params(
        model_name, override_params={"num_classes": classes}
    )

    if input_shape is None:
        input_shape = (default_input_shape, default_input_shape, 3)
        
    model = EfficientNet(
        input_shape,
        block_agrs_list,
        global_params,
        input_tensor=input_tensor,
        include_top=include_top,
        pooling=pooling,
    )

    model.name = model_name

    if weights:
        if not include_top:
            weights_name = model_name + "-notop"
        else:
            weights_name = model_name
        weights = IMAGENET_WEIGHTS[weights_name]
        weights_path = get_file(
            weights["name"],
            weights["url"],
            cache_subdir="models",
            md5_hash=weights["md5"],
        )
        model.load_weights(weights_path)

    return model


def EfficientNetB0(
    include_top=True, input_shape=None, input_tensor=None, weights=None, classes=1000, pooling=None
):
    return _get_model_by_name(
        "efficientnet-b0",
        include_top=include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        classes=classes,
        pooling=pooling,
    )


def EfficientNetB1(
    include_top=True, input_shape=None, input_tensor=None, weights=None, classes=1000, pooling=None
):
    return _get_model_by_name(
        "efficientnet-b1",
        include_top=include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        classes=classes,
        pooling=pooling,
    )


def EfficientNetB2(
    include_top=True, input_shape=None, input_tensor=None, weights=None, classes=1000, pooling=None
):
    return _get_model_by_name(
        "efficientnet-b2",
        include_top=include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        classes=classes,
        pooling=pooling,
    )


def EfficientNetB3(
    include_top=True, input_shape=None, input_tensor=None, weights=None, classes=1000, pooling=None
):
    return _get_model_by_name(
        "efficientnet-b3",
        include_top=include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        classes=classes,
        pooling=pooling,
    )


def EfficientNetB4(
    include_top=True, input_shape=None, input_tensor=None, weights=None, classes=1000, pooling=None
):
    return _get_model_by_name(
        "efficientnet-b4",
        include_top=include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        classes=classes,
        pooling=pooling,
    )


def EfficientNetB5(
    include_top=True, input_shape=None, input_tensor=None, weights=None, classes=1000, pooling=None
):
    return _get_model_by_name(
        "efficientnet-b5",
        include_top=include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        classes=classes,
        pooling=pooling,
    )


def EfficientNetB6(
    include_top=True, input_shape=None, input_tensor=None, weights=None, classes=1000, pooling=None
):
    return _get_model_by_name(
        "efficientnet-b6",
        include_top=include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        classes=classes,
        pooling=pooling,
    )


def EfficientNetB7(
    include_top=True, input_shape=None, input_tensor=None, weights=None, classes=1000, pooling=None
):
    return _get_model_by_name(
        "efficientnet-b7",
        include_top=include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        classes=classes,
        pooling=pooling,
    )


EfficientNetB0.__doc__ = _get_model_by_name.__doc__
EfficientNetB1.__doc__ = _get_model_by_name.__doc__
EfficientNetB2.__doc__ = _get_model_by_name.__doc__
EfficientNetB3.__doc__ = _get_model_by_name.__doc__
EfficientNetB4.__doc__ = _get_model_by_name.__doc__
EfficientNetB5.__doc__ = _get_model_by_name.__doc__
EfficientNetB6.__doc__ = _get_model_by_name.__doc__
EfficientNetB7.__doc__ = _get_model_by_name.__doc__
