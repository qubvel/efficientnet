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
import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.initializers import Initializer
from keras.utils.generic_utils import get_custom_objects


class EfficientConv2DKernelInitializer(Initializer):
    """Initialization for convolutional kernels.

    The main difference with tf.variance_scaling_initializer is that
    tf.variance_scaling_initializer uses a truncated normal with an uncorrected
    standard deviation, whereas here we use a normal distribution. Similarly,
    tf.contrib.layers.variance_scaling_initializer uses a truncated normal with
    a corrected standard deviation.

    Args:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused

    Returns:
      an initialization for the variable
    """

    def __call__(self, shape, dtype=K.floatx(), **kwargs):
        kernel_height, kernel_width, _, out_filters = shape
        fan_out = int(kernel_height * kernel_width * out_filters)
        return tf.random_normal(
            shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype
        )


class EfficientDenseKernelInitializer(Initializer):
    """Initialization for dense kernels.

    This initialization is equal to
      tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                      distribution='uniform').
    It is written out explicitly here for clarity.

    Args:
      shape: shape of variable
      dtype: dtype of variable

    Returns:
      an initialization for the variable
    """

    def __call__(self, shape, dtype=K.floatx(), **kwargs):
        """Initialization for dense kernels.

        This initialization is equal to
          tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                          distribution='uniform').
        It is written out explicitly here for clarity.

        Args:
          shape: shape of variable
          dtype: dtype of variable

        Returns:
          an initialization for the variable
        """
        init_range = 1.0 / np.sqrt(shape[1])
        return tf.random_uniform(shape, -init_range, init_range, dtype=dtype)


conv_kernel_initializer = EfficientConv2DKernelInitializer()
dense_kernel_initializer = EfficientDenseKernelInitializer()


get_custom_objects().update(
    {
        "EfficientDenseKernelInitializer": EfficientDenseKernelInitializer,
        "EfficientConv2DKernelInitializer": EfficientConv2DKernelInitializer,
    }
)
