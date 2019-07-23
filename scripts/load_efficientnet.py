#!/usr/bin/env bash
# =============================================================================
# Copyright 2019 Pavel Yakubovskiy, Sasha Illarionov. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import argparse
import sys

import numpy as np

import tensorflow as tf
import efficientnet.keras
from keras.layers import BatchNormalization, Conv2D, Dense


def _get_model_by_name(name, *args, **kwargs):
    models = {
        'efficientnet-b0': efficientnet.keras.EfficientNetB0,
        'efficientnet-b1': efficientnet.keras.EfficientNetB1,
        'efficientnet-b2': efficientnet.keras.EfficientNetB2,
        'efficientnet-b3': efficientnet.keras.EfficientNetB3,
        'efficientnet-b4': efficientnet.keras.EfficientNetB4,
        'efficientnet-b5': efficientnet.keras.EfficientNetB5,
    }

    model_fn = models[name]
    model = model_fn(*args, **kwargs)
    return model


def group_weights(weights):
    """
    Group each layer weights together, initially all weights are dict of 'layer_name/layer_var': np.array

    Example:
        input:  {
                    ...: ...
                    'conv2d/kernel': <np.array>,
                    'conv2d/bias': <np.array>,
                    ...: ...
                }
        output: [..., [...], [<conv2d/kernel-weights>, <conv2d/bias-weights>], [...], ...]

    """

    out_weights = []

    previous_layer_name = ""
    group = []

    for k, v in weights.items():

        layer_name = "/".join(k.split("/")[:-1])

        if layer_name == previous_layer_name:
            group.append(v)
        else:
            if group:
                out_weights.append(group)

            group = [v]
            previous_layer_name = layer_name

    out_weights.append(group)
    return out_weights


def load_weights(model, weights):
    """Load weights to Conv2D, BatchNorm, Dense layers of model sequentially"""
    layer_index = 0
    groupped_weights = group_weights(weights)
    for layer in model.layers:
        if isinstance(layer, (Conv2D, BatchNormalization, Dense)):
            print(layer)
            layer.set_weights(groupped_weights[layer_index])
            layer_index += 1


def convert_tensorflow_model(
        model_name, model_ckpt, output_file, example_img="misc/panda.jpg", weights_only=True
):
    """ Loads and saves a TensorFlow model. """
    image_files = [example_img]
    eval_ckpt_driver = eval_ckpt_main.EvalCkptDriver(model_name)
    with tf.Graph().as_default(), tf.Session() as sess:
        images, _ = eval_ckpt_driver.build_dataset(
            image_files, [0] * len(image_files), False
        )
        eval_ckpt_driver.build_model(images, is_training=False)
        sess.run(tf.global_variables_initializer())
        eval_ckpt_driver.restore_model(sess, model_ckpt)
        global_variables = tf.global_variables()
        weights = dict()
        for variable in global_variables:
            try:
                weights[variable.name] = variable.eval()
            except:
                print(f"Skipping variable {variable.name}, an exception occurred")
    model = _get_model_by_name(
        model_name, include_top=True, input_shape=None, weights=None, classes=1000
    )
    load_weights(model, weights)
    output_file = f"{output_file}.h5"
    if weights_only:
        model.save_weights(output_file)
    else:
        model.save(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert TF model to Keras and save for easier future loading"
    )
    parser.add_argument(
        "--source", type=str, default="dist/tf_src", help="source code path"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="efficientnet-b0",
        help="efficientnet-b{N}, where N is an integer 0 <= N <= 7",
    )
    parser.add_argument(
        "--tf_checkpoint",
        type=str,
        default="pretrained_tensorflow/efficientnet-b0/",
        help="checkpoint file path",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="pretrained_keras/efficientnet-b0",
        help="output Keras model file name",
    )
    parser.add_argument(
        "--weights_only",
        type=str,
        default="true",
        help="Whether to include metadata in the serialized Keras model",
    )
    args = parser.parse_args()

    sys.path.append(args.source)
    import eval_ckpt_main

    true_values = ("yes", "true", "t", "1", "y")
    convert_tensorflow_model(
        model_name=args.model_name,
        model_ckpt=args.tf_checkpoint,
        output_file=args.output_file,
        weights_only=args.weights_only in true_values,
    )
