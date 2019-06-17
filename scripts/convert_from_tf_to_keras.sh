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

# Make sure that we run Python 3.6 or newer
PYTHON_VERSION=$(python -V 2>&1 | sed 's/.* \([0-9]\).\([0-9]\).*/\1\2/')
if [ "$PYTHON_VERSION" -lt "36" ]; then
    echo "This script requires python 3.6 or newer."
    exit 1
fi

if ! [ -x "$(command -v aria2c)" ]; then
    echo "Please install aria2c before continuing."
fi

if [ -z "$1" ]; then
    echo "No target directory supplied, aborting."
    exit 1
fi

SOURCE_CODE_URL="https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/"
SOURCE_CODE_FILES=(
    "efficientnet_builder.py"
    "efficientnet_model.py"
    "eval_ckpt_main.py"
    "utils.py"
    "preprocessing.py"
)
SOURCE_CODE_DIR="tf_src"

CHECKPOINTS_DIR="pretrained_tensorflow"
CHECKPOINT_PREFIX="efficientnet-"
CHECKPOINTS_URL="https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/"
CHECKPOINTS_EXT=".tar.gz"

CONVERTED_MODELS_DIR="pretrained_keras"

MODELS=(
    "b0"
    "b1"
    "b2"
    "b3"
    "b4"
    "b5"
)

WORKING_DIR="dist"
# WORKING_DIR=$(mktemp -d)
# trap 'rm -rf -- "$WORKING_DIR"' INT TERM HUP EXIT

OUTPUT_DIR="dist"

PARENT_DIR="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

printf -- '=%.0s' {1..80}
echo ""
echo "Setting up the installation environment..."
printf -- '=%.0s' {1..80}
echo ""

set -e

mkdir -p $OUTPUT_DIR

mkdir -p $WORKING_DIR
cd $WORKING_DIR
# virtualenv --no-site-packages venv && \
# source venv/bin/activate && \
# pip install tensorflowjs numpy tensorflow keras scikit-image

if ! [ -d $CHECKPOINTS_DIR ]; then
    printf -- '=%.0s' {1..80}
    echo ""
    echo "Downloading the checkpoints..."
    printf -- '=%.0s' {1..80}
    echo ""
    mkdir -p $CHECKPOINTS_DIR
    for MODEL_VERSION in "${MODELS[@]}"; do
        if ! [ -d $CHECKPOINT_PREFIX$MODEL_VERSION ]; then
            cd $CHECKPOINTS_DIR
            aria2c -x 16 -k 1M -o $MODEL_VERSION$CHECKPOINTS_EXT $CHECKPOINTS_URL$CHECKPOINT_PREFIX$MODEL_VERSION$CHECKPOINTS_EXT
            tar xvf $MODEL_VERSION$CHECKPOINTS_EXT
            rm $MODEL_VERSION$CHECKPOINTS_EXT
            cd ..
        fi
    done
fi

printf -- '=%.0s' {1..80}
echo ""
echo "Converting the checkpoints to Keras..."
printf -- '=%.0s' {1..80}
echo ""

if ! [ -d $SOURCE_CODE_DIR ]; then
    printf -- '-%.0s' {1..80}
    echo ""
    echo "Downloading the source code..."
    printf -- '-%.0s' {1..80}
    echo ""
    mkdir -p $SOURCE_CODE_DIR
    touch $SOURCE_CODE_DIR/__init__.py
    for SOURCE_CODE_FILE in "${SOURCE_CODE_FILES[@]}"; do
        aria2c -x 16 -k 1M -o $SOURCE_CODE_DIR/$SOURCE_CODE_FILE $SOURCE_CODE_URL$SOURCE_CODE_FILE
    done
fi

cd $PARENT_DIR
mkdir -p $OUTPUT_DIR/$CONVERTED_MODELS_DIR
cd $OUTPUT_DIR

for MODEL_VERSION in "${MODELS[@]}"; do

    MODEL_NAME="efficientnet-"$MODEL_VERSION

    printf -- '-%.0s' {1..80}
    echo ""
    echo "Converting $MODEL_NAME..."
    printf -- '-%.0s' {1..80}
    echo ""

    WEIGHTS_ONLY="true"

    if ! [ -z "$2" ] && [ "$2" != "true" ]; then
        WEIGHTS_ONLY="false"
    fi

    PYTHONPATH=.. python $SCRIPT_DIR/load_efficientnet.py \
        --model_name $MODEL_NAME \
        --source $SOURCE_CODE_DIR \
        --tf_checkpoint $CHECKPOINTS_DIR/$MODEL_NAME \
        --output_file $CONVERTED_MODELS_DIR/$MODEL_NAME".h5" \
        --weights_only $WEIGHTS_ONLY
done

printf -- '=%.0s' {1..80}
echo ""
echo "Success!"
printf -- '=%.0s' {1..80}
echo ""
