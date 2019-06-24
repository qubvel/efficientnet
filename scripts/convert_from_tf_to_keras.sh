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

notify() {
    # $1: the string which encloses the message
    # $2: the message text
    MESSAGE="$1 $2 $1"
    PADDING=$(eval $(echo printf '"$1%0.s"' {1..${#MESSAGE}}))
    SPACING="$1 $(eval $(echo printf '.%0.s' {1..${#2}})) $1"
    echo $PADDING
    echo $SPACING
    echo $MESSAGE
    echo $SPACING
    echo $PADDING
}

elementIn() {
    # Checks whether $1 is in the array $2
    # Example:
    # The snippet
    #   array=("1" "a string" "3")
    #   containsElement "a string" "${array[@]}"
    # Returns 1
    # Source:
    # https://stackoverflow.com/questions/3685970/check-if-a-bash-array-contains-a-value
    local e match="$1"
    shift
    for e; do [[ "$e" == "$match" ]] && return 0; done
    return 1
}

# The mechanics of reading the long options are taken from
# https://mywiki.wooledge.org/ComplexOptionParsing#CA-78a4030cda5dc5c45377a4d98ebd4fe610e0aa7e_2
usage() {
    echo "Usage:"
    echo "  $0 [ --help | -h ]"
    echo "  $0 [ --target_dir=<value> | --target_dir <value> ] [options]"
    echo
    echo "Options:"
    echo "  --use_venv=true|yes|1|t|y:: Use the virtual env with pre-installed dependencies"
    echo "  --tmp_working_dir=true|yes|1|t|y :: Make a temporary working directory the working space"
    echo
    echo "The default target_dir is dist."
}

# set defaults
LAST_ARG_IDX=$(($# + 1))
TARGET_DIR="dist"
USE_VENV="false"
declare -A LONGOPTS
# Use associative array to declare how many arguments a long option
# expects. In this case we declare that loglevel expects/has one
# argument and range has two. Long options that aren't listed in this
# way will have zero arguments by default.
LONGOPTS=([target_dir]=1 [use_venv]=1)
OPTSPEC="h-:"
while getopts "$OPTSPEC" opt; do
    while true; do
        case "${opt}" in
        -) #OPTARG is name-of-long-option or name-of-long-option=value
            if [[ ${OPTARG} =~ .*=.* ]]; then # with this --key=value format only one argument is possible
                opt=${OPTARG/=*/}
                ((${#opt} <= 1)) && {
                    echo "Syntax error: Invalid long option '$opt'" >&2
                    exit 2
                }
                if (($((LONGOPTS[$opt])) != 1)); then
                    echo "Syntax error: Option '$opt' does not support this syntax." >&2
                    exit 2
                fi
                OPTARG=${OPTARG#*=}
            else #with this --key value1 value2 format multiple arguments are possible
                opt="$OPTARG"
                ((${#opt} <= 1)) && {
                    echo "Syntax error: Invalid long option '$opt'" >&2
                    exit 2
                }
                OPTARG=(${@:OPTIND:$((LONGOPTS[$opt]))})
                ((OPTIND += LONGOPTS[$opt]))
                echo $OPTIND
                ((OPTIND > $LAST_ARG_IDX)) && {
                    echo "Syntax error: Not all required arguments for option '$opt' are given." >&2
                    exit 3
                }
            fi
            continue
            ;;
        target_dir)
            TARGET_DIR=$(realpath $OPTARG)
            ;;
        use_venv)
            USE_VENV=$OPTARG
            ;;
        tmp_working_dir)
            MAKE_TMP_WORKING_DIR=$OPTARG
            ;;
        h | help)
            usage
            exit 0
            ;;
        ?)
            echo "Syntax error: Unknown short option '$OPTARG'" >&2
            exit 2
            ;;
        *)
            echo "Syntax error: Unknown long option '$opt'" >&2
            exit 2
            ;;
        esac
        break
    done
done

# internal variables
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

TRUE=(
    "true"
    "t"
    "1"
    "yes"
    "y"
)

if elementIn "$MAKE_TMP_WORKING_DIR" "${TRUE[@]}"; then
    WORKING_DIR=$(mktemp -d)
    trap 'rm -rf -- "$WORKING_DIR"' INT TERM HUP EXIT
else
    WORKING_DIR=$TARGET_DIR
fi

PARENT_DIR="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

notify "=" "Setting up the installation environment..."

set -e

mkdir -p $TARGET_DIR/$CONVERTED_MODELS_DIR

mkdir -p $WORKING_DIR
cd $WORKING_DIR

if elementIn "$USE_VENV" "${TRUE[@]}"; then
    if [ -d $VIRTUALENV_DIR ]; then
        source $VIRTUALENV_DIR/bin/activate
    else
        virtualenv --no-site-packages $VIRTUALENV_DIR
        source $VIRTUALENV_DIR/bin/activate
        pip install tensorflowjs keras scikit-image
    fi
fi

if ! [ -d $CHECKPOINTS_DIR ]; then
    notify "=" "Downloading the checkpoints..."
    mkdir -p $CHECKPOINTS_DIR
    for MODEL_VERSION in "${MODELS[@]}"; do
        if ! [ -d $CHECKPOINT_PREFIX$MODEL_VERSION ]; then
            cd $CHECKPOINTS_DIR
            # -x: maximum number of connections per server
            # -k: minimum split size for multi-source download
            # -o: target filename
            aria2c \
                -x 16 \
                -k 1M \
                -o $MODEL_VERSION$CHECKPOINTS_EXT \
                $CHECKPOINTS_URL$CHECKPOINT_PREFIX$MODEL_VERSION$CHECKPOINTS_EXT
            tar xvf $MODEL_VERSION$CHECKPOINTS_EXT
            rm $MODEL_VERSION$CHECKPOINTS_EXT
            cd ..
        fi
    done
fi

notify "=" "Converting the checkpoints to Keras..."

if ! [ -d $SOURCE_CODE_DIR ]; then
    notify "~" "Downloading the source code..."
    mkdir -p $SOURCE_CODE_DIR
    touch $SOURCE_CODE_DIR/__init__.py
    for SOURCE_CODE_FILE in "${SOURCE_CODE_FILES[@]}"; do
        aria2c -x 16 -k 1M -o $SOURCE_CODE_DIR/$SOURCE_CODE_FILE $SOURCE_CODE_URL$SOURCE_CODE_FILE
    done
fi

cd $PARENT_DIR
cd $TARGET_DIR
for MODEL_VERSION in "${MODELS[@]}"; do

    MODEL_NAME="efficientnet-"$MODEL_VERSION

    notify "~" "Converting $MODEL_NAME..."

    WEIGHTS_ONLY="true"

    if ! [ -z "$2" ] && [ "$2" != "true" ]; then
        WEIGHTS_ONLY="false"
    fi

    PYTHONPATH=.. python $SCRIPT_DIR/load_efficientnet.py \
        --model_name $MODEL_NAME \
        --source $SOURCE_CODE_DIR \
        --tf_checkpoint $CHECKPOINTS_DIR/$MODEL_NAME \
        --output_file $CONVERTED_MODELS_DIR/$MODEL_NAME \
        --weights_only $WEIGHTS_ONLY
done

notify "=" "Success!"
