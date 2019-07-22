import sys
import pytest

sys.path.insert(0, '/project')

import numpy as np
import efficientnet as efn

from skimage.io import imread
from keras.models import load_model


PANDA_PATH = 'misc/panda.jpg'
MODELS = [
    efn.EfficientNetB0,
    efn.EfficientNetB1,
    efn.EfficientNetB2,
    efn.EfficientNetB3,
    efn.EfficientNetB4,
    efn.EfficientNetB5,
]

def _get_dummy_input(input_shape):
    input_shape = [x if x else 1 for x in input_shape]
    return np.ones(input_shape)


def _get_panda_input(input_shape):
    image = imread(PANDA_PATH)
    image = efn.center_crop_and_resize(image, input_shape[1])
    image = efn.preprocess_input(image)
    return image


def test_model_predict():
    model = efn.EfficientNetB0()
    input_ = _get_dummy_input(model.input_shape)
    result = model.predict(input_)


def test_model_save_load():
    model = efn.EfficientNetB0()
    model.save('/tmp/model.h5')
    new_model = load_model('/tmp/model.h5')

@pytest.mark.parametrize('args', MODELS)
def test_models_result(args):
    model_builder, result = args
    model = model_builder(weights='imagenet')
    input_ = _get_panda_input(model.input_shape)
    prediction = model.predict(input_)
    assert result[0] == prediction.argmax()
    assert np.allclose(result[1], prediction.max())
