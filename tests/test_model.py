import os
import sys
import pytest
import numpy as np

from skimage.io import imread

sys.path.insert(0, '.')
if os.environ.get('TF_KERAS'):
    import efficientnet.tfkeras as efn
    from tensorflow.keras.models import load_model
else:
    import efficientnet.keras as efn
    from keras.models import load_model

PANDA_PATH = 'misc/panda.jpg'

PANDA_ARGS = [
    (efn.EfficientNetB0, (388, 0.7587869)),
    (efn.EfficientNetB1, (388, 0.8373562)),
    (efn.EfficientNetB2, (388, 0.8569102)),
    (efn.EfficientNetB3, (388, 0.8761664)),
    (efn.EfficientNetB4, (388, 0.7342420)),
    (efn.EfficientNetB5, (388, 0.8810669)),
    (efn.EfficientNetB6, (388, 0.8667784)),
    (efn.EfficientNetB7, (388, 0.8399882)),
]


def _select_args(args):
    is_travis = os.environ.get('TRAVIS', False)
    if is_travis:
        return args[:1]
    else:
        return args


def _get_dummy_input(input_shape):
    input_shape = [x if x else 1 for x in input_shape]
    return np.ones(input_shape)


def _get_panda_input(input_shape):
    image = imread(PANDA_PATH)
    image = efn.center_crop_and_resize(image, input_shape[1])
    image = efn.preprocess_input(image)
    image = np.expand_dims(image, 0)
    return image


def test_model_predict():
    model = efn.EfficientNetB0()
    input_ = _get_dummy_input(model.input_shape)
    result = model.predict(input_)


def test_model_save_load():
    model = efn.EfficientNetB0()
    model.save('/tmp/model.h5')
    new_model = load_model('/tmp/model.h5')


@pytest.mark.parametrize('args', _select_args(PANDA_ARGS))
def test_models_result(args):
    model_builder, result = args
    model = model_builder(weights='imagenet')
    input_ = _get_panda_input(model.input_shape)
    prediction = model.predict(input_)
    assert result[0] == prediction.argmax()
    assert np.allclose(result[1], prediction.max())
    
    
if __name__ == "__main__":
    pytest.main([__file__])