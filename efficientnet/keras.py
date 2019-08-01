from . import inject_keras_modules, init_keras_custom_objects
from . import model

from .preprocessing import center_crop_and_resize

EfficientNetB0 = inject_keras_modules(model.EfficientNetB0)
EfficientNetB1 = inject_keras_modules(model.EfficientNetB1)
EfficientNetB2 = inject_keras_modules(model.EfficientNetB2)
EfficientNetB3 = inject_keras_modules(model.EfficientNetB3)
EfficientNetB4 = inject_keras_modules(model.EfficientNetB4)
EfficientNetB5 = inject_keras_modules(model.EfficientNetB5)
EfficientNetB6 = inject_keras_modules(model.EfficientNetB6)
EfficientNetB7 = inject_keras_modules(model.EfficientNetB7)

preprocess_input = inject_keras_modules(model.preprocess_input)

init_keras_custom_objects()
