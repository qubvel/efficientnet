import numpy as np
from skimage.transform import resize

MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]

MAP_INTERPOLATION_TO_ORDER = {
    'nearest': 0,
    'bilinear': 1,
    'biquadratic': 2,
    'bicubic': 3,
}


def center_crop_and_resize(image, image_size, crop_padding=32, interpolation='bicubic'):
    assert image.ndim in {2, 3}
    assert interpolation in MAP_INTERPOLATION_TO_ORDER.keys()

    h, w = image.shape[:2]

    padded_center_crop_size = int((image_size / (image_size + crop_padding)) * min(h, w))
    offset_height = ((h - padded_center_crop_size) + 1) // 2
    offset_width = ((w - padded_center_crop_size) + 1) // 2

    image_crop = image[offset_height:padded_center_crop_size + offset_height,
                       offset_width:padded_center_crop_size + offset_width]
    resized_image = resize(
        image_crop,
        (image_size, image_size),
        order=MAP_INTERPOLATION_TO_ORDER[interpolation],
        preserve_range=True,
    )

    return resized_image


def preprocess_input(x):
    assert x.ndim in (3, 4)
    assert x.shape[-1] == 3

    x = x - np.array(MEAN_RGB)
    x = x / np.array(STDDEV_RGB)

    return x
