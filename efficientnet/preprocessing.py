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
from skimage.transform import resize

MAP_INTERPOLATION_TO_ORDER = {
    "nearest": 0,
    "bilinear": 1,
    "biquadratic": 2,
    "bicubic": 3,
}


def center_crop_and_resize(image, image_size, crop_padding=32, interpolation="bicubic"):
    assert image.ndim in {2, 3}
    assert interpolation in MAP_INTERPOLATION_TO_ORDER.keys()

    h, w = image.shape[:2]

    padded_center_crop_size = int(
        (image_size / (image_size + crop_padding)) * min(h, w)
    )
    offset_height = ((h - padded_center_crop_size) + 1) // 2
    offset_width = ((w - padded_center_crop_size) + 1) // 2

    image_crop = image[
                 offset_height: padded_center_crop_size + offset_height,
                 offset_width: padded_center_crop_size + offset_width,
                 ]
    resized_image = resize(
        image_crop,
        (image_size, image_size),
        order=MAP_INTERPOLATION_TO_ORDER[interpolation],
        preserve_range=True,
    )

    return resized_image
