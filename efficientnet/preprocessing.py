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
try:
    import cv2
except ModuleNotFoundError:
    import skimage.transform


try:
    # OpenCV: Map interpolation string to OpenCV interpolation enum value
    INTERPOLATION_DICT = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "laconzos": cv2.INTER_LANCZOS4,
    }
except NameError:
    # scikit-image: Map interpolation string to interpolation order
    INTERPOLATION_DICT = {
        "nearest": 0,
        "bilinear": 1,
        "biquadratic": 2,
        "bicubic": 3,
    }


def center_crop_and_resize(image, image_size, crop_padding=32, interpolation="bicubic"):
    assert image.ndim in {2, 3}
    assert interpolation in INTERPOLATION_DICT.keys()

    in_h, in_w = image.shape[:2]

    if isinstance(image_size, (int, float)):
        out_h = out_w = image_size
    else:
        out_h, out_w = image_size

    if isinstance(crop_padding, (int, float)):
        crop_padding_h = crop_padding_w = crop_padding
    else:
        crop_padding_h, crop_padding_w = crop_padding

    padded_center_crop_shape_post_scaling = (out_h + crop_padding_h,
                                             out_w + crop_padding_w)

    inv_scale = min(in_h / padded_center_crop_shape_post_scaling[0],
                in_w / padded_center_crop_shape_post_scaling[1])

    unpadded_center_crop_size_pre_scaling = (round(out_h * inv_scale),
                                             round(out_w * inv_scale))

    offset_h = ((in_h - unpadded_center_crop_size_pre_scaling[0]) + 1) // 2
    offset_w = ((in_w - unpadded_center_crop_size_pre_scaling[1]) + 1) // 2

    image_crop = image[
                 offset_h : unpadded_center_crop_size_pre_scaling[0] + offset_h,
                 offset_w : unpadded_center_crop_size_pre_scaling[1] + offset_w,
                 ]

    try:
        resized_image = cv2.resize(
            image_crop,
            (out_w, out_h),
            interpolation=INTERPOLATION_DICT[interpolation] if inv_scale < 1 else cv2.INTER_AREA,
        )
    except NameError:
        resized_image = skimage.transform.resize(
            image_crop,
            (out_h, out_w),
            order=INTERPOLATION_DICT[interpolation],
            preserve_range=True,
        )

    return resized_image
