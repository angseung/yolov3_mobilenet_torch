import os
import shutil
import random
from typing import List, Dict, Union
import cv2
import numpy as np
from matplotlib import pyplot as plt

random.seed(123)
base_dir: str = "./data/yperv1/images/train"
target_dir: str = "./data/resized_samples"
im_list: List[str] = os.listdir(base_dir)
num_samples: int = 1000
target_im_size: int = 320

im_to_select: List[int] = random.sample(range(0, len(im_list) - 1), num_samples)
interpolation_methods: Dict[str, Union[int, List[int]]] = {
    "INTER_NEAREST": cv2.INTER_NEAREST,
    "INTER_LINEAR": cv2.INTER_LINEAR,
    "INTER_CUBIC": cv2.INTER_CUBIC,
    "INTER_LANCZOS4": cv2.INTER_LANCZOS4,
    "INTER_AREA": cv2.INTER_AREA,
    "INTER_LINEAR_EXACT": cv2.INTER_LINEAR_EXACT,
    "INTER_NEAREST_EXACT": cv2.INTER_NEAREST_EXACT,
    "WEIGHTED_JOINT_BN": [cv2.INTER_LINEAR, cv2.INTER_NEAREST],
    "WEIGHTED_JOINT_CN": [cv2.INTER_CUBIC, cv2.INTER_NEAREST],
}
k = 0.27

if not os.path.isdir(target_dir):
    os.makedirs(target_dir)

for image_num in im_to_select:
    fname = im_list[image_num]
    image = cv2.imread(f"{base_dir}/{fname}")
    height, width, _ = image.shape
    image_ratio = max(width, height) / min(width, height)
    target_size = (target_im_size, int(image_ratio * target_im_size)) if width < height else (int(image_ratio * target_im_size), target_im_size)
    fig = plt.figure(figsize=(10, 10))

    for i, (method, method_enum) in enumerate(interpolation_methods.items()):
        plt.subplot(3, 3, i + 1)

        if isinstance(method_enum, list):
            image_resized = (1.0 + k) * cv2.resize(image, dsize=target_size, interpolation=method_enum[0]) - k * cv2.resize(image, dsize=target_size, interpolation=method_enum[1])
            image_resized = image_resized.astype(np.uint8)
        else:
            image_resized = cv2.resize(image, dsize=target_size, interpolation=method_enum)

        plt.imshow(image_resized)
        plt.xlabel(method)

    plt.tight_layout()
    plt.suptitle("fname")
    plt.show()
    fig.savefig(f"{target_dir}/{fname}.png")
    del fig