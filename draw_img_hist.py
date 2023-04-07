import os
import torch
from torchvision import transforms
import numpy as np
from matplotlib import pyplot as plt
import cv2

img_dir = "./data/regions"
# img_dir = "./data/cropped"
img_list = os.listdir(img_dir)

normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
)

transformer_ori = transforms.Compose([transforms.ToTensor()])
transformer_norm = transforms.Compose([transforms.ToTensor(), normalize])

for fname in img_list:
    curr_img = cv2.imread(f"{img_dir}/{fname}")  # BGR
    curr_img_ori = transformer_ori(curr_img)
    curr_img_normalized = transformer_norm(curr_img)

    fig = plt.figure(figsize=(15, 6))

    for channel in range(curr_img.shape[2]):  # channel last in pytorch
        if channel == 0:
            color = "Blue"
        elif channel == 1:
            color = "Green"
        else:
            color = "Red"

        histogram_ori, bin_edges_ori = np.histogram(
            curr_img_ori.numpy()[..., channel], bins=256, range=(0, 1)
        )
        histogram_norm, bin_edges_norm = np.histogram(
            curr_img_normalized.numpy()[..., channel], bins=256
        )

        hist_max_orig = histogram_ori.max()
        hist_max_norm = histogram_norm.max()

        mean_orig = curr_img_ori.mean()
        std_orig = curr_img_ori.std()

        mean_norm = curr_img_normalized.mean()
        std_norm = curr_img_normalized.std()

        plt.subplot(1, 3, channel + 1)
        # plt.plot(bin_edges_ori[0:-1], histogram_ori, label="orig")
        plt.plot(
            bin_edges_ori[0:-1],
            histogram_ori / hist_max_orig,
            # histogram_ori,
            label=f"orig (m: {mean_orig: .2f}, s: {std_orig: .2f})",
        )
        # plt.plot(bin_edges_norm[0:-1], histogram_norm, label="norm")
        plt.plot(
            bin_edges_norm[0:-1],
            histogram_norm / hist_max_norm,
            # histogram_norm,
            label=f"norm (m: {mean_norm: .2f}, s: {std_norm: .2f})",
        )

        plt.title(f"{color} Histogram")
        plt.xlabel(f"{color} value (scaled)")
        plt.xlim([-2.5, 2.5])
        plt.ylim([0, 1])
        # plt.ylim([0, hist_max_norm] if hist_max_norm > hist_max_orig else [0, hist_max_orig])
        plt.ylabel("pixel count (scaled)")
        plt.legend(loc="best")

    plt.tight_layout()
    plt.show()
    a = 1
