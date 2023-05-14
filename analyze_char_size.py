import os
from typing import *
import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils.augment_utils import label_yolo2voc
from utils.roi_utils import parse_detection_results, resize

base_dir = "./runs/detect/best_004"
height_list: List = []
width_list: List = []

for fname in os.listdir(base_dir):
    if "jpg" not in fname:
        continue
    img = cv2.imread(f"{base_dir}/{fname}")
    label = parse_detection_results(f"{base_dir}/labels/{fname[:-4]}.txt")
    img_resized = resize(img, size=320)
    height, width = img_resized.shape[:2]

    label_voc = label_yolo2voc(label, h=height, w=width)

    for bbox in label_voc:
        curr_width = bbox[3] - bbox[1]
        curr_height = bbox[4] - bbox[2]
        width_list.append(curr_width)
        height_list.append(curr_height)

result = np.array([width_list, height_list])
width_min, width_max = result[0, :].min(), result[0, :].max()
height_min, height_max = result[1, :].min(), result[1, :].max()

counts_width, bins_width = np.histogram(result[0, :], bins=20)
counts_height, bins_height = np.histogram(result[1, :], bins=20)

fig = plt.figure()
plt.subplot(211)
plt.stairs(counts_width, bins_width)
plt.subplot(212)
plt.stairs(counts_height, bins_height)
plt.show()
