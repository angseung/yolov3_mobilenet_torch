import os
import cv2
import numpy as np
from utils.augment_utils import label_yolo2voc
from utils.roi_utils import parse_detection_results

base_dir = "./runs/detect/best_004"

for fname in os.listdir(base_dir):
    if "jpg" not in fname:
        continue
    img = cv2.imread(f"{base_dir}/{fname}")
    label = parse_detection_results(f"{base_dir}/labels/{fname[:-4]}.txt")
    height, width = img.shape[:2]

    label_voc = label_yolo2voc(label, h=height, w=width)
