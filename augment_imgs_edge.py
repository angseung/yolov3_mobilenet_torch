import os
import random
import platform
import cv2
import numpy as np
from utils.augment_utils import (
    parse_label,
    write_label,
    auto_canny,
)

random.seed(123)


if __name__ == "__main__":
    if "Windows" in platform.platform():
        bg_img_dir = "./data/yperv1/images/train"
        bg_label_dir = "./data/yperv1/labels/train"
        fg_img_dir = "../kor_license_plate_generator/DB_new/images/train"
        fg_label_dir = "../kor_license_plate_generator/DB_new/labels/train"
        target_dir = "./data/yper_edge"

    elif "Linux" in platform.platform():
        bg_img_dir = "/data_yper/yperv2.1/images/train"
        bg_label_dir = "/data_yper/yperv2.1/labels/train"
        fg_img_dir = "/data_yper/addons_v1.2/images/train"
        fg_label_dir = "/data_yper/addons_v1.2/labels/train"
        target_dir = "/data_yper/yper_edge"

    if not os.path.isdir(target_dir):
        os.makedirs(f"{target_dir}/images/train", exist_ok=True)
        os.makedirs(f"{target_dir}/labels/train", exist_ok=True)

    if os.path.isfile("img_list.txt"):
        with open("img_list.txt", encoding="utf-8") as f:
            bg_img_list = f.readlines()
    else:
        bg_img_list = os.listdir(bg_img_dir)

    fg_img_list = os.listdir(fg_img_dir)
    random.shuffle(bg_img_list)
    random.shuffle(fg_img_list)
    processed = 0
    remainder = len(fg_img_list) - len(bg_img_list)

    for i, (bg_file_name, fg_file_name) in enumerate(zip(bg_img_list, fg_img_list)):
        print(f"processing {i}/{len(bg_img_list)}th sample, {bg_file_name}")

        # remove suffix and new line
        bg_file_name = bg_file_name[:-1]

        bg_img = cv2.imread(f"{bg_img_dir}/{bg_file_name}.jpg")
        bg_label = parse_label(f"{bg_label_dir}/{bg_file_name}.txt")

        # convert to gray and detect edge
        new_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
        new_img = auto_canny(new_img)

        # export augmented data
        cv2.imwrite(f"{target_dir}/images/train/{bg_file_name}.jpg", new_img)
        write_label(f"{target_dir}/labels/train", bg_file_name, bg_label)
