import os
import random
import cv2
from utils.augment_utils import (
    parse_label,
    write_label,
    draw_bbox_on_img,
    augment_img,
    random_resize,
)

random.seed(123)

if __name__ == "__main__":
    bg_img_dir = "./data/yper_edge/images/train"
    bg_label_dir = "./data/yper_edge/labels/train"
    target_dir = "./data/yper_bbox"

    if not os.path.isdir(target_dir):
        os.makedirs(f"{target_dir}/images/train", exist_ok=True)
        os.makedirs(f"{target_dir}/labels/train", exist_ok=True)

    if os.path.isfile("img_list_edge.txt"):
        with open("img_list_edge.txt", encoding="utf-8") as f:
            bg_img_list = f.readlines()
    else:
        bg_img_list = os.listdir(bg_img_dir)

    random.shuffle(bg_img_list)

    for i, bg_file_name in enumerate(bg_img_list):
        print(f"processing {i}/{len(bg_img_list)}th sample, {bg_file_name}")

        # remove suffix and new line
        bg_file_name = bg_file_name[:-1]

        bg_img = cv2.imread(f"{bg_img_dir}/{bg_file_name}.jpg")
        bg_label = parse_label(f"{bg_label_dir}/{bg_file_name}.txt")

        # draw bbox for debug
        bg_img = draw_bbox_on_img(bg_img, bg_label, color=(128, 128, 128))

        # export augmented data
        cv2.imwrite(f"{target_dir}/images/train/{bg_file_name}.jpg", bg_img)
        write_label(f"{target_dir}/labels/train", bg_file_name, bg_label)
