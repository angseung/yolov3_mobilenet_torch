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
    bg_img_dir = "./data/yperv1/images/train"
    bg_label_dir = "./data/yperv1/labels/train"
    fg_img_dir = "./data/addons/images/train"
    fg_label_dir = "./data/addons/labels/train"
    target_dir = "./data/yperv2"

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

    for i, (bg_file_name, fg_file_name) in enumerate(zip(bg_img_list, fg_img_list)):
        print(f"processing {i}/{len(bg_img_list)}th sample, {bg_file_name}")

        # remove suffix and new line
        bg_file_name = bg_file_name[:-1]
        fg_file_name = fg_file_name[:-4]

        bg_img = cv2.imread(f"{bg_img_dir}/{bg_file_name}.jpg")
        bg_label = parse_label(f"{bg_label_dir}/{bg_file_name}.txt")

        fg_img = cv2.imread(f"{fg_img_dir}/{fg_file_name}.jpg")
        fg_label = parse_label(f"{fg_label_dir}/{fg_file_name}.txt")

        # random resize fg img
        fg_img, fg_label = random_resize(img=fg_img, label=fg_label)

        new_img, new_label, is_processed = augment_img(
            fg_img=fg_img, fg_label=fg_label, bg_img=bg_img, bg_label=bg_label
        )

        # draw bbox for debug
        new_img = draw_bbox_on_img(new_img, new_label)

        # export augmented data
        cv2.imwrite(f"{target_dir}/images/train/{bg_file_name}.jpg", new_img)
        write_label(f"{target_dir}/labels/train", bg_file_name, new_label)
