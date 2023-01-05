import os
import cv2
from utils.augment_utils import (
    parse_label,
    write_label,
    draw_bbox_on_img,
    augment_img,
)


if __name__ == "__main__":
    bg_img_dir = "./data/yperv1/images/train"
    bg_label_dir = "./data/yperv1/labels/train"
    fg_img_dir = "./data/addons/images/train"
    fg_label_dir = "./data/addons/labels/train"
    target_dir = "./data/yperv2"

    if not os.path.isdir(target_dir):
        os.makedirs(f"{target_dir}/images/train")
        os.makedirs(f"{target_dir}/labels/train")

    img_list = os.listdir(bg_img_dir)

    for i, fname in enumerate(img_list[:]):
        fname = fname[:-4]  # remove suffix and new line
        bg_label = parse_label(f"{bg_label_dir}/{fname}.txt")
        bg_img = cv2.imread(f"{bg_img_dir}/{fname}.jpg")
        fg_img = cv2.imread("Z03ah4432X.jpg")
        fg_label = parse_label("Z03ah4432X.txt")

        new_img, new_label = augment_img(
            fg_img=fg_img, fg_label=fg_label, bg_img=bg_img, bg_label=bg_label
        )

        # draw bbox for debug
        new_img = draw_bbox_on_img(new_img, new_label)

        # export augmented data
        cv2.imwrite(f"{target_dir}/images/train/{fname}.jpg", new_img)
        write_label(f"{target_dir}/labels/train", fname, new_label)
