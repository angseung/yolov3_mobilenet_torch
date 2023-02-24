import os
import random
import platform
import cv2
import numpy as np
from utils.augment_utils import (
    parse_label,
    write_label,
    draw_bbox_on_img,
    augment_img,
    random_resize,
    blend_bgra_on_bgr,
)

random.seed(123)


def auto_canny(image, sigma=0.33):
    image = cv2.GaussianBlur(image, (7, 7), 0)
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


if __name__ == "__main__":
    if "Windows" in platform.platform():
        bg_img_dir = "./data/yperv1/images/train"
        bg_label_dir = "./data/yperv1/labels/train"
        fg_img_dir = "../kor_license_plate_generator/DB_new/images/train"
        fg_label_dir = "../kor_license_plate_generator/DB_new/labels/train"
        target_dir = "./data/yper_edge"

    elif "Linux" in platform.platform():
        bg_img_dir = "/data_yper/yperv1/images/train"
        bg_label_dir = "/data_yper/yperv1/labels/train"
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

    fg_img_list, fg_img_list_remainder = (
        fg_img_list[: len(bg_img_list)],
        fg_img_list[len(bg_img_list) :],
    )

    for i, (bg_file_name, fg_file_name) in enumerate(zip(bg_img_list, fg_img_list)):
        print(f"processing {i}/{len(bg_img_list)}th sample, {bg_file_name}")

        # remove suffix and new line
        bg_file_name = bg_file_name[:-1]
        fg_file_name = fg_file_name[:-4]

        bg_img = cv2.imread(f"{bg_img_dir}/{bg_file_name}.jpg")
        # bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
        bg_label = parse_label(f"{bg_label_dir}/{bg_file_name}.txt")

        fg_img = cv2.imread(f"{fg_img_dir}/{fg_file_name}.png", cv2.IMREAD_UNCHANGED)
        # fg_img = cv2.cvtColor(fg_img, cv2.COLOR_BGRA2GRAY)
        fg_label = parse_label(f"{fg_label_dir}/{fg_file_name}.txt")

        # random resize fg img
        fg_img, fg_label = random_resize(
            img=fg_img, label=fg_label, scale_min=1.0, scale_max=4.0
        )

        new_img, new_label, _ = augment_img(
            fg_img=fg_img, fg_label=fg_label, bg_img=bg_img, bg_label=bg_label
        )

        # append second fg_img
        if remainder > 1:
            fg_file_name_2 = fg_img_list_remainder[processed][:-4]

            fg_img_2 = cv2.imread(
                f"{fg_img_dir}/{fg_file_name_2}.png", cv2.IMREAD_UNCHANGED
            )
            # fg_img_2 = cv2.cvtColor(fg_img_2, cv2.COLOR_BGRA2GRAY)
            fg_label_2 = parse_label(f"{fg_label_dir}/{fg_file_name_2}.txt")

            # random resize fg img
            fg_img_2, fg_label_2 = random_resize(
                img=fg_img_2, label=fg_label_2, scale_min=1.0, scale_max=4.0
            )

            new_img, new_label, is_done = augment_img(
                fg_img=fg_img_2, fg_label=fg_label_2, bg_img=new_img, bg_label=new_label
            )

            processed += is_done
            remainder -= is_done


        # draw bbox for debug
        # new_img = draw_bbox_on_img(new_img, new_label)

        # convert to gray and detect edge
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        # new_img = cv2.GaussianBlur(new_img, (3, 3), 0)
        # threshold1 = 100
        # threshold2 = 500
        # new_img = cv2.Canny(new_img, threshold1, threshold2)
        new_img = auto_canny(new_img)

        # export augmented data
        cv2.imwrite(f"{target_dir}/images/train/{bg_file_name}.jpg", new_img)
        write_label(f"{target_dir}/labels/train", bg_file_name, new_label)
