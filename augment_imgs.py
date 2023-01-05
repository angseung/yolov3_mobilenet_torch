import os
import random
import numpy as np
import cv2
from utils.augment_utils import parse_label, label_yolo2voc, label_voc2yolo, find_draw_region, write_label, draw_bbox_on_img


if __name__ == "__main__":
    img_dir = "./data/yperv1/images/train"
    label_dir = "./data/yperv1/labels/train"
    target_dir = "./data/yperv2"

    if not os.path.isdir(target_dir):
        os.makedirs(f"{target_dir}/images/train")
        os.makedirs(f"{target_dir}/labels/train")

    img_list = os.listdir(img_dir)

    for i, fname in enumerate(img_list[:]):
        fname = fname[:-4]  # remove suffix and new line
        bg_label = parse_label(f"{label_dir}/{fname}.txt")
        bg_img = cv2.imread(f"{img_dir}/{fname}.jpg")
        bg_h, bg_w = bg_img.shape[:2]
        fg_img = cv2.imread("Z03ah4432X.jpg")
        fg_label = parse_label("Z03ah4432X.txt")
        fg_h, fg_w = fg_img.shape[:2]

        selected_region, area_xtl, area_ytl, area_xbr, area_ybr = find_draw_region(bg_img, bg_label, fg_img)
        assert area_xbr > area_xtl and area_ybr > area_ytl

        allowed_width = (area_xbr - area_xtl - fg_w)
        allowed_height = (area_ybr - area_ytl - fg_h)

        draw_xtl = random.randint(0, allowed_width - 1)
        draw_ytl = random.randint(0, allowed_height - 1)

        label_voc = label_yolo2voc(bg_label, bg_h, bg_w)
        label_yolo = label_voc2yolo(label_voc, bg_h, bg_w)

        abs_xtl, abs_ytl = draw_xtl + area_xtl, draw_ytl + area_ytl

        # draw fg_img on bg_img
        try:
            bg_img[abs_ytl : abs_ytl + fg_h, abs_xtl : abs_xtl + fg_w, :] = fg_img[:, :, :]
        except:
            a = 1

        # compensate bbox offset of fg_label
        fg_label_voc = label_yolo2voc(fg_label, fg_h, fg_w)
        fg_label_voc[:, [1, 3]] += abs_xtl
        fg_label_voc[:, [2, 4]] += abs_ytl
        fg_label_yolo = label_voc2yolo(fg_label_voc, bg_h, bg_w)

        label = np.concatenate((fg_label_yolo, bg_label), axis=0)

        # draw bbox for debug
        bg_img = draw_bbox_on_img(bg_img, label)

        # export augmented data
        cv2.imwrite(f"{target_dir}/images/train/{fname}.jpg", bg_img)
        write_label(f"{target_dir}/labels/train", fname, label)
        a = 1
