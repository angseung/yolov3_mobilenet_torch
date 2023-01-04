import os
import shutil
from typing import List, Tuple
import numpy as np


def find_draw_region(img: np.ndarray, label: np.ndarray, foreground: np.ndarray) -> Tuple[int]:
    w, h = img.shape[:2]
    w_fg, h_fg = foreground.shape[:2]
    label_pixel = np.copy(label)
    label_pixel[:, 1, 3] *= w
    label_pixel[:, 2, 4] *= h

    xtl = label_pixel[:, 1].min()
    ytl = label_pixel[:, 2].min()
    xbr = label_pixel[:, 3].max()
    ybr = label_pixel[:, 4].max()

    region_candidate = [False] * 8
    region_area = [0.0] * 8

    # region 1
    w1, h1 = xtl, ytl
    region_candidate[0] = (w1 >= w) and (h1 >= h)
    region_area[0] = w1 * h1

    # region 2
    w2, h2 = xbr - xtl, ytl
    region_candidate[1] = (w2 >= w) and (h2 >= h)
    region_area[1] = w2 * h2

    # region 3
    w3, h3 = w - xbr, ytl
    region_candidate[2] = (w3 >= w) and (h3 >= h)
    region_area[2] = w3 * h3

    # region 4
    w4, h4 = xtl, ybr - ytl
    region_candidate[3] = (w4 >= w) and (h4 >= h)
    region_area[3] = w4 * h4

    # region 5
    w5, h5 = w - xbr, ybr - ytl
    region_candidate[4] = (w5 >= w) and (h5 >= h)
    region_area[4] = w5 * h5

    # region 6
    w6, h6 = xtl, h - ybr
    region_candidate[5] = (w6 >= w) and (h6 >= h)
    region_area[5] = w6 * h6

    # region 7
    w7, h7 = xbr - xtl, h - ybr
    region_candidate[6] = (w7 >= w) and (h7 >= h)
    region_area[6] = w7 * h7

    # region 8
    w8, h8 = w - xbr, h - ybr
    region_candidate[7] = (w8 >= w) and (h8 >= h)
    region_area[7] = w8 * h8

    region_candidate = np.array(region_candidate, dtype=np.uint32)
    region_area = np.array(region_area, dtype=np.uint32)

    selected_region = (region_candidate * region_area).tolist().argmax() + 1

    if selected_region == 1:
        area = (0, 0, xtl, ytl)
    elif selected_region == 2:
        area = (xtl, 0, w - xbr, ytl)
    elif selected_region == 3:
        area = (w - xbr, 0, w, ytl)
    elif selected_region == 4:
        area = (0, ytl, xtl, h - ybr)
    elif selected_region == 5:
        area = (w - xbr, ytl, w, h - ybr)
    elif selected_region == 6:
        area = (0, h - ybr, xtl, h - ybr)
    elif selected_region == 7:
        area = (xtl, h - ybr, w - xbr, h)
    elif selected_region == 8:
        area = (xbr, ybr, w, h)

    return area


if __name__ == "__main__":
    img_dir = "/data_yper/yperv1/images/train"
    label_dir = "/data_yper/yperv1/labels/train"
    target_dir = "/data_yper/yperv2"

    if not os.path.isdir(target_dir):
        os.makedirs(f"{target_dir}/images/train")
        os.makedirs(f"{target_dir}/labels/train")

    img_list = os.listdir(img_dir)
