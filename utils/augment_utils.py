import os
from typing import Tuple
import numpy as np


def parse_label(fname: str) -> np.ndarray:
    with open(fname, encoding="utf-8") as f:
        bboxes = f.readlines()
        label = []

    for bbox in bboxes:
        label.append(bbox.split())

    return np.array(label, dtype=np.float64)


def label_yolo2voc(label_yolo: np.ndarray, h: int, w: int) -> np.ndarray:
    # from: (x_center, y_center, w, h), normalized
    # to:   (xtl, ytl, xbr, ybr)
    label_voc = np.zeros(label_yolo.shape, dtype=np.float64)
    label_voc[:, 0] = label_yolo[:, 0]

    label_yolo_temp = label_yolo.copy()
    label_yolo_temp[:, [1, 3]] *= w
    label_yolo_temp[:, [2, 4]] *= h

    # convert x_center, y_center to xtl, ytl
    label_voc[:, 1] = label_yolo_temp[:, 1] - 0.5 * label_yolo_temp[:, 3]
    label_voc[:, 2] = label_yolo_temp[:, 2] - 0.5 * label_yolo_temp[:, 4]

    # convert width, height to xbr, ybr
    label_voc[:, 3] = label_voc[:, 1] + label_yolo_temp[:, 3]
    label_voc[:, 4] = label_voc[:, 2] + label_yolo_temp[:, 4]

    return label_voc.astype(np.uint32)


def label_voc2yolo(label_voc: np.ndarray, h: int, w: int) -> np.ndarray:
    # from: (xtl, ytl, xbr, ybr)
    # to:   (x_center, y_center, w, h), normalized

    label_yolo = np.zeros(label_voc.shape, dtype=np.float64)
    label_yolo[:, 0] = label_voc[:, 0]

    # convert xtl, ytl to x_center, y_center
    label_yolo[:, 1] = 0.5 * (label_voc[:, 1] + label_voc[:, 3])
    label_yolo[:, 2] = 0.5 * (label_voc[:, 2] + label_voc[:, 4])

    # convert xbr, ybr to width, height
    label_yolo[:, 3] = label_voc[:, 3] - label_voc[:, 1]
    label_yolo[:, 4] = label_voc[:, 4] - label_voc[:, 2]

    # normalize
    label_yolo[:, [1, 3]] /= w
    label_yolo[:, [2, 4]] /= h

    return label_yolo


def find_draw_region(img: np.ndarray, label: np.ndarray, foreground: np.ndarray) -> Tuple[int]:
    h, w = img.shape[:2]
    w_fg, h_fg = foreground.shape[:2]
    label_pixel = np.copy(label)
    label_pixel[:, [1, 3]] *= w
    label_pixel[:, [2, 4]] *= h

    # convert x_center, y_center to xtl, ytl
    label_pixel[:, 1] = label_pixel[:, 1] - 0.5 * label_pixel[:, 3]
    label_pixel[:, 2] = label_pixel[:, 2] - 0.5 * label_pixel[:, 4]

    # convert width, height to xbr, ybr
    label_pixel[:, 3] = label_pixel[:, 1] + label_pixel[:, 3]
    label_pixel[:, 4] = label_pixel[:, 2] + label_pixel[:, 4]

    label_pixel = label_pixel.astype(np.uint32)

    xtl = label_pixel[:, 1].min()
    ytl = label_pixel[:, 2].min()
    xbr = label_pixel[:, 3].max()
    ybr = label_pixel[:, 4].max()

    region_candidate = [False] * 8
    region_area = [0.0] * 8

    # region 1
    w1, h1 = xtl, ytl
    region_candidate[0] = (w1 >= w_fg) and (h1 >= h_fg)
    region_area[0] = w1 * h1

    # region 2
    w2, h2 = xbr - xtl, ytl
    region_candidate[1] = (w2 >= w_fg) and (h2 >= h_fg)
    region_area[1] = w2 * h2

    # region 3
    w3, h3 = w - xbr, ytl
    region_candidate[2] = (w3 >= w_fg) and (h3 >= h_fg)
    region_area[2] = w3 * h3

    # region 4
    w4, h4 = xtl, ybr - ytl
    region_candidate[3] = (w4 >= w_fg) and (h4 >= h_fg)
    region_area[3] = w4 * h4

    # region 5
    w5, h5 = w - xbr, ybr - ytl
    region_candidate[4] = (w5 >= w_fg) and (h5 >= h_fg)
    region_area[4] = w5 * h5

    # region 6
    w6, h6 = xtl, h - ybr
    region_candidate[5] = (w6 >= w_fg) and (h6 >= h_fg)
    region_area[5] = w6 * h6

    # region 7
    w7, h7 = xbr - xtl, h - ybr
    region_candidate[6] = (w7 >= w_fg) and (h7 >= h_fg)
    region_area[6] = w7 * h7

    # region 8
    w8, h8 = w - xbr, h - ybr
    region_candidate[7] = (w8 >= w_fg) and (h8 >= h_fg)
    region_area[7] = w8 * h8

    region_candidate = np.array(region_candidate, dtype=np.uint32)
    region_area = np.array(region_area, dtype=np.uint32)

    selected_region = (region_candidate * region_area).argmax() + 1

    if selected_region == 1:
        area = (selected_region, 0, 0, xtl, ytl)
    elif selected_region == 2:
        area = (selected_region, xtl, 0, xbr, ytl)
    elif selected_region == 3:
        area = (selected_region, xbr, 0, w, ytl)
    elif selected_region == 4:
        area = (selected_region, 0, ytl, xtl, ybr)
    elif selected_region == 5:
        area = (selected_region, xbr, ytl, w, ybr)
    elif selected_region == 6:
        area = (selected_region, 0, ybr, xtl, h)
    elif selected_region == 7:
        area = (selected_region, xtl, ybr, xbr, h)
    elif selected_region == 8:
        area = (selected_region, xbr, ybr, w, h)

    return area


def write_label(target_dir: str, fname: str, bboxes: np.ndarray) -> None:
    num_boxes = bboxes.shape[0]

    with open(f"{target_dir}/{fname}.txt", "w") as f:
        for i in range(num_boxes):
            target_str = f"{int(bboxes[i][0])} {bboxes[i][1]} {bboxes[i][2]} {bboxes[i][3]} {bboxes[i][4]}"
            f.write(f"{target_str}\n")


