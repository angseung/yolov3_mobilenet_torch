from typing import *
import numpy as np
from .classes_map import class_labels
from .augment_utils import label_voc2yolo, label_yolo2voc


def angle_between(p1: List[float], p2: List[float]) -> float:
    delta_x = (p2[0] - p1[0])
    delta_y = (p2[1] - p1[1])

    return abs(np.arctan2(delta_y, delta_x) * 180 / np.pi)


def read_bboxes(bboxes: np.ndarray, tolerance: Optional[float] = 0.3) -> str:
    """
    bboxes: voc format, (xtl, ytl, xbr, ybr, confidence, classes_labels)
    """
    is_multi_row = False
    plate_string = ""
    index_second_row = None
    angular_thresh = 15.0

    # check plate has a row or 2 rows
    # bboxes_ytl = np.sort(bboxes[:, 1].detach().cpu().flatten())
    # ytl_first_row = bboxes_ytl[0]
    #
    # for i, ytl in enumerate(bboxes_ytl[1:]):
    #     if ytl_first_row - tolerance * ytl_first_row <= ytl <= ytl_first_row + tolerance * ytl_first_row:
    #         ytl_first_row = ytl
    #         continue
    #
    #     else:
    #         index_second_row = i
    #         is_multi_row = True
    #         break

    bboxes_xtl = np.sort(bboxes[:, 0].detach().cpu().flatten())
    bboxes_sequence = np.argsort(bboxes[:, 0].detach().cpu().flatten())
    bboxes_reordered = bboxes[bboxes_sequence]

    point_1 = bboxes_reordered[0][:2].tolist()
    point_2 = bboxes_reordered[1][:2].tolist()
    point_3 = bboxes_reordered[2][:2].tolist()

    angle_1 = angle_between(point_1, point_2) > angular_thresh
    angle_2 = angle_between(point_1, point_3) > angular_thresh
    angle_3 = angle_between(point_2, point_3) > angular_thresh

    is_multi_row = angle_1 or angle_2 or angle_3

    base_ytl = 1.25 * bboxes[:, 1].detach().cpu().flatten().numpy().mean()

    # read plates
    if is_multi_row:  # short plates
        bbox_mask = bboxes[:, 3] > base_ytl
        bboxes_in_second_row = bboxes[bbox_mask]
        bboxes_in_first_row = bboxes[np.logical_not(bbox_mask)]
        plate_string_first_row = bboxes_to_string(bboxes_in_first_row)
        plate_string_second_row = bboxes_to_string(bboxes_in_second_row)
        plate_string = plate_string_first_row + plate_string_second_row

    else:  # long plates
        plate_string = bboxes_to_string(bboxes)

    return plate_string


def bboxes_to_string(bboxes: np.ndarray) -> str:
    plate_string = ""
    bboxes_xtl = bboxes[:, 0].detach().cpu().flatten()
    bboxes_sequence = np.argsort(bboxes_xtl)

    for curr_bbox in bboxes_sequence:
        label_index = int(bboxes[curr_bbox][5].tolist())
        curr_string = class_labels[label_index]
        plate_string += curr_string

    return plate_string
