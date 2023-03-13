from typing import *
import numpy as np
from .classes_map import class_labels


def read_bboxes(bboxes: np.ndarray, tolerance: Optional[float] = 0.3) -> str:
    """
    bboxes: voc format, (xtl, ytl, xbr, ybr, confidence, classes_labels)
    """
    is_multi_row = False
    plate_string = ""
    index_second_row = None

    # check plate has a row or 2 rows
    bboxes_ytl = np.sort(bboxes[:, 1].flatten())
    ytl_first_row = bboxes_ytl[0]

    for i, ytl in enumerate(bboxes_ytl[1:]):
        if ytl_first_row - tolerance * ytl_first_row <= ytl <= ytl_first_row + tolerance * ytl_first_row:
            ytl_first_row = ytl
            continue
        else:
            index_second_row = i
            is_multi_row = True
            break

    # read plates
    if is_multi_row and index_second_row:  # short plates
        bboxes_h_thresh = int(bboxes_ytl[index_second_row])

        bboxes_in_first_row = bboxes[bboxes[:, 1] <= bboxes_h_thresh]
        bboxes_in_second_row = bboxes[bboxes[:, 1] > bboxes_h_thresh]
        plate_string_first_row = bboxes_to_string(bboxes_in_first_row)
        plate_string_second_row = bboxes_to_string(bboxes_in_second_row)
        plate_string = plate_string_first_row + plate_string_second_row

    else:  # long plates
        plate_string = bboxes_to_string(bboxes)

    return plate_string


def bboxes_to_string(bboxes: np.ndarray) -> str:
    plate_string = ""
    bboxes_xtl = bboxes[:, 0].flatten()
    bboxes_sequence = np.argsort(bboxes_xtl)

    for curr_bbox in bboxes_sequence:
        label_index = int(bboxes[curr_bbox][5].tolist())
        curr_string = class_labels[label_index]
        plate_string += curr_string

    return plate_string
