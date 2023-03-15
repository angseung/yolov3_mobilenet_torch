from typing import *
import numpy as np
from .classes_map import class_labels


def angle_between(p1: List[float], p2: List[float], signed: Optional[bool] = False) -> float:
    delta_x = (p2[0] - p1[0])
    delta_y = (p2[1] - p1[1])
    angle = np.arctan2(delta_y, delta_x) * 180 / np.pi

    return angle if signed else abs(angle)


def read_bboxes(bboxes: np.ndarray, angular_thresh: Optional[Union[float, int]] = 25.0) -> str:
    """
    bboxes: voc format, (xtl, ytl, xbr, ybr, confidence, classes_labels)
    """

    bboxes = bboxes.detach().cpu()
    bboxes_sequence = np.argsort(bboxes[:, 0].detach().cpu().flatten())
    bboxes_reordered = bboxes[bboxes_sequence]

    a, b = split_plate_line(bboxes_reordered, angular_thresh)
    a = bboxes_to_string(a)
    b = bboxes_to_string(b)

    plate_string = a + b

    return plate_string


def bboxes_to_string(bboxes: np.ndarray) -> str:
    bboxes = bboxes.detach().cpu()
    plate_string = ""
    bboxes_xtl = bboxes[:, 0].detach().cpu().flatten()
    bboxes_sequence = np.argsort(bboxes_xtl)

    for curr_bbox in bboxes_sequence:
        label_index = int(bboxes[curr_bbox][5].tolist())
        curr_string = class_labels[label_index]
        plate_string += curr_string

    return plate_string


def split_plate_line(bboxes: np.ndarray, angular_thresh: int) -> Tuple[np.ndarray, np.ndarray]:
    bboxes = bboxes.detach().cpu()

    # True line is first line, and False line is second line
    line_of_bbox = np.array([False] * bboxes.shape[0])

    for i, bbox in enumerate(bboxes[:-1, :]):
        if i == 0:
            line_of_bbox[i] = False

        p1 = bboxes[i][:2].tolist()
        p2 = bboxes[i + 1][:2].tolist()
        angle = angle_between(p1, p2)

        is_another_line = True if angle >= angular_thresh else False
        line_of_bbox[i + 1] = line_of_bbox[i] ^ is_another_line

    bbox_in_first_line = bboxes[line_of_bbox]
    bbox_in_second_line = bboxes[np.logical_not(line_of_bbox)]

    if bbox_in_first_line.shape[0] > bbox_in_second_line.shape[0]:
        bbox_in_first_line, bbox_in_second_line = bbox_in_second_line, bbox_in_first_line

    return bbox_in_first_line, bbox_in_second_line
