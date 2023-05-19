# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov3.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""
import os
import sys
from pathlib import Path
from typing import *

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from PIL import ImageFont, ImageDraw, Image

try:
    import RPi.GPIO as GPIO
    from picamera2 import Picamera2

except ImportError:
    raise RuntimeError("this code can be run only on Raspberry Pi")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
fontpath = "fonts/NanumBarunGothic.ttf"
font = ImageFont.truetype(fontpath, 36)

from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import (
    LOGGER,
    check_img_size,
    check_requirements,
    non_max_suppression,
    print_args,
    scale_coords,
    xyxy2xywh,
)

from utils.classes_map import map_class_index_to_target
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync, normalizer
from utils.augment_utils import label_yolo2voc, label_voc2yolo
from utils.detect_utils import read_bboxes, correction_plate
from utils.roi_utils import crop_region_of_plates, resize, rescale_roi
from utils.augmentations import wrap_letterbox, letterbox


@torch.no_grad()
def run(
    weights=ROOT / "weights/107.pt",  # model.pt path(s)
    imgsz=640,  # inference size (pixels)
    cropped_imgsz=256,
    conf_thres=0.2,  # confidence threshold
    iou_thres=0.05,  # NMS IOU threshold
    max_det=300,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=True,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    normalize=True,
    gray=False,
    rm_doubled_bboxes=True,
    use_soft=False,
    print_string=True,
    roi_crop=True,
    use_yolo=True,
):

    picam2 = init_cam(pin_num=18, color_format="RGB888")
    assert not (
        normalize and gray
    )  # select gray or normalize. when selected both, escapes.

    plate_string = ""

    # Load model
    device = select_device(device)

    model = DetectMultiBackend(weights, device=device, dnn=dnn)

    # use ROI detection with yolo
    if roi_crop and use_yolo:
        pth_path = os.path.join(str(FILE.parents[0]), "weights", "202.pt")
        roi_model = DetectMultiBackend(pth_path, device=device, dnn=dnn)
        roi_model.model.float()

    stride, names, pt, jit, onnx = (
        model.stride,
        model.names,
        model.pt,
        model.jit,
        model.onnx,
    )
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    transform_normalize = normalizer()

    # Mapping class index to real value for yper data
    if len(names) == 84:
        class_labels = map_class_index_to_target(names)
        if class_labels:
            model.names = class_labels
            names = class_labels
    if pt:
        model.model.half() if half else model.model.float()

    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    cv2.namedWindow("demo")

    # Run inference
    while True:
        t1 = time_sync()
        im0s = picam2.capture_array()  # im0s: HWC, BGR
        im = letterbox(im0s, [imgsz, imgsz], stride=32, auto=True)[0]
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # im: CHW, RGB

        # crop with ROI detector
        if roi_crop:
            im_befroe_crop = im.copy()
            xtl_crop, ytl_crop, xbr_crop, ybr_crop = crop_region_of_plates(
                img=im.copy().transpose([1, 2, 0]),
                model=roi_model if use_yolo else None,
                target_imgsz=cropped_imgsz,
                char_size=25,
                imgsz=imgsz,
                use_yolo=use_yolo,
                top_only=True,
                img_show_opt=False,
                return_as_img=False,
            )

            # crop around the roi
            im_before_letterboxed = im[
                :, ytl_crop:ybr_crop, xtl_crop:xbr_crop
            ].transpose(
                [1, 2, 0]
            )  # (H, W, C)
            size_of_im_before_letterboxed = im_before_letterboxed.shape
            (
                scaled_xtl_crop,
                scaled_ytl_crop,
                scaled_xbr_crop,
                scaled_ybr_crop,
            ) = rescale_roi(
                region_of_roi=(xtl_crop, ytl_crop, xbr_crop, ybr_crop),
                prev_shape=im0s.shape[:2],
                resized_shape=im_befroe_crop.shape[1:],
            )

            # resize image
            im_before_letterboxed_resized = resize(im_before_letterboxed, cropped_imgsz)
            size_of_im_before_letterboxed_resized = im_before_letterboxed_resized.shape

            # pad img to make square-shaped img
            im, pad_axis = wrap_letterbox(
                im_before_letterboxed_resized, target_size=cropped_imgsz
            )  # (H, W, C)

            # convert channel first to last and BGR to RGB
            im = im.transpose((2, 0, 1))[::-1]  # (C, H, W)

            # calculate additional offset between letterboxed one and cropped one
            pad_offset = (
                cropped_imgsz - min(*im_before_letterboxed_resized.shape[:2])
            ) // 2

        im = torch.from_numpy(im.copy()).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0

        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        if normalize:
            im = transform_normalize(im)

        t2 = time_sync()
        dt[0] += t2 - t1  # elapsed time to preprocess

        pred = model(im, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2  # elapsed time to inference

        # NMS
        pred = non_max_suppression(
            pred,
            conf_thres,
            iou_thres,
            classes,
            agnostic_nms,
            max_det=max_det,
            use_soft=use_soft,
        )

        # secondary nms to drop missing doubled bbox
        if rm_doubled_bboxes:
            tmp = (
                nms(boxes=pred[0][:, :4], scores=pred[0][:, 4], iou_threshold=iou_thres)
                .detach()
                .tolist()
            )
            pred = [pred[0][tmp]]

        # compensate point offset if im is cropped
        if roi_crop:
            # pred: [xtl, ytl, xbr, ybr, conf, label] in VOC format
            pred_numpy = pred[0].numpy()

            # step 1. compensate padding offset
            if pad_axis == 0:
                pred_numpy[:, [1, 3]] -= pad_offset
            else:
                pred_numpy[:, [0, 2]] -= pad_offset

            # step 2. rescale bboxes
            pred_numpy_yolo = label_voc2yolo(
                pred_numpy[:, [5, 0, 1, 2, 3]],
                *size_of_im_before_letterboxed_resized[:2],
            )
            pred_numpy_scaled = label_yolo2voc(
                pred_numpy_yolo, *size_of_im_before_letterboxed[:2]
            )
            pred_numpy[:, [0, 1, 2, 3]] = pred_numpy_scaled[:, 1:]

            # step 2. compensate crop offset
            pred_numpy[:, [0, 2]] += xtl_crop
            pred_numpy[:, [1, 3]] += ytl_crop

            # step 3. convert numpy ndarray to torch tensor
            pred = [torch.from_numpy(pred_numpy)]

            # finally, replace im to original one
            im = im_befroe_crop.copy()[None]

        t4 = time_sync()
        dt[2] += t4 - t3  # elapsed time to nms

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            im0 = im0s.copy()
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det.size()):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Reorder: number first, Korean last
                _, indices = torch.sort(det[:, 5], descending=True)
                det = det[indices]

                # make bboxes to korean string
                if print_string:
                    plate_string = (
                        correction_plate(read_bboxes(det, angular_thresh=28.0))
                        if len(det) < 9
                        else ""
                    )

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if len(det.size()):  # Add bbox to image
                        c = int(cls)  # integer class
                        label = (
                            None
                            if hide_labels
                            else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        )
                        try:
                            annotator.box_label(xyxy, label, color=colors(c, True))

                        except ValueError:
                            print("Error occured")

            # Stream results
            im0 = annotator.result()
            img_pillow = Image.fromarray(im0)
            draw = ImageDraw.Draw(img_pillow)

            if len(det.size()):
                draw.text(
                    (10, 10),
                    plate_string,
                    font=font,
                    fill=(0, 0, 0),
                    stroke_width=2,
                    stroke_fill=(255, 255, 255),
                )

            im0 = np.array(img_pillow)

            # print plate string on im0
            if roi_crop and len(det):
                im0 = cv2.rectangle(
                    im0,
                    (scaled_xtl_crop, scaled_ytl_crop),
                    (scaled_xbr_crop, scaled_ybr_crop),
                    (255, 0, 0),
                    5,
                )

        dt[3] += time_sync() - t4  # elapsed time to draw bboxes and plate string
        print(f"curr FPS: {(1 / (dt[3] - dt[0])): .4f}")

        cv2.imshow("demo", im0)
        cv2.waitKey(1)


def init_cam(pin_num: int = 18, resolution: Union[Tuple[int, int], int] = (640, 360), color_format: str = "RGB888"):
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(pin_num, GPIO.IN, GPIO.PUD_UP)

    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(
            main={"format": color_format, "size": resolution}
        )
    )
    picam2.start()

    return picam2


def main():
    check_requirements(exclude=("tensorboard", "thop"))
    run()


if __name__ == "__main__":
    main()
