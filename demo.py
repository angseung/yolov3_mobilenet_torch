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
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from PIL import ImageFont, ImageDraw, Image

try:
    import RPi.GPIO as GPIO
    from picamera2 import Picamera2

    btnPin = 18
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(btnPin, GPIO.IN, GPIO.PUD_UP)

    picam2 = Picamera2()
    resolution = (640, 360)
    picam2.configure(
        picam2.create_preview_configuration(
            main={"format": "RGB888", "size": resolution}
        )
    )
    picam2.start()

except ImportError:
    raise RuntimeError("this code can be run only on Raspberry Pi")

cropped_imgsz = 256
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

    # Dataloader
    dt, seen = [0.0, 0.0, 0.0], 0
    cv2.namedWindow("demo")

    # Run inference
    while True:
        # im0s: HWC, BGR
        im0s = picam2.capture_array()
        im = letterbox(im0s, [imgsz, imgsz], stride=32, auto=True)[0]
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB

        # im:   CHW, RGB
        im = np.ascontiguousarray(im)

        t1 = time_sync()

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

            # pad img to make square-shape
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
        dt[0] += t2 - t1

        pred = model(im, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

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

        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            im0 = im0s.copy()
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
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

            # Print time (inference-only)
            LOGGER.info(f"Done. ({t3 - t2:.3f}s)")

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

        cv2.imshow("demo", im0)
        k = cv2.waitKey(1) & 0xFF


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default=ROOT / "weights/107.pt",
        help="model path(s)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=ROOT / "data/images",
        help="file/dir/URL/glob, 0 for webcam",
    )
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[640],
        help="inference size h,w",
    )
    parser.add_argument(
        "--compile-model", action="store_true", help="compile model (GPU ONLY)"
    )
    parser.add_argument(
        "--quantize-model", action="store_true", help="quantize model (CPU ONLY)"
    )
    parser.add_argument(
        "--normalize", action="store_true", help="apply normalizer or not"
    )
    parser.add_argument(
        "--print-string", action="store_true", help="apply normalizer or not"
    )
    parser.add_argument(
        "--rm-doubled-bboxes", action="store_true", help="use additional nms"
    )
    parser.add_argument("--gray", action="store_true", help="apply grayscale or not")
    parser.add_argument("--edge", action="store_true", help="apply canny edge or not")
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="NMS IoU threshold"
    )
    parser.add_argument(
        "--use-soft", action="store_true", help="use soft nms rather than normal nms"
    )
    parser.add_argument(
        "--roi-crop", action="store_true", help="crop input around the roi"
    )
    parser.add_argument(
        "--use-yolo", action="store_true", help="crop input around the roi"
    )
    parser.add_argument(
        "--max-det", type=int, default=1000, help="maximum detections per image"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument(
        "--save-crop", action="store_true", help="save cropped prediction boxes"
    )
    parser.add_argument(
        "--nosave", action="store_true", help="do not save images/videos"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --classes 0, or --classes 0 2 3",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument(
        "--project", default=ROOT / "runs/detect", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument(
        "--line-thickness", default=3, type=int, help="bounding box thickness (pixels)"
    )
    parser.add_argument(
        "--hide-labels", default=False, action="store_true", help="hide labels"
    )
    parser.add_argument(
        "--hide-conf", default=False, action="store_true", help="hide confidences"
    )
    parser.add_argument(
        "--half", action="store_true", help="use FP16 half-precision inference"
    )
    parser.add_argument(
        "--dnn", action="store_true", help="use OpenCV DNN for ONNX inference"
    )
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=("tensorboard", "thop"))
    run()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
