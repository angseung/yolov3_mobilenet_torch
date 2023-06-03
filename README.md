## <div align="center">Quick Start Guide</div>

<details open>
<summary>Install</summary>

**Python>=3.9.2** is required with all
requirements.txt installed including
**PyTorch>=2.0.0**:

```bash
$ git clone https://gitlab.surromind.ai/license-plate-recognition/yolov3_mobilenet_torch
$ cd yolov3_mobilenet_torch
$ pip install -r requirements.txt
```

</details>

<details open>
<summary>Run Train</summary>


```bash
$ python train.py --weights "" --cfg yolov5l-qat.yaml --data yperv2.yaml --normalize --imgsz 640 --setseed --seednum 123 --epochs 100 --batch-size 16
```

</details>

<details open>
<summary>Run Detect</summary>


```bash
$ python detect.py --weights runs/train/Case_107/weights/best.pt --imgsz 640 --roi-crop --print-string --normalize --source data/regions --rm-doubled-bboxes --iou-thres 0.05 --conf-thres 0.25 --quantize-model --use-yolo
```

</details>

