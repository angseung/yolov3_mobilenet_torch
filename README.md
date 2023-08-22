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
- Arguments
    - `weights` : 학습한 모델의 pt 파일 포인터
    - `cfg` : Scratch 하는 경우, 모델 구조가 정의된 yaml 파일
    - `data` : Dataset의 디렉토리 및 클래스 구조가 정의된 yaml 파일
    - `imgsz` : 학습 시의 이미지 해상도
    - `normalize` : 이미지에 ImageNet 통계 데이터를 활용한 정규화 적용
    - `qat` : QAT로 모델 학습
    - `multi-scale` : 학습 이미지 해상도를 batch마다 바꾸어 multi-scale로 학습하도록 하는 모드
    - `adam` : Optimizer에 SGD 대신 Adam을 적용
    - `noautoanchor` : Anchor Size를 조절하는 과정을 생략
</details>

<details open>
<summary>Run Detect</summary>


```bash
$ python detect.py --weights runs/train/Case_107/weights/best.pt --imgsz 640 --roi-crop --print-string --normalize --source data/regions --rm-doubled-bboxes --iou-thres 0.05 --conf-thres 0.25 --quantize-model --use-yolo
```
- Arguments
    - `weights` : 학습한 모델의 pt 파일 포인터
    - `source` : 추론할 데이터가 있는 디렉토리 경로
    - `imgsz` : 추론할 때의 입력 이미지 해상도
    - `cropped-imgsz` : ROI Crop을 적용하는 경우 Crop한 이미지의 해상도, `roi-crop` 이 True일 때만 유효
    - `compile-model` : GPU를 사용하여 추론하는 경우, 모델 compile 수행
    - `quantize-model` : CPU를 사용하는 경우, 모델에 PTQ를 적용하고 추론
    - `normalize` : 이미지에 ImageNet 통계 데이터를 활용한 정규화 적용
    - `print-string` : 추론 결과에 번호판 인식 결과를 문자열로 출력
    - `rm-doubled-bboxes` : 번호판의 글자가 겹쳐진 경우, 겹쳐진 BBOX를 제거하기 위해 사용
    - `conf-thres` : Confident Threshold
    - `iou-thres` : IoU Threshold
    - `use-soft` : NMS 과정에 Soft-NMS를 적용
    - `roi-crop` : Yolo 모델 추론 이전에 Edge Detection 기반 ROI Crop 과정 수행
    - `use-yolo` : Yolo 모델 추론 이전에 Yolo 기반 ROI Crop 과정 수행 (Recommended)
    - `half` : GPU로 추론하는 경우, float16 자료형으로 under casting 후 추론 수행
</details>

