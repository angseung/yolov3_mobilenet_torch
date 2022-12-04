import os
import platform
import torch
from models.yolo import Model

curr_os = platform.system()
print("Current OS : %s" % curr_os)

if "Windows" in curr_os:
    device = "cuda" if torch.cuda.is_available() else "cpu"
elif "Darwin" in curr_os:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
elif "Linux" in curr_os:
    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = "./models/yolov3-nano.yaml"
img_size = 320
n_classes = 62
model = Model(cfg=cfg, nc=n_classes, anchors=5).to(device)
model.eval()
dummy_input = torch.randn((1, 3, img_size, img_size)).to(device)
[dummy_bboxes, dummy_output] = model(dummy_input)
