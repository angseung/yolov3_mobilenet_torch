import time
import platform
import torch

try:
    from torchinfo import summary

    TORCH_INFO_OPT = True
except ImportError or ModuleNotFoundError:
    TORCH_INFO_OPT = False
from models.yolo import Model
from utils.general import convert_model_to_onnx

curr_os = platform.system()
print("Current OS : %s" % curr_os)

MODEL_EXPORT_OPT = False

if "Windows" in curr_os:
    device = "cuda" if torch.cuda.is_available() else "cpu"
elif "Darwin" in curr_os:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
elif "Linux" in curr_os:
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = "cpu"


cfg = "./models/yolov3-nano.yaml"
img_size: int = 320
n_classes: int = 62  # default value is 62
anchors: int = 3  # default value is 3
output_channels = anchors * (
    (img_size // 8) ** 2 + (img_size // 16) ** 2 + (img_size // 32) ** 2
)
model = Model(cfg=cfg, nc=n_classes, anchors=anchors).to(device)
model.eval()
dummy_input = torch.randn((1, 3, img_size, img_size), requires_grad=True).to(device)
[dummy_bboxes, dummy_output] = model(dummy_input)

# check model output bbox shape
assert dummy_bboxes.shape[1] == output_channels

if MODEL_EXPORT_OPT:
    convert_model_to_onnx(
        model.to("cpu"),
        input_size=(1, 3, img_size, img_size),
        onnx_name="yolov3-nano.onnx",
    )

if TORCH_INFO_OPT:
    print(summary(model, (1, 3, img_size, img_size)))

# for check elapsed time test on raspberry pi
RASPI_CALC_OPT = False
if RASPI_CALC_OPT:
    a = torch.randn((1, 3, img_size, img_size))

    total_time = 0
    for _ in range(50):
        start_time = time.time()
        [dummy_bboxes, dummy_output] = model(a)
        end_time = time.time()
        total_time += end_time - start_time
    print(total_time / 50)
