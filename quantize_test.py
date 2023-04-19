import platform
import torch
from utils.quantization_utils import resnet152, resnet18
from utils.quantization_utils import static_quantizer, QuantModel

architecture = platform.uname().machine

if architecture == "AMD64":
    backend = "x86"
elif architecture == "aarch64":
    backend = "ARM64"
else:
    exit(1)

# backend = 'x86'
# qconfig = torch.quantization.get_default_qconfig(backend)
# torch.backends.quantized.engine = backend

# model = torch.load("yolov3.pt", map_location="cpu")["model"]
model = resnet18()
model = QuantModel(model)
model = static_quantizer(model, configs=backend).to("cpu").eval()

x = torch.randn((1, 3, 320, 320)).to("cpu")
# x = prepare(x)
# x = torch.quantization.QuantStub()(x)
x = model(x)
# x = torch.quantization.DeQuantStub()(x)
