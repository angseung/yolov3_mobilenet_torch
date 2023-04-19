import torch
from torchvision.models import resnet152
from utils.quantization_utils import static_quantizer, QuantModel

# backend = 'x86'
# qconfig = torch.quantization.get_default_qconfig(backend)
# torch.backends.quantized.engine = backend

# model = torch.load("yolov3.pt", map_location="cpu")["model"]
model = resnet152()
model = QuantModel(model)
model = static_quantizer(model, configs="x86").to("cpu").eval()

x = torch.randn((1, 3, 320, 320)).to("cpu")
# x = prepare(x)
# x = torch.quantization.QuantStub()(x)
x = model(x)
# x = torch.quantization.DeQuantStub()(x)
