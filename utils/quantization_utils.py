import os.path
import platform
import copy
from typing import *
from pathlib import Path
import torch
from torch import nn as nn
from torch.ao.quantization import fuse_modules
from torchvision.models.resnet import BasicBlock, Bottleneck, _resnet
from torch.ao.quantization import quantize_dynamic
from torch.ao.nn.quantized import Conv2d as qConv2d
from models.resnet import resnet18 as ResNet18
from models.resnet import resnet152 as ResNet152
from models.common import ConvBnReLU, BottleneckReLU

FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent  # root directory


class AddModel(nn.Module):
    def __init__(self, quantized=False):
        super().__init__()
        self.ff = torch.ao.nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        out = self.ff.add(x, y)

        return out


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            3, 32, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")  # Success
        # self.upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")  # Success
        self.bn = torch.nn.BatchNorm2d(32)
        self.concat = torch.cat  # Success
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        x = self.bn(x)
        x1 = x.clone()
        x = self.relu(x)
        x = self.concat([x, x1], dim=1)
        x = self.sigmoid(x)
        return x


class QuantBasicBlock(BasicBlock):
    # ff = torch.ao.nn.quantized.FloatFunctional()
    ff = torch.ao.nn.quantized.FloatFunctional()
    relu1 = nn.ReLU()
    relu2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.ff.add(out, identity)
        out = self.relu2(out)

        return out


class QuantBottleNeck(Bottleneck):
    # ff = torch.ao.nn.quantized.FloatFunctional()
    ff = torch.ao.nn.quantized.FloatFunctional()
    relu1 = nn.ReLU()
    relu2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.ff.add(out, identity)
        out = self.relu2(out)

        return out


def resnet18():
    return _resnet(QuantBasicBlock, [2, 2, 2, 2], weights=None, progress=False)


def resnet152():
    return _resnet(QuantBottleNeck, [3, 8, 36, 3], weights=None, progress=False)


def dynamic_quantizer(
    model: nn.Module, layers: List[type], dtype: torch.dtype
) -> nn.Module:
    model_quant = quantize_dynamic(
        model,
        {*layers},
        dtype=dtype,
    )

    return model_quant


def static_quantizer(
    model: nn.Module,
    data_to_calibrate: torch.Tensor,
    layers_to_fuse: List[List[Union[str, Type[nn.Module]]]],
    configs: Optional[Union[str, None]] = None,
) -> nn.Module:
    """
    https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
    """

    if isinstance(configs, str):
        model.qconfig = torch.ao.quantization.get_default_qconfig(configs)
    else:
        model.qconfig = torch.ao.quantization.default_qconfig

    model = model.to("cpu")
    torch.ao.quantization.fuse_modules(model, layers_to_fuse, inplace=True)
    prepare = torch.ao.quantization.prepare(model)
    prepare(data_to_calibrate)  # calibrates model
    quantized_model = torch.ao.quantization.convert(prepare)
    print("Post Training Quantization: Convert done")

    return quantized_model


def fuse_resnet(model_fp32: nn.Module) -> None:
    for module_name, module in model_fp32.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(
                    basic_block,
                    [["conv1", "bn1", "relu1"], ["conv2", "bn2"]],
                    inplace=True,
                )
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        torch.quantization.fuse_modules(
                            sub_block, [["0", "1"]], inplace=True
                        )


class QuantModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.ao.quantization.QuantStub()

        self.model = model.to("cpu")

        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)

        return x


class QuantizedYolo(nn.Module):
    def __init__(self, fname: str = "yolov3-qat.pt"):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.model = torch.load(os.path.join(ROOT, fname), map_location=torch.device("cpu")).eval()  # nn.Sequential
        self.dequant = torch.ao.quantization.DeQuantStub()

    def fuse_model(self):
        for i, block in self.model.model.named_children():
            if isinstance(block, ConvBnReLU):
                fuse_modules(block, [['conv', 'bn', 'act']], inplace=True)

            elif isinstance(block, BottleneckReLU):
                for j, sub_block in block.named_children():
                    if isinstance(sub_block, ConvBnReLU):
                        fuse_modules(sub_block, [['conv', 'bn', 'act']], inplace=True)

            # TODO: Implement fusing codes for other blocks here...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)

        return x


if __name__ == "__main__":
    # create a model instance
    architecture = platform.uname().machine

    yolo_qint8 = QuantizedYolo()
    yolo_qint8.fuse_model()
    torch.ao.quantization.prepare(yolo_qint8, inplace=True)
    yolo_qint8(torch.randn(1, 3, 320, 320))
    torch.ao.quantization.convert(yolo_qint8, inplace=True)

    # model_fp32 = ResNet18().eval()
    model_fp32 = M().eval()
    model_ori = copy.deepcopy(model_fp32)

    # fuse BasicBlock for ResNet18 ONLY
    if "ResNet" in model_fp32.__class__.__name__:
        fuse_resnet(model_fp32)

    else:
        torch.quantization.fuse_modules(
            model_fp32, [["conv", "bn", "relu"]], inplace=True
        )

    model_qint8 = QuantModel(model=model_fp32)
    model_qint8.qconfig = torch.ao.quantization.get_default_qconfig("x86")

    # the model that will observe activation tensors during calibration.
    torch.ao.quantization.prepare(model_qint8, inplace=True)
    input_fp32 = torch.randn(1, 3, 224, 224)

    # it calibrates model
    model_qint8(input_fp32)

    # quantize model
    model_int8 = torch.ao.quantization.convert(model_qint8)

    # run the model, relevant calculations will happen in int8
    res = model_int8(input_fp32)
    res_fp = model_ori(input_fp32)

    # onnx export test
    torch.onnx.export(
        model_ori,
        input_fp32,
        "../model_ori.onnx",
        opset_version=17,
    )

    torch.onnx.export(
        model_int8,
        input_fp32,
        "../model_int8.onnx",
        opset_version=17,  # offset <= 11 occurs error
    )
