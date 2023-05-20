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
from models.common import ConvBnReLU, BottleneckReLU, Concat
from models.common import DetectMultiBackend
from utils.general import non_max_suppression

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


class QuantizedYoloBackbone(nn.Module):
    def __init__(self, model: Union[str, nn.Module] = None):
        super().__init__()
        if isinstance(model, str):
            if model.endswith(".pt"):
                self.model = torch.load(
                    os.path.join(ROOT, model), map_location=torch.device("cpu")
                )
            elif model.endswith(".yaml"):
                self.model = DetectMultiBackend(
                    os.path.join(ROOT, "models", model), torch.device("cpu"), False
                )
        elif isinstance(model, nn.Module):
            self.model = model

        self.quant = torch.ao.quantization.QuantStub()
        self.model.model[28] = nn.Identity()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.model = self.model.eval()

    def fuse_model(self):
        for i, block in self.model.model.named_children():
            if isinstance(block, ConvBnReLU):
                fuse_modules(block, [["conv", "bn", "act"]], inplace=True)

            elif isinstance(block, BottleneckReLU):
                for j, sub_block in block.named_children():
                    if isinstance(sub_block, ConvBnReLU):
                        fuse_modules(sub_block, [["conv", "bn", "act"]], inplace=True)
            # TODO: Implement fusing codes for other blocks here...

    def check_fused_layers(self):
        for i, block in self.model.model.named_children():
            if isinstance(block, ConvBnReLU):
                print(block)

            elif isinstance(block, BottleneckReLU):
                for j, sub_block in block.named_children():
                    if isinstance(sub_block, ConvBnReLU):
                        print(sub_block)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.quant(x)
        for i, block in self.model.model.named_children():
            if isinstance(block, Concat):
                if i == "18":
                    x = block([x8, x17])
                elif i == "25":
                    x = block([x6, x24])
            else:  # ConvBnReLU, BottleneckReLU
                if i == "16":
                    x = block(x14)
                elif i == "23":
                    x = block(x21)
                else:
                    x = block(x)

                # save feature map for concat/conv layers...
                if i == "6":
                    x6 = x.clone()
                elif i == "8":
                    x8 = x.clone()
                elif i == "14":
                    x14 = x.clone()
                elif i == "15":
                    x15 = x.clone()
                elif i == "17":
                    x17 = x.clone()
                elif i == "21":
                    x21 = x.clone()
                elif i == "22":
                    x22 = x.clone()
                elif i == "24":
                    x24 = x.clone()
                elif i == "27":
                    x27 = x.clone()

        x15 = self.dequant(x15)
        x22 = self.dequant(x22)
        x27 = self.dequant(x27)

        return [x27, x22, x15]


class QuantizedYoloHead(nn.Module):
    def __init__(self, model: Union[str, nn.Module] = None):
        super().__init__()
        if isinstance(model, str):
            if model.endswith(".pt"):
                yolo_model = torch.load(
                    os.path.join(ROOT, model), map_location=torch.device("cpu")
                )
            elif model.endswith(".yaml"):
                yolo_model = DetectMultiBackend(
                    os.path.join(ROOT, "models", model), torch.device("cpu"), False
                )
        elif isinstance(model, nn.Module):
            yolo_model = model

        self.model = yolo_model.model[28]
        self.model = self.model.eval()

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.model(x)


if __name__ == "__main__":
    # create a model instance
    architecture = platform.uname().machine
    input = torch.randn(1, 3, 320, 320)
    fname: str = "yolov3-qat.pt"
    model = torch.load(
        os.path.join(ROOT, fname), map_location=torch.device("cpu")
    ).eval()
    model_head = torch.load(
        os.path.join(ROOT, fname), map_location=torch.device("cpu")
    ).eval()
    fp_output = model(input)[0]
    yolo_qint8 = QuantizedYoloBackbone(model)
    yolo_qint8.fuse_model()
    yolo_qint8.check_fused_layers()
    yolo_qint8.qconfig = torch.ao.quantization.get_default_qconfig("x86")
    torch.ao.quantization.prepare(yolo_qint8, inplace=True)
    yolo_qint8(input)
    torch.ao.quantization.convert(yolo_qint8, inplace=True)
    dummy_output = yolo_qint8(input)
    yolo_detector = QuantizedYoloHead(model_head)
    pred = yolo_detector(dummy_output)[0]

    pred_fp = non_max_suppression(
        fp_output,
        0.1,
        0.25,
    )

    pred_qint = non_max_suppression(
        pred,
        0.1,
        0.25,
    )

    # onnx export test
    # failed.
    # torch.onnx.errors.UnsupportedOperatorError: Exporting the operator 'quantized::batch_norm2d' to ONNX opset version 17 is not supported.
    torch.onnx.export(
        yolo_qint8,
        input,
        "../yolov3_backbone_qint8.onnx",
        opset_version=17,
    )
