import platform
from typing import *
import torch
from torch import nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck, _resnet
from torch.ao.quantization import quantize_dynamic
from torch.ao.nn.quantized import Conv2d as qConv2d
from models.resnet import resnet18 as ResNet18


class AddModel(nn.Module):
    def __init__(self, quantized=False):
        super().__init__()
        self.ff = torch.nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        out = self.ff.add(x, y)

        return out


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.ao.quantization.QuantStub()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        # self.bn = torch.nn.BatchNorm2d(1)
        self.relu = torch.nn.ReLU()
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.conv(x)
        # x = self.bn(x)
        x = self.relu(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x


class QuantBasicBlock(BasicBlock):
    # ff = torch.ao.nn.quantized.FloatFunctional()
    ff = torch.nn.quantized.FloatFunctional()
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
    ff = torch.nn.quantized.FloatFunctional()
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
    configs: Optional[Union[str, None]] = None,
    return_prepare: Optional[bool] = False,
) -> nn.Module:
    """
    https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
    """

    if isinstance(configs, str):
        model.qconfig = torch.ao.quantization.get_default_qconfig(configs)
    else:
        model.qconfig = torch.ao.quantization.default_qconfig

    model = model.to("cpu")
    # model_fp32_fused = torch.ao.quantization.fuse_modules(model, [['conv', 'relu']])
    prepare = torch.ao.quantization.prepare(model)
    quantized_model = torch.ao.quantization.convert(prepare)
    print("Post Training Quantization: Convert done")

    return quantized_model


class QuantModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.ao.quantization.QuantStub()

        self.model = model.to("cpu")

        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.ao.quantization.DeQuantStub()

    def fuse_modules(self, modules_to_fuse: List[Type[nn.Module]]) -> nn.Module:
        torch.ao.quantization.fuse_modules(self.model, modules_to_fuse=modules_to_fuse, inplace=True)

        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)

        return x


if __name__ == "__main__":
    # create a model instance
    architecture = platform.uname().machine
    modules_to_fuse = [nn.Conv2d, nn.BatchNorm2d, nn.ReLU]

    model_fp32 = ResNet18().eval()
    model_fp32 = torch.quantization.fuse_modules(
        model_fp32, [["conv1", "bn1", "relu"]], inplace=True
    )

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

    model_qint8 = QuantModel(model=model_fp32)
    model_qint8.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    # model_fp32_fused = torch.ao.quantization.fuse_modules(model, [['conv', 'relu']])
    # the model that will observe activation tensors during calibration.
    torch.ao.quantization.prepare(model_qint8, inplace=True)
    input_fp32 = torch.randn(1, 3, 224, 224)

    # it calibrates model
    model_qint8(input_fp32)

    model_int8 = torch.ao.quantization.convert(model_qint8)

    # run the model, relevant calculations will happen in int8
    res = model_int8(input_fp32)

    # model_fp32 = resnet18().to("cpu").eval()
    # model_fp32 = AddModel().to("cpu").eval()
    # model = QuantModel(model=model_fp32)
    # model = static_quantizer(model=model, configs="x86")
    # dummy_input = torch.randn(1, 3, 224, 224)
    # dummy_output = model(dummy_input)
