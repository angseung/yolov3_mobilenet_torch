import os.path
import platform
import copy
from typing import *
from pathlib import Path
import cv2
import torch
from torch import nn as nn
from torch.ao.quantization import fuse_modules
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.resnet import BasicBlock, Bottleneck, _resnet
from torch.ao.quantization import quantize_dynamic
from models.common import ConvBnReLU, BottleneckReLU, Concat, C3ReLU, SPPFReLU
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import normalizer
from utils.roi_utils import resize
from utils.augmentations import wrap_letterbox

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
    def __init__(self, model: Union[str, nn.Module] = None, yolo_version: int = 3):
        super().__init__()
        self.yolo_version = yolo_version

        if isinstance(model, str):
            if model.endswith(".pt"):
                self.model = torch.load(
                    os.path.join(ROOT, model), map_location=torch.device("cpu")
                )
                if isinstance(self.model, dict):
                    self.model = self.model["model"].float()
            elif model.endswith(".yaml"):
                self.model = DetectMultiBackend(
                    os.path.join(ROOT, model), torch.device("cpu"), False
                )
        elif isinstance(model, nn.Module):
            self.model = model

        elif isinstance(model, DetectMultiBackend):
            self.model = model.model

        else:
            raise AttributeError("Unsupported model type")

        self.quant = torch.ao.quantization.QuantStub()
        self.model.model[-1] = nn.Identity()
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

            elif isinstance(block, C3ReLU):
                for j, sub_block in block.named_children():
                    if isinstance(sub_block, ConvBnReLU):
                        fuse_modules(sub_block, [["conv", "bn", "act"]], inplace=True)
                    elif isinstance(sub_block, BottleneckReLU):
                        for k, sub_sub_block in sub_block:
                            fuse_modules(sub_sub_block, [["conv", "bn", "act"]], inplace=True)

            elif isinstance(block, SPPFReLU):
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

    def _forward_v3(self, x: torch.Tensor) -> List[torch.Tensor]:
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

    def _forward_v5(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.quant(x)

        for i, block in self.model.model.named_children():
            if isinstance(block, Concat):
                if i == "12":
                    x = block([x6, x11])
                elif i == "16":
                    x = block([x4, x15])
                elif i == "19":
                    x = block([x14, x18])
                elif i == "22":
                    x = block([x10, x21])
            else:  # ConvBnReLU, BottleneckReLU, C3ReLU, SPPFReLU
                x = block(x)

                # save feature map for concat/conv layers...
                if i == "4":
                    x4 = x.clone()
                elif i == "6":
                    x6 = x.clone()
                elif i == "10":
                    x10 = x.clone()
                elif i == "11":
                    x11 = x.clone()
                elif i == "14":
                    x14 = x.clone()
                elif i == "15":
                    x15 = x.clone()
                elif i == "17":
                    x17 = x.clone()
                elif i == "18":
                    x18 = x.clone()
                elif i == "20":
                    x20 = x.clone()
                elif i == "21":
                    x21 = x.clone()
                elif i == "23":
                    x23 = x.clone()

        x17 = self.dequant(x17)
        x20 = self.dequant(x20)
        x23 = self.dequant(x23)

        return [x17, x20, x23]

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        if self.yolo_version == 3:
            return self._forward_v3(x)

        elif self.yolo_version == 5:
            return self._forward_v5(x)


class QuantizedYoloHead(nn.Module):
    def __init__(self, model: Union[str, nn.Module] = None):
        super().__init__()
        if isinstance(model, str):
            if model.endswith(".pt"):
                yolo_model = torch.load(
                    os.path.join(ROOT, model), map_location=torch.device("cpu")
                )
                if isinstance(yolo_model, dict):
                    yolo_model = yolo_model["model"].float()
            elif model.endswith(".yaml"):
                yolo_model = DetectMultiBackend(
                    os.path.join(model), torch.device("cpu"), False
                )
        elif isinstance(model, nn.Module):
            yolo_model = model

        elif isinstance(model, DetectMultiBackend):
            yolo_model = model.model

        else:
            raise AttributeError("Unsupported model type")

        self.model = copy.deepcopy(yolo_model.model[-1])
        self.model = self.model.eval()

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.model(x)


class CalibrationDataLoader(Dataset):
    def __init__(self, img_dir: str, target_size: int = 320):
        super().__init__()
        self.transform = transforms.Compose([transforms.ToTensor(), normalizer()])
        self.target_size = target_size
        self.img_dir = img_dir
        self.img_list = os.listdir(img_dir)

    def __getitem__(self, item):
        img = cv2.imread(f"{self.img_dir}/{self.img_list[item]}")  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
        img = resize(img, self.target_size)
        img = wrap_letterbox(img, self.target_size)[0]  # padded as square shape

        return self.transform(img)

    def __len__(self):
        return len(self.img_list)


if __name__ == "__main__":
    # create a model instance
    architecture = platform.uname().machine
    dataset = CalibrationDataLoader(os.path.join(ROOT, "data", "cropped"))
    calibration_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    input = torch.randn(1, 3, 320, 320)
    fname: str = os.path.join("weights", "yolov5m-qat.pt")
    yolo_detector = QuantizedYoloHead(fname)
    yolo_fp32 = QuantizedYoloBackbone(fname, yolo_version=5)
    yolo_qint8 = QuantizedYoloBackbone(fname, yolo_version=5)
    yolo_qint8.fuse_model()
    yolo_qint8.check_fused_layers()
    yolo_qint8.qconfig = torch.ao.quantization.get_default_qconfig("x86")
    torch.ao.quantization.prepare(yolo_qint8, inplace=True)

    for i, img in enumerate(calibration_dataloader):
        print(f"\rcalibrating... {i + 1} / {dataset.__len__()}", end="")
        yolo_qint8(img)

    torch.ao.quantization.convert(yolo_qint8, inplace=True)
    dummy_output = yolo_qint8(input)
    pred = yolo_detector(dummy_output)[0]

    pred_fp32 = yolo_detector(yolo_fp32(input))[0]

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
        opset_version=11,
    )
