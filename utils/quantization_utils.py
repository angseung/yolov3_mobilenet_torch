from typing import *
import torch
from torch import nn as nn
from torch.ao.quantization import quantize_dynamic
from torch.ao.nn.quantized import Conv2d as qConv2d

# define a floating point model
class M(nn.Module):
    def __init__(self, quantized=False):
        super().__init__()
        if quantized:
            self.conv1 = qConv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)  # 224
            self.conv2 = qConv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)  # 112
            self.conv3 = qConv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)  # 56
            self.conv4 = qConv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)  # 28
            self.conv5 = qConv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)  # 14
            self.conv6 = qConv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False)  # 7

        else:
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)  # 224
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)  # 112
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)  # 56
            self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)  # 28
            self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)  # 14
            self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False)  # 7

        self.avgpool = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.avgpool(x)
        x = self.fc(torch.flatten(x, start_dim=1))

        return x


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
        model: nn.Module, configs: Optional = None
) -> nn.Module:

    if configs is not None:
        raise NotImplementedError
    else:
        model.qconfig = torch.ao.quantization.default_qconfig

    torch.ao.quantization.prepare(model, inplace=True)
    torch.ao.quantization.convert(model, inplace=True)
    print('Post Training Quantization: Convert done')

    return model


if __name__ == "__main__":
    # create a model instance
    model_fp32 = M(quantized=False).to("cpu").eval()
    model_int8 = dynamic_quantizer(
        model=model_fp32, layers=[nn.Linear, nn.Conv2d, nn.AvgPool2d], dtype=torch.qint8
    )
    model_fp32.qconfig = torch.ao.quantization.default_qconfig
    print(model_fp32.qconfig)
    torch.ao.quantization.prepare(model_fp32, inplace=True)

    # Calibrate first
    print('Post Training Quantization Prepare: Inserting Observers')
    print('\n Inverted Residual Block:After observer insertion \n\n', model_fp32.conv1)

    # Calibrate with the training set
    print('Post Training Quantization: Calibration done')

    # Convert to quantized model
    torch.ao.quantization.convert(model_fp32, inplace=True)
    print('Post Training Quantization: Convert done')
    print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',model_fp32.conv1)

    # run the model
    input_fp32 = torch.randn(1, 3, 224, 224)
    res = model_int8(input_fp32)
