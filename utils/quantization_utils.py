from typing import *
import torch
from torch import nn as nn
from torch.ao.quantization import quantize_dynamic


# define a floating point model
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)

    def forward(self, x):
        x = self.fc(x)
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


if __name__ == "__main__":
    # create a model instance
    model_fp32 = M()
    model_int8 = dynamic_quantizer(
        model=model_fp32, layers=[nn.Linear], dtype=torch.qint8
    )

    # run the model
    input_fp32 = torch.randn(4, 4, 4, 4)
    res = model_int8(input_fp32)
