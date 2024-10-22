from ultralytics.nn.modules.block import SPIUpResolution
import torch

A = torch.rand(1, 3, 128, 128)

model = SPIUpResolution(3,3)
B = model(A)

print(B.shape)