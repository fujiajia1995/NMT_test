import numpy as np
import torch
from torch.nn import Embedding
from torch.nn import Module
import os
loss = torch.nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)


x = torch.randn(10, 5)
y = torch.randint(5, (10,))

print(input)
print(target)

torch.nn.CrossEntropyLoss()

print(x)
print(y)

a = torch.randn((2, 3, 4))
a = a[:, :, -2:]
print(a.size())
