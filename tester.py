import torch as torch

a = torch.randn(1, 2, 3)
b = torch.randn(1, 4, 3)

c = torch.cat((a,b), dim=1)
print(c.shape)
