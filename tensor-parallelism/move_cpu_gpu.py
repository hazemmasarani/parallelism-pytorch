import torch

# create from Python data (defaults: copy, requires_grad=False)
a = torch.tensor([1.0, 2.0, 3.0])         # dtype inferred (float32)
b = torch.randn(2, 3)                     # random normal tensor
c = torch.zeros(4, dtype=torch.float64)   # specify dtype

torch.cuda.is_available()   # returns True/False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(3, 3, device=device)      # create directly on device
y = torch.randn(3, 3).to(device)          # or move an existing tensor
z = x + y                                 # both must be on same device

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2
z = y.sum()
z.backward()            # computes dz/dx; x.grad will be [2., 2., 2.]
print(x.grad)
