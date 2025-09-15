import torch

A = torch.randn(10000, 10000, device="cuda:0")
B = torch.randn(10000, 10000, device="cuda:0")
C = torch.randn(10000, 10000, device="cuda:0")
D = torch.randn(10000, 10000, device="cuda:0")

stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

with torch.cuda.stream(stream1):
    result1 = A @ B

with torch.cuda.stream(stream2):
    result2 = C @ D

# Wait for both to finish
torch.cuda.synchronize()
