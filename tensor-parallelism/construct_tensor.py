import torch


# Create a tensor on CPU
tensor_cpu = torch.randn(3, 3)
print("Tensor on CPU:", tensor_cpu)

# Create a tensor on GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor_gpu = torch.randn(3, 3, device=device)
print("Tensor on GPU:", tensor_gpu.device)
print("Tensor on CPU:", tensor_cpu.device)
# Move tensor from CPU to GPU
tensor_cpu_to_gpu = tensor_cpu.to(device)

# Move tensor from GPU to CPU
tensor_gpu_to_cpu = tensor_gpu.to("cpu")