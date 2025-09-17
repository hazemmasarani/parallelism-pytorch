import torch
import time


# Function to perform a heavy computation on GPU
A = torch.randn(2000, 100000, device="cuda")
B = torch.randn(100000, 2000, device="cuda")

def heavy_computation_gpu():
    return A @ B

# Measure time taken for GPU computation
start_time = time.time()
result_gpu = heavy_computation_gpu()
torch.cuda.synchronize()  # Wait for all kernels to finish
end_time = time.time()
print(f"GPU computation time: {end_time - start_time:.4f} seconds")
