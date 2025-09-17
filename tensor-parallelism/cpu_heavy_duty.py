import torch
import time

# Function to perform a heavy computation on CPU
A = torch.randn(2000, 100000)
B = torch.randn(100000, 2000)

def heavy_computation_cpu():
    return A @ B

# Measure time taken for CPU computation
start_time = time.time()
result_cpu = heavy_computation_cpu()
end_time = time.time()
print(f"CPU computation time: {end_time - start_time:.4f} seconds")