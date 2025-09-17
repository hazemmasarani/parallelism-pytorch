import torch
import time

# choose a size big enough to stress CPU/GPU
N = 4000

# function to measure elapsed time
def benchmark(device):
    print(f"\nRunning on {device}...")
    a = torch.randn(N, N, device=device)
    b = torch.randn(N, N, device=device)

    # warm up (especially important for GPU)
    c = a @ b

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    c = a @ b
    if device.type == "cuda":
        torch.cuda.synchronize()  # wait for GPU to finish
    end = time.time()

    print(f"Time: {end - start:.4f} seconds")

# Run on CPU
benchmark(torch.device("cpu"))

# Run on GPU (if available)
if torch.cuda.is_available():
    benchmark(torch.device("cuda"))
else:
    print("\nCUDA not available on this machine.")
