import torch
import torch.multiprocessing as mp

# worker function
def worker(rank, nprocs):
    """
    rank: process index (0, 1, ...)
    nprocs: total number of processes we launched
    """
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(f"Process {rank} using device {device}")

    # each process creates a tensor on "its" device
    x = torch.randn(2, 2, device=device)
    print(f"Rank {rank} tensor:\n", x)

def main():
    nprocs = torch.cuda.device_count()  # number of GPUs
    if nprocs == 0:
        print("No CUDA devices available, running on CPU only.")
        nprocs = 1
    mp.spawn(worker, args=(nprocs,), nprocs=nprocs, join=True)

if __name__ == "__main__":
    main()
