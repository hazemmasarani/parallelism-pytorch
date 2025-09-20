import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def worker(rank, world_size):
    # 1. Initialize the process group
    dist.init_process_group(
        backend="nccl",         # for GPU
        init_method="tcp://127.0.0.1:29500",  # how processes find each other
        world_size=world_size,  # total number of processes
        rank=rank               # this process's ID
    )

    # 2. Each process reports in
    print(f"Hello from rank {rank} out of {world_size} processes")

    # 3. Cleanup when done
    dist.destroy_process_group()

def main():
    world_size = torch.cuda.device_count() or 4  # fallback to 2 if no GPUs
    world_size = max(world_size, 4)  # limit to 4 for this example
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
