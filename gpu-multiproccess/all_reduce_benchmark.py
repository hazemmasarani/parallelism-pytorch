import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from time import time

def worker(rank, world_size, return_dict, typ = torch.float32, N=5):
    # Initialize process group
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:29500",
        world_size=world_size,
        rank=rank
    )

    # Define tensor on all ranks
    A = torch.empty(N, N, device=f"cuda:{rank}", dtype=typ)
    if rank == 0:
        A = torch.randn(N, N, device=f"cuda:0", dtype=typ)
    
    dist.broadcast(A, src=0)
    
    # Each rank computes its own A @ A
    A = torch.matmul(A, A)

    # All-reduce to compute the sum across ranks
    start = time()
    dist.all_reduce(A, op=dist.ReduceOp.SUM)
    end = time()
    print(f"Rank {rank} all-reduce time: {end - start:.6f} seconds")
    # Divide to get the average
    A /= world_size

    # Only rank 0 returns the result
    if rank == 0:
        return_dict["A"] = A.cpu()

    dist.destroy_process_group()


def main(typ = torch.float32):
    world_size = torch.cuda.device_count() # Number of available GPUs
    manager = mp.Manager()
    return_dict = manager.dict()

    mp.spawn(worker, args=(world_size, return_dict, typ, 10000), nprocs=world_size, join=True)

    if "A" in return_dict:
        A_master = return_dict["A"]
        
        # max_error = torch.max(torch.abs(A_master - validate))
        # print("Maximum absolute error:", max_error.item())

if __name__ == "__main__":
    for typ in [torch.float64, torch.float32, torch.float16, torch.bfloat16]:
        print(f"\nRunning benchmark with data type: {typ}")
        main(typ)