import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def worker(rank, world_size):
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29500",
        world_size=world_size,
        rank=rank
    )

    # Each worker starts with a tensor containing just its rank
    my_tensor = torch.tensor([rank], device=f"cuda:0", dtype=torch.int64) # {rank % torch.cuda.device_count()}")

    # Allocate a list of tensors to gather into
    gather_list = [torch.zeros(1, device=my_tensor.device, dtype=torch.int64) for _ in range(world_size)]

    # All_gather: every process gets all ranks
    dist.all_gather(gather_list, my_tensor)

    print(f"Rank {rank} received ranks: {[t.item() for t in gather_list]}")

    dist.destroy_process_group()

def main():
    world_size = torch.cuda.device_count()
    world_size = max(world_size, 4)  # limit to 4 for this example
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
