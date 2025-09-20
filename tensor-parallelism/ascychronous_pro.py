import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import random

def worker(rank, world_size):
    # Initialize the default process group
    dist.init_process_group(
        backend="gloo",  # use gloo for CPU async testing
        init_method="tcp://127.0.0.1:29500",
        world_size=world_size,
        rank=rank
    )

    # Simulate a model parameter
    param = torch.tensor([1.0], dtype=torch.float32)

    # Simulate local gradient computation
    # We'll make it slightly different per rank
    torch.manual_seed(rank)
    local_grad = torch.randn(1) * (rank + 1)
    print(f"Rank {rank} local gradient: {local_grad.item():.4f}")

    # Perform async all_reduce to sum gradients across ranks
    handle = dist.all_reduce(local_grad, op=dist.ReduceOp.SUM, async_op=True)

    # Do some "other work" while async communication happens
    time.sleep(random.uniform(0.1, 0.5))  # simulate computation

    # Wait for all_reduce to finish
    handle.wait()

    # Simulate gradient update (e.g., SGD step)
    lr = 0.1
    param -= lr * local_grad
    print(f"Rank {rank} updated parameter: {param.item():.4f}")

    dist.destroy_process_group()


def main():
    world_size = 4  # number of processes
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
