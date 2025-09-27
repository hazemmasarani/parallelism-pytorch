import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def worker(rank, world_size, return_dict):
    # Initialize process group
    dist.init_process_group(
        backend="gloo",  # "nccl",
        init_method="tcp://127.0.0.1:29500",
        world_size=world_size,
        rank=rank
    )

    # Each rank computes its own A @ A
    A = torch.randn(100, 100, device=f"cuda:{rank}")
    A = torch.matmul(A, A)

    # All-reduce to compute the sum across ranks
    dist.all_reduce(A, op=dist.ReduceOp.SUM)
    # Divide to get the average
    A /= world_size

    # Only rank 0 returns the result
    if rank == 0:
        return_dict["A"] = A.cpu()

    dist.destroy_process_group()


def main():
    world_size = dist.get_world_size()
    manager = mp.Manager()
    return_dict = manager.dict()

    mp.spawn(worker, args=(world_size, return_dict), nprocs=world_size, join=True)

    if "A" in return_dict:
        A_master = return_dict["A"]
        print("Final averaged result on rank 0:", A_master.shape)
        return A_master


if __name__ == "__main__":
    A = main()