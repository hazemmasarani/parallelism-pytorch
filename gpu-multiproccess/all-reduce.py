import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def worker(rank, world_size, return_dict):
    print(f"Rank {rank} starting...")
    # Initialize process group
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:29500",
        world_size=world_size,
        rank=rank
    )

    # Define tensor on all ranks
    A = torch.empty(5, 5, device=f"cuda:{rank}")
    if rank == 0:
        A = torch.randn(5, 5, device=f"cuda:0")
        return_dict["A_orig"] = A.clone().cpu()
        print(A)
    
    dist.broadcast(A, src=0)
    
    # Each rank computes its own A @ A
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
    world_size = torch.cuda.device_count() # Number of available GPUs
    manager = mp.Manager()
    return_dict = manager.dict()

    mp.spawn(worker, args=(world_size, return_dict), nprocs=world_size, join=True)

    if "A" in return_dict:
        A_master = return_dict["A"]
        A_orig = return_dict["A_orig"]
        print("A_orig from rank 0:\n", A_orig)
        validate = torch.matmul(A_orig, A_orig)
        if torch.allclose(A_master, validate):
            print("Validation successful: A_master matches A_orig @ A_orig")
        else:
            print("Validation failed: A_master does not match A_orig @ A_orig")
        print("Validation result:\n", validate)
        print("Final averaged result on rank 0:\n", A_master)
        
        max_error = torch.max(torch.abs(A_master - validate))
        print("Maximum absolute error:", max_error.item())

if __name__ == "__main__":
    main()