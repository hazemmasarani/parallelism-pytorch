import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def worker(rank, world_size):
    # Initialize global process group
    dist.init_process_group(
        backend="gloo",  # use gloo for multi-proc on single GPU
        init_method="tcp://127.0.0.1:29500",
        world_size=world_size,
        rank=rank
    )

    # Define a subgroup of ranks, e.g., ranks 1,2,3
    subgroup_ranks = [1, 2, 3]
    if rank in subgroup_ranks:
        group = dist.new_group(ranks=subgroup_ranks)

        # Each rank in the subgroup contributes a tensor
        tensor = torch.tensor([rank])
        gather_list = [torch.zeros(1, dtype=torch.int64) for _ in subgroup_ranks]

        # Perform all_gather within the subgroup
        dist.all_gather(gather_list, tensor, group=group)

        print(f"Rank {rank} subgroup gather: {[t.item() for t in gather_list]}")

    else:
        print(f"Rank {rank} not in subgroup, skipping subgroup communication.")

    # Cleanup
    dist.destroy_process_group()

def main():
    world_size = 4  # total number of processes
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
