import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def worker(rank, world_size, N=8):
    """
    rank: rank of this worker
    world_size: total number of workers
    N: matrix size (NxN)
    """
    dist.init_process_group(
        backend="gloo",  # can use nccl if on GPU
        init_method="tcp://127.0.0.1:29500",
        world_size=world_size,
        rank=rank
    )

    # Create the full matrix only on rank 0 for demonstration
    if rank == 0:
        full_matrix = torch.randn(N, N)
    else:
        full_matrix = torch.zeros(N, N)

    # Broadcast full_matrix to all workers
    dist.broadcast(full_matrix, src=0)

    # Split the matrix row-wise
    rows_per_worker = N // world_size
    start = rank * rows_per_worker
    end = (rank + 1) * rows_per_worker if rank != world_size - 1 else N
    local_rows = full_matrix[start:end, :]

    # Each worker computes its partial product asynchronously
    partial_result = torch.matmul(local_rows, full_matrix)

    # Prepare a list to gather partial results on all ranks
    gather_list = [torch.zeros_like(partial_result) for _ in range(world_size)]
    handle = dist.all_gather(gather_list, partial_result, async_op=True)

    # Simulate doing some other work
    # torch.manual_seed(rank)
    # _ = torch.randn(10, 10).mm(torch.randn(10, 10))

    # Wait for all_gather to complete
    handle.wait()

    # Combine the gathered partial results into the final result
    final_result = torch.cat(gather_list, dim=0)

    if rank == 0:
        print("Final matrix multiplication result A.dot(A):")
        print(final_result)

    dist.destroy_process_group()


def main():
    N = 8  # size of matrix
    world_size = 4  # number of processes
    mp.spawn(worker, args=(world_size, N), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
