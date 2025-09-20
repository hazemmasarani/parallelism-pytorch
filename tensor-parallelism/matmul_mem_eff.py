import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import math

def worker(rank, world_size, N, block_size):
    device = torch.device("cuda:0")
    dist.init_process_group(
        backend="gloo",  # can use nccl if on GPU
        init_method="tcp://127.0.0.1:29500",
        world_size=world_size,
        rank=rank
    )

    # Determine row slice for this rank
    rows_per_proc = math.ceil(N / world_size)
    row_start = rank * rows_per_proc
    row_end = min(row_start + rows_per_proc, N)
    local_rows_count = row_end - row_start

    # Rank 0 creates full matrix and scatters row blocks
    if rank == 0:
        A_full = torch.randn(N, N, device=device)
        row_slices = [A_full[i*rows_per_proc : min((i+1)*rows_per_proc, N), :] for i in range(world_size)]
    else:
        row_slices = [torch.zeros(1, N, device=device) for _ in range(world_size)]

    local_rows = torch.zeros(local_rows_count, N, device=device)
    dist.scatter(local_rows, scatter_list=row_slices if rank == 0 else None, src=0)

    # Initialize local result block
    local_result = torch.zeros_like(local_rows)

    # Compute matrix multiplication in column blocks
    num_blocks = math.ceil(N / block_size)
    for b in range(num_blocks):
        col_start = b * block_size
        col_end = min((b+1) * block_size, N)
        col_block_size = col_end - col_start

        # Prepare column block for this step (scatter among ranks)
        if rank == 0:
            col_slices = [A_full[:, col_start:col_end] for _ in range(world_size)]
        else:
            col_slices = [torch.zeros(N, col_block_size, device=device) for _ in range(world_size)]

        col_block = torch.zeros(N, col_block_size, device=device)
        dist.scatter(col_block, scatter_list=col_slices if rank == 0 else None, src=0)

        # Partial multiplication: local_rows @ col_block
        partial = local_rows @ col_block  # shape: (local_rows_count, col_block_size)

        # Add partial to local_result in correct column slice
        local_result[:, col_start:col_end] += partial

    # Use all_reduce to sum partial results across ranks if needed
    handle = dist.all_reduce(local_result, op=dist.ReduceOp.SUM, async_op=True)
    handle.wait()

    if rank == 0:
        print("Final A @ A shape:", local_result.shape)
        # Optional: compare with single-GPU result
        with torch.no_grad():
            check = A_full @ A_full
            print("Max difference vs single-GPU matmul:", (local_result - check).abs().max().item())

    dist.destroy_process_group()


def main():
    N = 1024         # matrix size
    world_size = 4   # number of processes
    block_size = 256 # column block size
    mp.spawn(worker, args=(world_size, N, block_size), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
