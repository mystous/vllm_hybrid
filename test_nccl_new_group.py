
import os
import torch
import torch.distributed as dist
import multiprocessing

def worker(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29555'
    
    print(f"Rank {rank}: Initializing process group (gloo)...")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(f"Rank {rank}: Initialized.")

    gpu_ranks = [0]
    print(f"Rank {rank}: Calling new_group(nccl) for ranks {gpu_ranks}...")
    try:
        group = dist.new_group(gpu_ranks, backend="nccl")
        print(f"Rank {rank}: Finished new_group(nccl). Group: {group}")
    except Exception as e:
        print(f"Rank {rank}: Failed new_group(nccl): {e}")
        import traceback
        traceback.print_exc()

    print(f"Rank {rank}: Exiting.")

if __name__ == "__main__":
    world_size = 2
    mp_ctx = multiprocessing.get_context("spawn")
    p0 = mp_ctx.Process(target=worker, args=(0, world_size))
    p1 = mp_ctx.Process(target=worker, args=(1, world_size))
    
    p0.start()
    p1.start()
    p0.join()
    p1.join()
