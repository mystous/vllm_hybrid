
import os
import torch
import torch.distributed as dist
import multiprocessing
import uuid

def run_worker(rank, world_size, init_method):
    if rank == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print(f"Rank {rank}: Starting. Init method: {init_method}")
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            world_size=world_size,
            rank=rank
        )
        print(f"Rank {rank}: Init success!")
    except Exception as e:
        print(f"Rank {rank}: Init failed: {e}")
        return

    print(f"Rank {rank}: Barrier...")
    dist.barrier()
    print(f"Rank {rank}: Barrier passed!")
    dist.destroy_process_group()

def main():
    world_size = 2
    init_method = f"file:///tmp/test_dist_init_{uuid.uuid4()}"
    # Match vLLM environment
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.init()
    print(f"Rank 0: CUDA initialized. Device count: {torch.cuda.device_count()}")
    
    # Spawn Rank 1 with CPU env
    ctx = multiprocessing.get_context("spawn")
    # We need to set env execution for child?
    # spawn picks up env from parent, but we want child to have NO gpu.
    # We can't easily change env for child process object only in python multiprocessing without wrapper or os.environ modification before spawn.
    # We will modify os.environ temporarily? No, `spawn` copies env at spawn time?
    # Actually spawn launches new python interpreter. It inherits os.environ.
    # So we must set CUDA_VISIBLE_DEVICES="" for child inside child?
    
    p = ctx.Process(target=run_worker, args=(1, world_size, init_method))
    # We want run_worker to clear CUDA_VISIBLE_DEVICES or we set it here?
    # We can pass a flag to run_worker to modify env.
    
    p.start()
    
    # Main process (Rank 0)
    print(f"Rank 0: Starting (Main). Init method: {init_method}")
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            world_size=world_size,
            rank=0
        )
        print(f"Rank 0: Init success!")
        print(f"Rank 0: Barrier...")
        dist.barrier()
        print(f"Rank 0: Barrier passed!")
        dist.destroy_process_group()
    except Exception as e:
        print(f"Rank 0: Failed: {e}")
    
    p.join()
    print("Done")

if __name__ == "__main__":
    main()
