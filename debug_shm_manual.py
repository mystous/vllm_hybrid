
import multiprocessing
import random
import time
import os
import torch.distributed as dist
import numpy as np
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.distributed.utils import StatelessProcessGroup
from vllm.utils import get_open_port, update_environment_variables

def worker_fn(env):
    update_environment_variables(env)
    dist.init_process_group(backend="gloo")
    
    rank = dist.get_rank()
    print(f"Rank {rank} started, pid {os.getpid()}")
    
    if rank == 0:
        port = get_open_port()
        ip = '127.0.0.1'
        dist.broadcast_object_list([ip, port], src=0)
    else:
        recv = [None, None]
        dist.broadcast_object_list(recv, src=0)
        ip, port = recv

    stateless_pg = StatelessProcessGroup.create(ip, port, rank, dist.get_world_size())
    
    # Test on WORLD group
    pg = dist.group.WORLD
    writer_rank = 0 # Changed to 0 for simplicity
    
    # Create MessageQueue
    try:
        broadcaster = MessageQueue.create_from_process_group(pg, 40 * 1024, 2, writer_rank)
        print(f"Rank {rank} created MessageQueue")
    except Exception as e:
        print(f"Rank {rank} failed to create MessageQueue: {e}")
        return

    # Sync
    dist.barrier()
    
    test_obj = {"hello": "world", "rank": rank, "data": [1, 2, 3]}
    
    if rank == writer_rank:
        print(f"Rank {rank} broadcasting...")
        broadcaster.broadcast_object(test_obj)
        print(f"Rank {rank} broadcast done")
    else:
        print(f"Rank {rank} waiting for broadcast...")
        obj = broadcaster.broadcast_object(None)
        print(f"Rank {rank} received: {obj}")
        assert obj == test_obj
    
    dist.barrier()
    print(f"Rank {rank} finished")

def test_shm_broadcast():
    world_size = 2
    processes = []
    for i in range(world_size):
        env = {}
        env['RANK'] = str(i)
        env['LOCAL_RANK'] = str(i)
        env['WORLD_SIZE'] = str(world_size)
        env['LOCAL_WORLD_SIZE'] = str(world_size)
        env['MASTER_ADDR'] = '127.0.0.1'
        env['MASTER_PORT'] = str(get_open_port())
        p = multiprocessing.Process(target=worker_fn, args=(env,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
        if p.exitcode != 0:
            print(f"Process failed with exit code {p.exitcode}")

if __name__ == "__main__":
    test_shm_broadcast()
