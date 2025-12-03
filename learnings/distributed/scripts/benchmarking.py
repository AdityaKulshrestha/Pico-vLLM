import os 
import sys
import time
import torch 
import torch.distributed as dist
import torch.multiprocessing as mp



## Config
TENSOR_SIZE = 10_00_000
ITERS = 20
WORLD_SIZES = [2]
BACKEND = "gloo"            # MPI error: RuntimeError: Distributed package doesn't have MPI built in. MPI is only included if you build PyTorch from source on a host that has MPI installed.
MODE = "dist"
VERBOSE = True

def single_process_workload():
    """Single process baseline (without communication)"""
    
    data = torch.ones(TENSOR_SIZE, dtype=torch.float32)
    start = time.perf_counter()
    for i in range(ITERS):
        # Perform computation
        data = data * 1.1  + 1.0
        
        if VERBOSE and i % 10 == 0:
            print(f"[SINGLE] Iter {i}/{ITERS}")
    end = time.perf_counter()
    
    elapsed_time = end - start
    print(f"[SINGLE] Time taken for {ITERS} iterations: {elapsed_time:.4f} seconds")
    return elapsed_time


def distributed_workload(rank, world_size):
    """Distributed workload with communication"""
    
    
    data = torch.ones(TENSOR_SIZE, dtype=torch.float32)
    
    # sync all ranks before timing
    dist.barrier()
    
    start = time.perf_counter()
    for i in range(ITERS):
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
        data = data * 1.1 + 1.0
        if VERBOSE and rank == 0 and i % 10 == 0:
            print(f"[DIST] Rank {rank} Iter {i}/{ITERS}")
            
    end = time.perf_counter()
    
    elapsed_time = end - start
    print(f"[Rank: {rank}] worload time: {elapsed_time:.2f} s")
    # Accumulate the time for all the rank process and average it.
    t = torch.tensor([elapsed_time], dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    average_time = t.item() / world_size
    
    if rank == 0:
        print(f"[Rank 0] Average workload time across {world_size} ranks: {average_time:.2f}s")
        
    return elapsed_time


def init_process(rank, world_size, backend, fn):
    """Initialize the distributed environment"""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    fn(rank, world_size)
    
    dist.destroy_process_group()
    
    
def run_dist_world_size(world_size, backend=BACKEND):
    """Launch world size proceses and bnechmark."""
    print(f"\n======== Running distributed benchmark with world size: {world_size} and Backend: {backend} ========")  
    
    processes = []
    
    script_start = time.perf_counter()
    
    mp.set_start_method("spawn", force=True)
    
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, backend, distributed_workload))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    script_end = time.perf_counter()
    print(f"[MAIN] World size = {world_size} Total Wall time: {script_end - script_start:.2f} s")  
    
    
if __name__ == "__main__":
    print(f"Running benchmark with TENSOR_SIZE={TENSOR_SIZE}, ITERS={ITERS}, VERBOSE={VERBOSE}")
    
    if MODE == "single":
        total_start = time.perf_counter()
        _ = single_process_workload()
        total_end = time.perf_counter()
        print(f"[MAIN] Single process total wall time: {total_end - total_start:.2f} s")
        
    elif MODE == "dist":
        overall_start = time.perf_counter()
        
        for ws in WORLD_SIZES:
            run_dist_world_size(ws, backend=BACKEND)
        overall_end = time.perf_counter()
        
        print(f"[MAIN] Distributed benchmark overall wall time: {overall_end - overall_start:.2f} s")
        
