import os
import torch
import torch.distributed as dist


def run(rank, world_size):
    data = torch.tensor([10 * (rank + 1)], dtype=torch.float32)
    dist.all_reduce(data)
    print(f"Rank {rank} has data {data[0].item()}")


def main():
    backend = "gloo"
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )
    run(rank, world_size)
    dist.destroy_process_group()

if __name__ == "__main__":
    # How to run
    # Terminal 1: rank 0
    # numactl -C 0-23 -m 0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 WORLD_SIZE=2 RANK=0 python multi_process_numaaware.py

    # # Terminal 2: rank 1
    # numactl -C 24-47 -m 1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 WORLD_SIZE=2 RANK=1 python multi_process_numaware.py
    main()
