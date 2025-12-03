# Distributed processing in CPU



## Setup
1. Install the requirements. 
2. Go to scripts


### Run the distributed script

1. Distributed processing on CPU using "gloo backend"



2. Run NUMA-aware distributed processing on CPU using "gloo backend"
    - Here we are using `numactl` to bind each process to a specific NUMA node. This helps in optimizing memory access patterns and improving performance.
    - Example command to run 2 processes, each bound to a different NUMA node:
    On terminal 1:
    ```bash
    MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 WORLD_SIZE=2 RANK=0 numactl -C 0-23 -m 0  python multi_process_numaaware.py
    ```

    On terminal 2:
    ```bash
    MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 WORLD_SIZE=2 RANK=1 numactl -C 24-47 -m 1 python multi_process_numaaware.py
    ``` 

NOTE (Interesting fact) - The process will not give you output until you start both the process.

Try:
   1. Try putting `dist.barrier()` at different locations in the code and see how it affects the execution.
   2. Try changing the CPU core bindings and observe any performance changes.

