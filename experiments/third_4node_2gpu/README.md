# 4 Nodes x 2 GPUs

## Network

- Intra-node NVLink: 400 GB/s, 936.25 ns
- Intra-node PCIe: 32 GB/s, 1500 ns
- Inter-node: 25 GB/s, 5000 ns

## Parallelism Layout

Node 0: [0,1]
Node 1: [2,3]
Node 2: [4,5]
Node 3: [6,7]

TP2PP4:
[0,1] -> [2,3] -> [4,5] -> [6,7]

TP4PP2:
[0,1,3,2] -> [4,5,7,6]

TP8:
[0,1,3,2,4,5,7,6]

PP8:
[0] -> [1] -> [3] -> [2] -> [4] -> [5] -> [7] -> [6]

## Run

```bash
cd /home/netsys/PC_project/astra-sim
bash experiments/third_4node_2gpu/run_third_config.sh
```
