# 2 Nodes x 4 GPUs

## Network

- Intra-node NVLink: 400 GB/s, 936.25 ns
- Intra-node PCIe: 32 GB/s, 1500 ns
- Inter-node: 25 GB/s, 5000 ns

Node 0: [0,1,2,3]
Node 1: [4,5,6,7]

TP4PP2:
[0,1,2,3] -> [4,5,6,7]

TP2PP4:
[0,1] -> [2,3] -> [6,7] -> [4,5]

TP8:
[0,1,2,3,7,6,5,4]

PP8:
[0] -> [1] -> [2] -> [3] -> [7] -> [6] -> [5] -> [4]

## Run

```bash
cd /home/netsys/PC_project/astra-sim
bash experiments/second_2node_4gpu/run_second_config.sh
```

