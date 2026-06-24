# 8 Nodes x 1 GPU

## Network

- Inter-node: 25 GB/s, 5000 ns

## Parallelism Layout

Node 0: [0]
Node 1: [1]
Node 2: [2]
Node 3: [3]
Node 4: [4]
Node 5: [5]
Node 6: [6]
Node 7: [7]

TP8:
[0,1,2,3,4,5,6,7]

PP8:
[0] -> [1] -> [2] -> [3] -> [4] -> [5] -> [6] -> [7]

## Run

```bash
cd /home/netsys/PC_project/astra-sim
bash experiments/fourth_8node_1gpu/run_fourth_config.sh
```
