# 1 Node x 8 GPUs

## Network

- Intra-node NVLink: 400 GB/s, 936.25 ns
- Intra-node PCIe: 32 GB/s, 1500 ns

Node 0: [0,1,2,3,4,5,6,7]

TP8:
[0,1,2,3,4,5,6,7]

PP8:
[0] -> [1] -> [2] -> [3] -> [4] -> [5] -> [6] -> [7]

## Run

```bash
cd /home/netsys/PC_project/astra-sim
bash experiments/first_1node_8gpu/run_first_config.sh
```
