# ASTRA-sim 기반 LLM 분산 추론 성능 분석

## 프로젝트 개요

본 프로젝트는 ASTRA-sim을 이용하여 대규모 언어 모델(LLM)의 분산 추론 환경에서 모델 병렬화 방식과 GPU 배치 구조가 성능에 미치는 영향을 분석한 실험 프로젝트이다.

LLM의 모델 크기가 증가하면서 단일 GPU에 전체 모델을 올리기 어려워졌고, 이에 따라 Tensor Parallelism(TP)과 Pipeline Parallelism(PP)과 같은 모델 병렬화 기법이 필수적으로 사용되고 있다. 일반적으로 노드 내부의 고대역폭 통신 환경, 특히 NVLink를 사용하는 환경에서는 TP가 유리하고, 노드 간 통신이 많아지는 환경에서는 PP가 유리하다고 알려져 있다. 또한 PCIe 기반 환경에서는 TP 사용을 권장하지 않는 경우가 많은데, 이는 TP에서 반복적으로 발생하는 collective communication 비용이 PCIe의 상대적으로 낮은 대역폭에서 큰 병목이 될 수 있기 때문이다.

본 프로젝트에서는 이러한 경향을 ASTRA-sim 시뮬레이션을 통해 수치적으로 확인하고자 하였다. 이를 위해 총 8개의 GPU를 사용하는 환경을 가정하고, 노드 수와 GPU 배치 구조를 달리하여 TP, PP, Hybrid Parallelism의 성능 차이를 비교하였다. 또한 노드 내부 연결 방식에 따른 차이를 확인하기 위해 NVLink 환경과 PCIe 환경을 각각 나누어 실험하였다.

실험은 LLaMA 2 70B와 유사한 Transformer 구조를 기준으로 하였으며, prefill 단계의 추론 workload를 대상으로 batch completion time과 throughput을 측정하였다.
## 진행 기간

2026.05.14 ~ 2026.06.03

## 실험 환경

### Hardware Assumption

총 8개의 H100급 GPU를 사용하는 환경을 가정하였다.

| 항목                | 값            |
| ----------------- | ------------ |
| GPU 수             | 8            |
| GPU peak compute  | 1000 TFLOP/s |
| Activation dtype  | FP16 / BF16  |
| Bytes per element | 2 bytes      |

### Network Assumption

| 구분                 | Bandwidth |    Latency |
| ------------------ | --------: | ---------: |
| Intra-node NVLink  |  400 GB/s | 0.93625 us |
| Intra-node PCIe    |   32 GB/s |   1.500 us |
| Inter-node Network |   25 GB/s |   5.000 us |

NVLink 환경은 고성능 NVLink/NVSwitch 기반의 intra-node 통신 환경을 가정하였다. PCIe 환경은 PCIe 4.0 x16의 단방향 이론 대역폭을 기준으로 설정하였다. Inter-node network는 200 Gb/s InfiniBand 또는 Ethernet 수준의 대역폭을 가정하였다.

## Model Assumption

실험 대상 모델은 LLaMA 2 70B와 유사한 LLaMA-like Transformer 구조로 설정하였다.

| 항목                        |      값 |
| ------------------------- | -----: |
| Layers                    |     80 |
| Hidden size               |   8192 |
| Intermediate size         |  28672 |
| Sequence length           |   2048 |
| Batch size per microbatch |      1 |
| Activation size           | 32 MiB |

본 실험은 길이 2048 token의 prompt를 한 번에 처리하는 prefill workload를 가정하였다. 긴 sequence에서는 attention 계산량과 activation communication 비용이 커지기 때문에 병렬화 방식에 따른 성능 차이를 관찰하기 적합하다.

## Parallelism Strategies

실험에서는 다음 병렬화 전략을 비교하였다.

* TP8
* PP8
* TP4PP2
* TP2PP4

각 전략은 8개의 GPU를 사용하는 조건에서 Tensor Parallelism과 Pipeline Parallelism의 비율을 다르게 설정한 것이다.

### 1 Node x 8 GPUs
비교 전략: TP8/PP8
```text
Node 0: [0, 1, 2, 3, 4, 5, 6, 7]
```

### 2 Nodes x 4 GPUs
비교 전략: TP8/PP8/TP4PP2/TP2PP4
```text
Node 0: [0, 1, 2, 3]
Node 1: [4, 5, 6, 7]
```

### 4 Nodes x 2 GPUs
비교 전략: TP8/PP8/TP4PP2/TP2PP4
```text
Node 0: [0, 1]
Node 1: [2, 3]
Node 2: [4, 5]
Node 3: [6, 7]
```

### 8 Nodes x 1 GPU
비교 전략: TP8/PP8
```text
Node 0: [0]
Node 1: [1]
Node 2: [2]
Node 3: [3]
Node 4: [4]
Node 5: [5]
Node 6: [6]
Node 7: [7]
```

## Metrics

실험에서는 다음 지표를 측정하였다.

| Metric            | Description                |
| ----------------- | -------------------------- |
| `batch_time_ms`   | 전체 workload가 완료될 때까지 걸린 시간 |
| `throughput_mb_s` | 초당 처리 가능한 microbatch 수     |

## Backend

ASTRA-sim의 Analytical Congestion-Unaware backend를 사용하였다. ASTRA-sim의 Congestion-Aware backend는 링크 경합과 병목을 더 현실적으로 반영할 수 있지만, 현재 1D topology 중심으로 지원되기 때문에 본 실험의 2D node/GPU topology에는 바로 적용하기 어렵다.

## Experimental Results

### 1 Node x 8 GPUs

1 node x 8 GPUs 환경에서는 NVLink를 사용할 때 TP8이 PP8보다 전반적으로 더 낮은 batch time을 보였다. microbatch 수가 1일 때 TP8의 batch time은 74.04 ms, PP8은 331.03 ms로 TP8이 약 4.47배 빠르게 나타났다. microbatch 수가 8로 증가하면 TP8은 592.29 ms, PP8은 620.81 ms로 차이가 줄어들었지만, 여전히 TP8이 약간 더 낮은 batch time을 보였다.

반면 PCIe 환경에서는 microbatch 수가 증가할수록 PP8이 TP8보다 뚜렷하게 유리해졌다. microbatch 수가 1일 때는 TP8이 335.14 ms, PP8이 343.61 ms로 거의 비슷했지만, microbatch 수가 8일 때는 TP8이 2681.15 ms, PP8이 645.99 ms로 나타났다. 즉, PCIe 환경에서 microbatch 수가 8인 경우 TP8은 PP8보다 약 4.15배 더 긴 시간이 걸렸다.

### 2 Nodes x 4 GPUs

2 nodes x 4 GPUs의 NVLink 환경에서는 microbatch 수가 증가함에 따라 TP4PP2가 TP2PP4보다 더 좋은 성능을 보였다. microbatch 수가 1일 때 TP4PP2는 97.68 ms, TP2PP4는 174.42 ms로 TP4PP2가 약 1.79배 빠르게 나타났다. microbatch 수가 8일 때도 TP4PP2는 440.68 ms, TP2PP4는 482.84 ms로 TP4PP2가 더 낮은 batch time을 보였다. 이는 노드 내부 NVLink의 높은 대역폭을 활용하면서 TP 비율을 상대적으로 높게 유지한 구성이 효과적이었기 때문으로 해석할 수 있다.

PCIe 환경에서는 microbatch 수에 따라 유리한 전략이 달라졌다. microbatch 수가 1일 때는 TP8이 166.03 ms로 가장 낮은 batch time을 보였지만, microbatch 수가 4 이상이 되면 TP8의 비용이 크게 증가하였다. microbatch 수가 8일 때 TP8은 1328.26 ms, PP8은 634.56 ms, TP2PP4는 683.38 ms로 나타났다. 즉, PCIe 환경에서 microbatch 수가 커지면 순수 TP8보다 PP8 또는 TP 비율이 낮은 hybrid 구성이 더 효율적이었다.

### 4 Nodes x 2 GPUs

4 nodes x 2 GPUs의 NVLink 환경에서는 microbatch 수가 1일 때 TP8이 140.71 ms로 가장 낮은 batch time을 보였다. 그러나 microbatch 수가 증가하면서 TP8의 batch time은 빠르게 증가하였고, microbatch 수가 8일 때는 TP8이 1125.67 ms, TP2PP4가 484.02 ms로 나타났다. 이 경우 TP2PP4가 TP8보다 약 2.33배 빠르게 나타났다.

PCIe 환경에서도 비슷한 경향을 보였다. microbatch 수가 1일 때는 TP8이 177.01 ms로 가장 낮은 batch time을 보였지만, microbatch 수가 8일 때는 TP8이 1416.05 ms, PP8이 635.12 ms, TP2PP4가 683.66 ms로 나타났다. 즉, PCIe 환경에서는 microbatch 수가 증가할수록 TP8의 반복적인 collective communication 비용이 커지고, PP 또는 TP 비율이 낮은 hybrid 구성이 더 좋은 성능을 보였다.

다만, 2 node x 4 GPUs 환경과 4 node x 2 GPUs 환경에서는 NVLink를 사용하든 PCIe를 사용하든 microbatch 수가 1일 때 TP8이 가장 좋은 성능을 보였다. 이는 microbatch 수가 1인 경우 PP의 pipeline parallelism 효과가 거의 나타나지 않고, pipeline bubble로 인해 여러 stage가 충분히 활용되지 못하기 때문으로 보인다. 반면 TP8은 하나의 microbatch에 대해서도 layer 내부 연산을 8개의 GPU에 나누어 수행할 수 있으므로, 통신 비용이 존재하더라도 단일 microbatch의 batch completion time 측면에서는 더 짧은 실행 시간을 보인 것으로 해석할 수 있다.

### 8 Nodes x 1 GPU

8 nodes x 1 GPU 환경에서는 모든 GPU가 서로 다른 노드에 배치되므로 TP8의 inter-node communication 비용이 가장 크게 나타났다. microbatch 수가 1일 때도 TP8은 444.43 ms, PP8은 347.49 ms로 PP8이 더 낮은 batch time을 보였다. microbatch 수가 8일 때는 TP8이 3555.47 ms, PP8이 653.74 ms로, TP8이 PP8보다 약 5.44배 더 긴 시간이 걸렸다.

이는 TP8이 layer마다 여러 노드에 걸친 AllReduce를 반복적으로 수행해야 하는 반면, PP8은 stage 간 activation 전송 중심으로 동작하기 때문이다. 따라서 노드 간 통신이 많은 환경에서는 TP보다 PP가 더 적합하다는 경향을 수치적으로 확인할 수 있었다.

### Summary of PCIe Results

PCIe 환경에서는 TP8이 PP8보다 microbatch 수 증가에 더 민감하게 성능이 악화되었다. 1 node x 8 GPUs 환경에서 microbatch 수가 8일 때 TP8은 PP8보다 약 4.15배 느렸고, 2 nodes x 4 GPUs 환경에서는 약 2.09배 느렸다. 4 nodes x 2 GPUs 환경에서도 TP8은 PP8보다 약 2.23배 느렸으며, 8 nodes x 1 GPU 환경에서는 약 5.44배 느렸다.

따라서 PCIe 환경에서는 TP의 반복적인 AllReduce 비용이 큰 병목으로 작용하며, 특히 microbatch 수가 증가할수록 PP 또는 TP 비율이 낮은 hybrid parallelism이 더 효율적인 선택이 될 수 있음을 확인하였다.


## Conclusion

본 프로젝트는 ASTRA-sim을 이용하여 GPU 클러스터 구조와 모델 병렬화 전략에 따른 LLM 분산 추론 성능을 비교하였다.

실험 결과, 예상대로 노드 내부의 고대역폭 NVLink 환경에서는 TP가 유리한 경향을 보였다. 반면 PCIe 환경이나 노드 간 통신이 많아지는 환경에서는 TP의 collective communication 비용이 커지면서 PP 또는 hybrid parallelism이 더 효율적인 성능을 보임을 수치적으로 확인할 수 있었다.


## Repository Structure

```text
astra-sim/
├── astra-sim/                    # ASTRA-sim의 기존 소스 코드 디렉터리
└── experiments/                  # 본 프로젝트를 위해 추가한 실험 디렉터리
    ├── csv/                      # 실험 결과 CSV 파일 저장 디렉터리
    ├── first_1node_8gpu/          # 1 node x 8 GPUs 실험
    │   ├── configs/               # 해당 실험에 사용한 ASTRA-sim 설정 파일
    │   ├── scripts/               # 해당 실험 실행에 필요한 스크립트
    │   ├── README.md              # 실험 설정 및 실행 방법 설명
    │   └── run_first_config.sh     # 1 node x 8 GPUs 실험 실행 스크립트
    ├── second_2node_4gpu/         # 2 nodes x 4 GPUs 실험
    │   ├── configs/
    │   ├── scripts/
    │   ├── README.md
    │   └── run_second_config.sh
    ├── third_4node_2gpu/          # 4 nodes x 2 GPUs 실험
    │   ├── configs/
    │   ├── scripts/
    │   ├── README.md
    │   └── run_third_config.sh
    └── fourth_8node_1gpu/         # 8 nodes x 1 GPU 실험
        ├── configs/
        ├── scripts/
        ├── README.md
        └── run_fourth_config.sh
```

`experiments/` 디렉터리는 본 프로젝트에서 새로 추가한 실험용 디렉터리이다.
각 하위 디렉터리는 GPU 배치 구조에 따라 나뉘며, `configs/`에는 ASTRA-sim 실행에 필요한 설정 파일이, `scripts/`에는 실험 실행을 위한 스크립트가 포함되어 있다.
각 실험은 `run_*_config.sh` 스크립트를 통해 실행할 수 있으며, 실험 결과는 `csv/` 디렉터리에 저장된다.

