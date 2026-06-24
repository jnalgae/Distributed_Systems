#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR="${SCRIPT_DIR}/../.."
ASTRA_SIM="${PROJECT_DIR}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware"
if [ ! -x "${ASTRA_SIM}" ]; then
  ASTRA_SIM="${PROJECT_DIR}/build/astra-sim/network_frontend/bin/AstraSim_Analytical_Congestion_Unaware"
fi
REMOTE_MEMORY="${PROJECT_DIR}/examples/remote_memory/analytical/no_memory_expansion.json"
SYSTEM="${SCRIPT_DIR}/configs/system_replay_8gpu.json"
COMM_GROUPS="${SCRIPT_DIR}/configs/comm_groups.json"
WORKLOAD_DIR="${SCRIPT_DIR}/workloads"
RESULT_DIR="${SCRIPT_DIR}/results"
RAW_DIR="${RESULT_DIR}/raw"
SUMMARY="${RESULT_DIR}/summary.csv"

mkdir -p "${WORKLOAD_DIR}" "${RAW_DIR}"

cd "${PROJECT_DIR}"

echo "[second_2node_4gpu] Building analytical backend"
"${PROJECT_DIR}/build/astra_analytical/build.sh"

echo "[second_2node_4gpu] Generating synthetic LLM Chakra traces"
python3 "${SCRIPT_DIR}/scripts/generate_llm_2node_4gpu.py" \
  --output-dir "${WORKLOAD_DIR}"

printf "case,parallelism,interconnect,microbatches,wall_time_ns_max,latency_ms,gpu_time_ns_max,gpu_time_ms,comm_time_ns_max,comm_time_ms,throughput_reqs_per_sec,raw_log\n" > "${SUMMARY}"

run_case() {
  local case_name="$1"
  local parallelism="$2"
  local interconnect="$3"
  local microbatches="$4"
  local workload_prefix="$5"
  local network_cfg="$6"
  local raw_log="${RAW_DIR}/${case_name}.log"

  echo "[second_2node_4gpu] Running ${case_name}"
  "${ASTRA_SIM}" \
    --workload-configuration="${WORKLOAD_DIR}/${workload_prefix}" \
    --comm-group-configuration="${COMM_GROUPS}" \
    --system-configuration="${SYSTEM}" \
    --remote-memory-configuration="${REMOTE_MEMORY}" \
    --network-configuration="${network_cfg}" \
    > "${raw_log}" 2>&1

  local wall
  local gpu
  local comm
  local latency_ms
  local gpu_ms
  local comm_ms
  local throughput

  wall=$(awk -F'Wall time: ' '/Wall time:/ { if ($2+0 > max) max=$2+0 } END { print max+0 }' "${raw_log}")
  gpu=$(awk -F'GPU time: ' '/GPU time:/ { if ($2+0 > max) max=$2+0 } END { print max+0 }' "${raw_log}")
  comm=$(awk -F'Comm time: ' '/Comm time:/ { if ($2+0 > max) max=$2+0 } END { print max+0 }' "${raw_log}")
  latency_ms=$(awk -v ns="${wall}" 'BEGIN { printf "%.2f", ns / 1000000 }')
  gpu_ms=$(awk -v ns="${gpu}" 'BEGIN { printf "%.2f", ns / 1000000 }')
  comm_ms=$(awk -v ns="${comm}" 'BEGIN { printf "%.2f", ns / 1000000 }')
  throughput=$(awk -v mb="${microbatches}" -v ns="${wall}" 'BEGIN { if (ns > 0) printf "%.2f", mb * 1000000000 / ns; else printf "0.00" }')

  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "${case_name}" "${parallelism}" "${interconnect}" "${microbatches}" \
    "${wall}" "${latency_ms}" "${gpu}" "${gpu_ms}" "${comm}" "${comm_ms}" \
    "${throughput}" "${raw_log}" \
    >> "${SUMMARY}"
}

for microbatches in 1 2 4 8; do
  for mode in tp4pp2 tp2pp4 tp8 pp8; do
    label=$(echo "${mode}" | tr '[:lower:]' '[:upper:]')
    run_case \
      "${mode}_mb${microbatches}_nvlink_intra" \
      "${label}" \
      "NVLink-intra" \
      "${microbatches}" \
      "${mode}_mb${microbatches}" \
      "${SCRIPT_DIR}/configs/network_nvlink_intra_2node_4gpu.yml"

    run_case \
      "${mode}_mb${microbatches}_pcie_intra" \
      "${label}" \
      "PCIe-intra" \
      "${microbatches}" \
      "${mode}_mb${microbatches}" \
      "${SCRIPT_DIR}/configs/network_pcie_intra_2node_4gpu.yml"
  done
done

echo "[second_2node_4gpu] Summary written to ${SUMMARY}"
cat "${SUMMARY}"
