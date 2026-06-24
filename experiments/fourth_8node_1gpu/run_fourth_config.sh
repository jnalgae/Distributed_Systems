#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR="${SCRIPT_DIR}/../.."
ASTRA_SIM="${PROJECT_DIR}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
if [ ! -x "${ASTRA_SIM}" ]; then
  ASTRA_SIM="${PROJECT_DIR}/build/astra-sim/network_frontend/bin/AstraSim_Analytical_Congestion_Aware"
fi
REMOTE_MEMORY="${PROJECT_DIR}/examples/remote_memory/analytical/no_memory_expansion.json"
SYSTEM="${SCRIPT_DIR}/configs/system_replay_8gpu.json"
WORKLOAD_DIR="${SCRIPT_DIR}/workloads"
RESULT_DIR="${SCRIPT_DIR}/results"
RAW_DIR="${RESULT_DIR}/raw"
SUMMARY="${RESULT_DIR}/summary.csv"

mkdir -p "${WORKLOAD_DIR}" "${RAW_DIR}"
cd "${PROJECT_DIR}"

echo "[fourth_8node_1gpu] Building analytical backend"
"${PROJECT_DIR}/build/astra_analytical/build.sh"

echo "[fourth_8node_1gpu] Generating synthetic LLM Chakra traces"
python3 "${SCRIPT_DIR}/scripts/generate_llm_8node_1gpu.py" --output-dir "${WORKLOAD_DIR}"

printf "case,parallelism,microbatches,wall_time_ns_max,latency_ms,gpu_time_ns_max,gpu_time_ms,comm_time_ns_max,comm_time_ms,throughput_reqs_per_sec,raw_log\n" > "${SUMMARY}"

run_case() {
  local case_name="$1"
  local parallelism="$2"
  local microbatches="$3"
  local workload_prefix="$4"
  local raw_log="${RAW_DIR}/${case_name}.log"
  echo "[fourth_8node_1gpu] Running ${case_name}"
  "${ASTRA_SIM}" \
    --workload-configuration="${WORKLOAD_DIR}/${workload_prefix}" \
    --system-configuration="${SYSTEM}" \
    --remote-memory-configuration="${REMOTE_MEMORY}" \
    --network-configuration="${SCRIPT_DIR}/configs/network_inter_node_8gpu.yml" \
    > "${raw_log}" 2>&1
  local wall gpu comm latency_ms gpu_ms comm_ms throughput
  wall=$(awk -F'Wall time: ' '/Wall time:/ { if ($2+0 > max) max=$2+0 } END { print max+0 }' "${raw_log}")
  gpu=$(awk -F'GPU time: ' '/GPU time:/ { if ($2+0 > max) max=$2+0 } END { print max+0 }' "${raw_log}")
  comm=$(awk -F'Comm time: ' '/Comm time:/ { if ($2+0 > max) max=$2+0 } END { print max+0 }' "${raw_log}")
  latency_ms=$(awk -v ns="${wall}" 'BEGIN { printf "%.2f", ns / 1000000 }')
  gpu_ms=$(awk -v ns="${gpu}" 'BEGIN { printf "%.2f", ns / 1000000 }')
  comm_ms=$(awk -v ns="${comm}" 'BEGIN { printf "%.2f", ns / 1000000 }')
  throughput=$(awk -v mb="${microbatches}" -v ns="${wall}" 'BEGIN { if (ns > 0) printf "%.2f", mb * 1000000000 / ns; else printf "0.00" }')
  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "${case_name}" "${parallelism}" "${microbatches}" "${wall}" "${latency_ms}" \
    "${gpu}" "${gpu_ms}" "${comm}" "${comm_ms}" "${throughput}" "${raw_log}" >> "${SUMMARY}"
}

for microbatches in 1 2 4 8; do
  run_case "tp8_mb${microbatches}" "TP8" "${microbatches}" "tp8_mb${microbatches}"
  run_case "pp8_mb${microbatches}" "PP8" "${microbatches}" "pp8_mb${microbatches}"
done

echo "[fourth_8node_1gpu] Summary written to ${SUMMARY}"
cat "${SUMMARY}"
