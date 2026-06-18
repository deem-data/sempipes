#!/usr/bin/env bash
# Midwest prompt-level experiment runner.
#
# Usage (from repo root or this directory):
#   ./experiments/prompt_levels/midwest/run.sh
#   ./experiments/prompt_levels/midwest/run.sh --tiers lightweight,elaborate --mode baseline
#   ./experiments/prompt_levels/midwest/run.sh --mode optimized --seed 0 --nreps 10
#   ./experiments/prompt_levels/midwest/run.sh --foreground --tiers medium --mode both
#
# Logs and PIDs: experiments/prompt_levels/midwest/logs/
# Results CSVs: experiments/prompt_levels/midwest/results/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

LOG_DIR="${REPO_ROOT}/experiments/prompt_levels/midwest/logs"
PIPELINE="experiments/prompt_levels/midwest/pipeline.py"

TIERS="lightweight,medium,elaborate"
MODE="both"
SEED=42
NREPS=5
NUM_TRIALS=36
CV=5
FOREGROUND=0

usage() {
  cat <<'EOF'
Midwest prompt-level experiment runner.

Usage (from repo root or this directory):
  ./experiments/prompt_levels/midwest/run.sh
  ./experiments/prompt_levels/midwest/run.sh --tiers lightweight,elaborate --mode baseline
  ./experiments/prompt_levels/midwest/run.sh --mode optimized --seed 0 --nreps 10
  ./experiments/prompt_levels/midwest/run.sh --foreground --tiers medium --mode both

Logs and PIDs: experiments/prompt_levels/midwest/logs/
Results CSVs: experiments/prompt_levels/midwest/results/
EOF
  echo ""
  echo "Options:"
  echo "  --tiers LIST       Comma-separated: lightweight,medium,elaborate (default: all)"
  echo "  --mode MODE        baseline | optimized | both (default: both)"
  echo "  --seed N           Starting seed (default: 42)"
  echo "  --nreps N          Repetitions per job (default: 5)"
  echo "  --num-trials N     Colopro trials for optimized mode (default: 36)"
  echo "  --cv N             CV folds for optimized mode (default: 5)"
  echo "  --foreground       Run in foreground (no nohup)"
  echo "  -h, --help         Show this help"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tiers) TIERS="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --nreps) NREPS="$2"; shift 2 ;;
    --num-trials) NUM_TRIALS="$2"; shift 2 ;;
    --cv) CV="$2"; shift 2 ;;
    --foreground) FOREGROUND=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

mkdir -p "${LOG_DIR}"

IFS=',' read -ra TIER_LIST <<< "${TIERS}"

MODES=()
case "${MODE}" in
  baseline) MODES=(baseline) ;;
  optimized) MODES=(optimized) ;;
  both) MODES=(baseline optimized) ;;
  *) echo "Invalid --mode: ${MODE}" >&2; exit 1 ;;
esac

run_job() {
  local tier="$1"
  local mode="$2"
  local log="${LOG_DIR}/midwest_${tier}_${mode}_seed${SEED}_nreps${NREPS}.log"
  local pid_file="${log%.log}.pid"
  local cmd=(
    poetry run python "${PIPELINE}" "${tier}" "${mode}"
    --seed "${SEED}"
    --nreps "${NREPS}"
    --num-trials "${NUM_TRIALS}"
    --cv "${CV}"
  )

  echo "Starting ${tier} / ${mode} -> ${log}"

  if [[ "${FOREGROUND}" -eq 1 ]]; then
    "${cmd[@]}" 2>&1 | tee "${log}"
  else
    nohup "${cmd[@]}" > "${log}" 2>&1 &
    echo $! > "${pid_file}"
    echo "  PID $(cat "${pid_file}")"
  fi
}

for tier in "${TIER_LIST[@]}"; do
  tier="$(echo "${tier}" | xargs)"
  case "${tier}" in
    lightweight|medium|elaborate) ;;
    *) echo "Invalid tier: ${tier}" >&2; exit 1 ;;
  esac
  for mode in "${MODES[@]}"; do
    run_job "${tier}" "${mode}"
  done
done

echo ""
if [[ "${FOREGROUND}" -eq 1 ]]; then
  echo "Done. Logs in ${LOG_DIR}/"
else
  echo "Started $(( ${#TIER_LIST[@]} * ${#MODES[@]} )) job(s). Logs in ${LOG_DIR}/"
  echo "Monitor: tail -f ${LOG_DIR}/midwest_<tier>_<mode>_seed${SEED}_nreps${NREPS}.log"
fi
