#!/usr/bin/env bash
# Historical filename retained; this entry point now VALIDATES the existing table.
# It never runs the old plain tuner or searches 1,920 candidates.
set -euo pipefail

phase="${1:-help}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
validator="$script_dir/sglang_triton_moe_glm_4_7_flash_bf16_tp1_h200_validate.py"
H200_PYTHON="${H200_PYTHON:-python3}"

case "$phase" in
  help|-h|--help)
    printf '%s\n' \
      'Phases: check (no GPU), smoke, validate, summarize (no GPU)' \
      'Required: SGLANG_REPO, RUN_ROOT, MODEL_CONFIG=/path/to/config.json' \
      'Optional: CANDIDATE_JSON (defaults to the file in SGLANG_REPO)' \
      'Optional: H200_PYTHON, CUDA_VISIBLE_DEVICES=0, TRITON_CACHE_DIR' \
      'Optional: M_LIST_OVERRIDE=1,16,1024 VALIDATION_ROUNDS=3 VALIDATION_ITERS=100' \
      'Use a NEW RUN_ROOT for this corrected-routing comparison.' \
      'Run smoke in its own RUN_ROOT; rerun the same validate command to resume.' \
      'Legacy prepare/baseline/pilot/tune/assemble phases are disabled.'
    exit 0 ;;
  check|smoke|validate|summarize) ;;
  *) printf 'Unsupported phase: %s. Use help; legacy random-top4 tuning is disabled.\n' "$phase" >&2; exit 2 ;;
esac

: "${RUN_ROOT:?Set RUN_ROOT to a NEW corrected-routing result directory}"
if [[ "$phase" == summarize ]]; then
  exec "$H200_PYTHON" "$validator" summarize --output "$RUN_ROOT"
fi
: "${SGLANG_REPO:?Set SGLANG_REPO to the installed SGLang source checkout}"
: "${MODEL_CONFIG:?Set MODEL_CONFIG to the local GLM-4.7-Flash config.json file}"
CANDIDATE_JSON="${CANDIDATE_JSON:-$SGLANG_REPO/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=65,N=1536,device_name=NVIDIA_H200.json}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$RUN_ROOT/triton-cache}"
export PYTHONUNBUFFERED=1

command=("$H200_PYTHON" "$validator" run --sglang-repo "$SGLANG_REPO"
  --model-config "$MODEL_CONFIG" --candidate "$CANDIDATE_JSON" --output "$RUN_ROOT"
  --rounds "${VALIDATION_ROUNDS:-3}" --iterations "${VALIDATION_ITERS:-100}")
if [[ -n "${M_LIST_OVERRIDE:-}" ]]; then
  IFS=', ' read -r -a m_values <<< "$M_LIST_OVERRIDE"
  command+=(--batch-sizes "${m_values[@]}")
fi
if [[ "$phase" == check ]]; then
  command+=(--check-only)
elif [[ "$phase" == smoke ]]; then
  command+=(--batch-sizes 1 1024 --rounds 1)
fi
exec "${command[@]}"
