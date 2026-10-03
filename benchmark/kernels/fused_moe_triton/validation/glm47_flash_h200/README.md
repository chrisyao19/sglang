# GLM-4.7-Flash BF16 TP1 / H200 validation handoff

Repository edition: 2026-10-02. This directory brings the English 2026-09-28 handoff into Git; no ZIP download is required.

## Status: candidate only; H200 validation NOT RUN

The original 20-row table was tuned using random top-4 routing over 65 experts. Target GLM serving uses four routed experts plus shared expert 64 on every token: total top-k five. The candidate therefore has no demonstrated corrected-workload speedup yet.

The supplied validator uses synthetic 4+1 routing and normal SGLang configuration lookup. It has CPU tests, but its H200 smoke, independent kernel numerical checks, kernel performance, full-model serving and accuracy still need to be performed. Do not mistake this commit for a validation result.

## Branches and commits

- Config-only branch: `chrisyao19/sglang-triton-moe-glm-4-7-flash-h200-config`.
- Original config-only commit: [`ffadbf56faded64f2666925c3146624f4cfba95e`](https://github.com/chrisyao19/sglang/commit/ffadbf56faded64f2666925c3146624f4cfba95e).
- Handoff branch: `chrisyao19/sglang-triton-moe-glm-4-7-flash-h200-validation`, based on that config commit.

The original config branch and runtime configuration file are unchanged. This branch adds an isolated handoff directory, not a runtime fix or an upstream PR. Its candidate copy is a reproducibility snapshot and must match the inherited runtime JSON by SHA256.

## Read these first

1. [Detailed English handoff prompt](SGLANG_TRITON_MOE_GLM_4_7_FLASH_BF16_TP1_H200_VALIDATION_HANDOFF_PROMPT_20260928_EN.md): history, root cause, workload contract, exact stages, independent cross-validation, acceptance and stop conditions.
2. [Short English runbook](SGLANG_TRITON_MOE_GLM_4_7_FLASH_BF16_TP1_H200_RUNBOOK_EN.md): helper commands after completing the audit.
3. The helper and CPU tests below. Audit the implementation rather than assuming an agent-written script is correct.

The prompt also explains the earlier ZIP. For this Git edition, use the checkout instructions below instead of its ZIP extraction step. Historical dates and source commits in that prompt describe the earlier experiment, not an assertion that the current checkout is at the old commit.

## Contents and purpose

| Item | Purpose |
| --- | --- |
| `...VALIDATION_HANDOFF_PROMPT_20260928_EN.md` | Detailed English instructions and file-by-file inventory. |
| `...RUNBOOK_EN.md` | Short English command reference. |
| `sglang_triton_moe_glm_4_7_flash_bf16_tp1_h200_validate.py` | Synthetic 4+1 kernel A/B helper with checkpoints and summaries; does not search candidates or run a full model. |
| `sglang_triton_moe_glm_4_7_flash_bf16_tp1_h200_tune.sh` | Entry point for `check/smoke/validate/summarize`; historical name, no longer a tuner. |
| `test_sglang_triton_moe_glm_4_7_flash_h200_validate.py` | CPU tests; not GPU numerical or performance acceptance. |
| `E=65,N=1536,device_name=NVIDIA_H200.json` | Unchanged snapshot of the existing 20-row candidate. |
| `model-config/config.json` | Shape metadata for kernel testing, not model weights. |
| `provenance/` | Original environment, top-4 baseline log and 20 per-M winner rows. Historical evidence only. |
| `references/` | Original broad task and follow-up review, retained for context; not automatic authorization to execute their full scope. |
| `SHA256SUMS` | Hashes of every other file in this directory, including `.gitattributes`. |
| `.gitattributes` | Enforces LF text checkouts for reproducible hashes and Linux shell use. |

No model weights, compiled caches, ZIPs, account credentials or cloud-control tools are committed. Repository formatting may change helper hashes from the previous ZIP; use this directory's manifest and the identities in this edition of the prompt.

The historical baseline log has trailing whitespace normalized for repository checks; its recorded commands and measurements are unchanged. The local ZIP remains the original byte-for-byte snapshot. No historical top-4 timing is used as corrected-workload performance evidence.

## Get this branch and verify it

Use a new directory; do not overwrite an existing checkout or environment. These commands fetch code, but do not install dependencies or use a GPU:

```bash
git clone --single-branch \
  --branch chrisyao19/sglang-triton-moe-glm-4-7-flash-h200-validation \
  https://github.com/chrisyao19/sglang.git /workspace/sglang-glm47-h200-validation

export SGLANG_REPO=/workspace/sglang-glm47-h200-validation
export HANDOFF="$SGLANG_REPO/benchmark/kernels/fused_moe_triton/validation/glm47_flash_h200"
cd "$HANDOFF"
sha256sum -c SHA256SUMS
git -C "$SGLANG_REPO" rev-parse HEAD

RUNTIME_JSON="$SGLANG_REPO/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=65,N=1536,device_name=NVIDIA_H200.json"
cmp "$HANDOFF/E=65,N=1536,device_name=NVIDIA_H200.json" "$RUNTIME_JSON"
```

Record the checked-out validation commit. The validation branch inherits the config commit, and adds no runtime changes; do not check out the older config-only commit in the same directory and expect these helpers to remain present. If deliberately testing another runtime revision, retain the handoff separately and record both revisions.

## CPU tests and input checks

Use an existing appropriate Python environment. All CPU checks require PyTorch and `SGLANG_REPO`; inspect skips. The GPU worker later requires the original PyTorch 2.13.0 / CUDA runtime 13.0 / Triton 3.7.1 environment and an idle H200.

```bash
export H200_PYTHON=python3
export MODEL_CONFIG="$HANDOFF/model-config/config.json"
export CANDIDATE_JSON="$HANDOFF/E=65,N=1536,device_name=NVIDIA_H200.json"
export RUN_ROOT=/workspace/glm47-h200-4plus1-smoke-r1
export TRITON_CACHE_DIR=/workspace/glm47-h200-validation-cache
SCRIPT="$HANDOFF/sglang_triton_moe_glm_4_7_flash_bf16_tp1_h200_tune.sh"

"$H200_PYTHON" -B -m unittest -v test_sglang_triton_moe_glm_4_7_flash_h200_validate
bash -n "$SCRIPT"
bash "$SCRIPT" check
```

`check` performs no GPU work and does not create RUN_ROOT. Continue with stages B and C in the full prompt only after confirming hardware, dependencies, exclusive GPU access and authorization. Use a separate directory for the full comparison after smoke. The scripts do not provision resources, monitor billing or enforce spending limits.

## Local verification recorded on 2026-10-02

- All 16 CPU unittests passed in WSL with the pinned runtime source available, including the upstream Python routing-reference check. No tests were skipped.
- Repository pre-commit checks passed for the added files, including Python syntax, import sorting, Ruff formatting/lint, spelling and private-key detection.
- The script changes from the earlier ZIP are repository formatting/import cleanup, not changes to the experiment. The original runtime JSON and inherited runtime source are unchanged.
- H200 execution, independent kernel numerical validation, corrected-workload performance, serving and accuracy remain `NOT RUN`.

Do not immediately rerun the full tuning search. Never compare the old top-4 baseline log against new top-5 timings. Separate synthetic kernel evidence, real serving evidence and model accuracy in the final report. Preserve any unmeasured category as `NOT RUN`.
