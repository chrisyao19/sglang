# GLM-4.7-Flash BF16 TP1 / H200: existing-table validation runbook

English handoff edition, 2026-09-28.

Repository edition added 2026-10-02: use the adjacent [README](README.md) to clone the validation branch and set `SGLANG_REPO` and `HANDOFF`. No ZIP is needed. Keep those Git paths instead of the standalone example paths below; runtime source is unchanged and GPU validation is still NOT RUN.

Read `SGLANG_TRITON_MOE_GLM_4_7_FLASH_BF16_TP1_H200_VALIDATION_HANDOFF_PROMPT_20260928_EN.md` first. Its opening explains every item in the ZIP, the original workload mismatch, required independent numerical checks and stop conditions. This file is only the short command reference.

## Purpose and current status

Compare the generic heuristic with the existing 20-row table under synthetic four-routed-plus-one-shared routing. Do not search the 1,920-candidate grid, change SGLang runtime code, or modify the candidate JSON.

The original table was generated with random top-4 routing. The target uses `[M,5]`, with shared expert 64 on every token. Old top-4 timings cannot be compared directly with these new measurements.

The helper has passed CPU tests only. New H200 execution, independent kernel numerical validation, serving performance and accuracy remain NOT RUN. The supplied helper does not itself implement the independent numerical reference required by the detailed prompt.

## Requirements

- Linux and exactly one otherwise idle GPU named `NVIDIA H200`; not H100 or H200 NVL.
- The pinned source and a compatible installed environment. The helper enforces PyTorch 2.13.0 / CUDA runtime 13.0 / Triton 3.7.1.
- The bundled model `config.json` and candidate JSON. No full weights or model server are needed for the kernel comparison.
- The shell and Python helper in the same directory.
- Resource-use authorization. These scripts do not rent, start, stop or delete cloud resources or enforce a spending cap.

The historical `_tune.sh` filename is retained, but it supports only `check`, `smoke`, `validate`, and `summarize`. Legacy tuning phases are disabled.

## 1. Set paths and check inputs

Use actual paths for your machine. The example assumes the English ZIP was extracted into its own folder; it does not contain an extra enclosing folder.

```bash
export HANDOFF=/workspace/sglang-triton-moe-glm-4-7-flash-h200-validation-handoff-20260928-en
export SGLANG_REPO=/workspace/sglang-glm47-h200-validation
export MODEL_CONFIG="$HANDOFF/model-config/config.json"
export CANDIDATE_JSON="$HANDOFF/E=65,N=1536,device_name=NVIDIA_H200.json"
export H200_PYTHON=python3
export CUDA_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR=/workspace/glm47-h200-validation-cache
export RUN_ROOT=/workspace/glm47-h200-4plus1-smoke-r1
SCRIPT="$HANDOFF/sglang_triton_moe_glm_4_7_flash_bf16_tp1_h200_tune.sh"

cd "$HANDOFF"
sha256sum -c SHA256SUMS
bash -n "$SCRIPT"
"$H200_PYTHON" -m unittest -v test_sglang_triton_moe_glm_4_7_flash_h200_validate
bash "$SCRIPT" check
```

Check mode verifies inputs and source state without importing CUDA or creating a result directory. It does not establish GPU/dependency compatibility. Check test skips: all CPU checks require PyTorch and `SGLANG_REPO`.

Use a checkout without tracked edits. If the candidate is already committed there, `CANDIDATE_JSON` may be omitted; the shell defaults to the file in the checkout. Explicitly setting the bundled candidate makes the input easier to identify.

## 2. Smoke and independent numerical check

After source/environment review and `nvidia-smi` confirms an authorized idle H200:

```bash
bash "$SCRIPT" smoke
```

This runs M=1 and M=1024 with one round per arm. Logs must show `[M,5]`, shared ID=64, shared visits=M and a passing actual-SGLang selector comparison. Verify both config lookups and paired hashes. A one-round summary is intentionally `incomplete`.

Before expanding, implement/run the small independent PyTorch numerical check described in Stage B3 of the detailed prompt. The current helper only checks finiteness of the final output; it does not establish A/B/reference numerical equivalence.

## 3. Full comparison, not a re-search

Use a NEW directory after smoke and numerical checks pass:

```bash
export RUN_ROOT=/workspace/glm47-h200-4plus1-kernel-ab-r1
unset M_LIST_OVERRIDE
export VALIDATION_ROUNDS=3
export VALIDATION_ITERS=100
bash "$SCRIPT" validate
```

Default M values:

```text
1 2 4 8 16 24 32 48 64 96 128 256 512 1024 1536 2048 3072 4096 6144 8192
```

This creates 120 M/arm/round measurement units, each with 100 samples, in A B A B A B process order. It does not use Ray or search candidates. `M_LIST_OVERRIDE=1,16,1024` can restrict diagnostics, but that is not validation of the complete table. Changing M/repeats/iterations requires a new RUN_ROOT.

## What is measured

- Private complete config trees: stock excludes this key's main/down files across every Triton version; tuned adds the candidate to that base. The checkout is not edited.
- `SGLANG_MOE_CONFIG_DIR` is set to the parent of `configs/`. Stock must resolve to the heuristic; tuned must resolve both projections to the candidate. Reusing the main table for down is expected.
- Normal SGLang `fused_moe` lookup, not a Python tile override.
- Synthetic sigmoid, zero-bias routing: four experts from IDs 0..63 plus shared 64. Shared weight is `1/scale`, becoming one after kernel scaling. This is not captured model routing.
- Paired input/routing seeds and saved hashes. Random weights are generated with the same seed; their tensor hashes are not saved by the current helper.
- One fused-MoE invocation per replay, input restored before every sample, routing copied outside timing, 256 MiB written outside timing to encourage L2 eviction.
- CUDA-event timing after compilation/warmup. This cache-eviction kernel experiment does not establish real serving performance or model accuracy.

## Saving and recovery

Each completed M is atomically saved under `measurements/round-NN-ARM-M=xxx.json`. Re-run the identical validate command to retain completed units and fill missing ones. An interrupted unit is rerun.

Changes to the helper, candidate, model config, source/config tree, settings or GPU/environment cannot be mixed into the same comparison. Use a new directory. Old top-4 tuning markers are not new validation checkpoints.

```bash
# Uses only saved results; no GPU execution.
bash "$SCRIPT" summarize
```

Keep the manifest, environment, raw measurements, logs and summaries. Share compilation caches across compatible runs if useful, but treat them as disposable. Back up evidence locally and verify hashes before releasing any GPU/storage.

## Interpreting results

The helper takes the median of 100 samples per round, then the median across rounds. It reports A/B medians, speedup, latency reduction and round spreads. `within_noise` is a screening label, not proof of nonregression; high variance needs investigation. Round seeds differ, so spread is not a pure timing-noise confidence interval.

Perform independent numerical and timing cross-checks from the detailed prompt before accepting results. Real serving and GSM8K are separate later stages requiring full weights and resource authorization. Re-tuning, separate down/TMA tables and other model postures are not automatic next steps.
