# Agent handoff: validate the SGLang Triton MoE table for GLM-4.7-Flash BF16 TP1 on H200

Date: 2026-09-28. Read this document completely before executing commands.

Repository edition added 2026-10-02: this prompt and its companion files are now committed together on `chrisyao19/sglang-triton-moe-glm-4-7-flash-h200-validation`. Read the adjacent [README](README.md) for Git checkout instructions, current file identities and the `HANDOFF` path. No ZIP is required for this edition. The ZIP inventory and experiment history below remain useful context. When using this branch, skip the old-commit checkout in A1 and ZIP extraction in A3, preserve the README's `SGLANG_REPO`/`HANDOFF` values, and record the actual validation commit. Runtime source is inherited unchanged from the config-only commit.

## 0. Start here: what you are receiving and what you must do

You are receiving this prompt together with:

```text
sglang-triton-moe-glm-4-7-flash-h200-validation-handoff-20260928-en.zip
```

The ZIP is a self-contained handoff of an existing candidate tuning table, validation helpers, and historical evidence. It is **not** a package of completed, corrected H200 validation results. It does not contain the SGLang source checkout, a runnable Python environment, full model weights, GPU resources, or Triton compilation caches; those must be supplied separately when needed.

We previously spent approximately USD 20 tuning one GLM-4.7-Flash MoE shape on a rented H200. We generated a 20-row JSON table and committed it to GitHub. A subsequent review found that the original benchmark used random top-4 routing over 65 experts, whereas the target serving path uses four routed experts plus one shared expert that every token visits: total top-k 5.

The table is therefore an existing candidate, not a validated performance improvement. New local validation helpers were written to compare it against the default heuristic under synthetic 4+1 shared-expert routing. Those helpers passed CPU tests, but have **not** run on an H200 yet. They must themselves be audited and cross-checked.

Your immediate task is to determine whether the existing table is useful and whether the new measurement method is sound. Do **not** automatically restart the full 1,920-candidate search for all 20 M values. Start with a source audit, a small GPU smoke test and numerical checks, then compare the existing table against the default. Re-tune only when measurements justify it and the user approves.

Use only resources the user has authorized. Do not automatically rent GPUs, add funds, download full weights, change cloud resources, push a branch, or open a PR. The owner mentioned an eight-GPU node, but its model and current availability must be verified. One idle H200 is sufficient for this kernel comparison. An H100 cannot certify an H200 table's performance.

## 1. What is inside the ZIP, and what each item is for

Paths below are relative to the extracted ZIP root. The archive contains files directly at its root, without an extra enclosing directory. Extract it into a dedicated folder. All active instructions in this English package are in English; the original reference prompts are also supplied unchanged.

| File or directory | Purpose | Status and how to use it |
| --- | --- | --- |
| `README.md` | Package entry point and reading order. | Read first. |
| `SGLANG_TRITON_MOE_GLM_4_7_FLASH_BF16_TP1_H200_VALIDATION_HANDOFF_PROMPT_20260928_EN.md` | This detailed task prompt: history, root cause, audit, validation, cross-checking, acceptance and reporting. | Primary execution plan. Independently verify claims against the pinned code; report discrepancies. |
| `SGLANG_TRITON_MOE_GLM_4_7_FLASH_BF16_TP1_H200_RUNBOOK_EN.md` | Short command reference for the existing helper. | Convenience reference, not a replacement for the audit and numerical checks in this prompt. |
| `sglang_triton_moe_glm_4_7_flash_bf16_tp1_h200_validate.py` | Compares the default heuristic with the existing table using synthetic 4+1 routing and normal SGLang config lookup. Saves per-M measurements and supports resuming. | Newly written local helper; CPU-tested, **not H200-tested**. Does not search tuning candidates or run a full model. |
| `sglang_triton_moe_glm_4_7_flash_bf16_tp1_h200_tune.sh` | Shell entry point for `check`, `smoke`, `validate`, and `summarize`. | Historical filename only: it now validates, not tunes. Old `prepare/baseline/pilot/tune/assemble` phases are disabled. Keep it beside the Python helper. |
| `test_sglang_triton_moe_glm_4_7_flash_h200_validate.py` | CPU tests for routing, file isolation, resume handling, and summaries, including an optional upstream Python-reference comparison. | Does not run CUDA kernels or establish model accuracy. PyTorch and `SGLANG_REPO` are needed for all checks to run without skips. |
| `E=65,N=1536,device_name=NVIDIA_H200.json` | The existing 20-row candidate table, identical to the GitHub commit. | Input to validation. Do not treat it as an accepted performance result or edit it before collecting a baseline. |
| `model-config/config.json` | GLM-4.7-Flash architecture and shape metadata. | Sufficient model metadata for the synthetic kernel test. This is **not** the model's weights. |
| `provenance/original-run-info.json` | Original tuning GPU, software versions, source commit and measured shape, including `topk=4`. | Historical evidence of what actually ran. Not metadata for the new validator. |
| `provenance/original-baseline-reference.log` | Original default-config benchmark log under the old top-4 workload. | Historical reference only. **Never divide these old timings by new top-5 timings to claim a speedup.** |
| `provenance/original-tuning-rows/` | Twenty original per-M winner JSON files. | Used to check how the combined table was assembled. Not a full ranking or timing trace for all candidates. |
| `references/H200_TRITON_MOE_TUNING_PROMPT.md` | Original broad, multi-model H200 task. | Background and intended acceptance requirements. Do not automatically execute its broader scope. |
| `references/GLM47_FLASH_H200_TABLE_FOLLOWUP_PROMPT.md` | Review from the recipient's agent, which identified the mismatch and proposed follow-up work. | Review input, not an unquestionable specification. Some claims and commands require the corrections explained below. |
| `SHA256SUMS` | SHA256 hashes of all other files in this package. | Run `sha256sum -c SHA256SUMS` after extraction. The manifest does not include itself. |

Important distinctions:

- The GitHub commit contains **only the candidate JSON**, not these validation helpers. Cherry-picking it alone is insufficient to follow this prompt.
- This ZIP contains no corrected-routing H200 results, serving A/B results, GSM8K results, account credentials, cloud-control scripts, or full search-space timing matrix.
- Older filenames containing `validation` or a `tune-complete` marker do not prove performance or accuracy acceptance. Historical artifacts must stay separate from new results.
- The previous Chinese handoff remains a separate snapshot. Use this English package for the recipient; do not mix manifests or replace files across packages without regenerating the manifest.

Recommended reading order: this opening and inventory -> Sections 2-6 -> helper and tests -> execution stages A-F -> reporting requirements. Consult the two original prompts as historical context, not as instructions to expand the task.

## 2. Fixed target, GitHub state, and artifact identity

- Repository: https://github.com/chrisyao19/sglang
- Branch: `chrisyao19/sglang-triton-moe-glm-4-7-flash-h200-config`
- Configuration commit: `ffadbf56faded64f2666925c3146624f4cfba95e`
- Commit link: https://github.com/chrisyao19/sglang/commit/ffadbf56faded64f2666925c3146624f4cfba95e
- The commit adds exactly one file:

```text
python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=65,N=1536,device_name=NVIDIA_H200.json
```

- Candidate SHA256: `b2bfa97a7f46138af5931a79a2c2f328950aa0c5509ed4749a85273d7e9642df`
- No separate `_down.json` was added. For this key on the pinned source, the down projection reuses the main table.
- No upstream PR was opened. At the handoff snapshot, the local `D:\MLAI\sglang` checkout was clean at the configuration commit above.

The historical commit message says `top-k 4, fused shared expert`. Distinguish four **routed** experts from the total fused-MoE top-k of five. The old tuner actually measured total top-k four. The message also mentions a planned eight-H200-node validation; that is neither a completed validation nor evidence that those resources are currently available. Preserve the historical commit and make these qualifications explicit.

Target:

```text
Model                     zai-org/GLM-4.7-Flash
Architecture              Glm4MoeLiteForCausalLM
Precision                 BF16, unquantized
TP / EP                   1 / 1
Hidden size               2048
Routed experts            64
Shared experts            1, fused by default for the target posture
MoE intermediate size     1536
Routed experts per token  4
Total fused top-k         5 in the target serving path
Lookup key                E=65,N=1536,device_name=NVIDIA_H200
```

The table covers these 20 M values:

```text
1 2 4 8 16 24 32 48 64 96 128 256 512 1024 1536 2048 3072 4096 6144 8192
```

M is the number of tokens entering a MoE invocation, `hidden_states.shape[0]`. It is not model size and is not always request concurrency. Prefill, decode, scheduling and CUDA Graph padding affect it. Runtime lookup chooses the nearest row for an unlisted M; validating a listed row alone does not prove every neighboring runtime M behaves well.

Helper identities for this repository edition (formatting may differ from the earlier ZIP; measurement logic is unchanged):

| File | SHA256 |
| --- | --- |
| `..._validate.py` | `3c0718453b8cd3a77ef4f5dfe409ac88d2a9ac098dee4c15b314a35dac5165d9` |
| `..._tune.sh` | `a3a20e4ccde29a4df7cd0e20ad9846db241442fcf4831bc63f821a2e9a6f9221` |
| `test_..._validate.py` | `598b0f81b03b5187868f90315c8c7e329c602f6a669116889453e215489970b6` |

## 3. What we actually did in the original experiment

The original task covered many models, shapes, precisions and TP/EP postures. To control cost, we narrowed the first experiment to the single GLM shape above.

We downloaded only the model's `config.json`, not full weights. The upstream plain tuner constructs a synthetic MoE layer using random inputs and random weights. One GPU can tune this shape without serving the complete model. However, being able to construct the per-GPU shape does not prove that its synthetic routing matches serving.

Historical environment, taken from `provenance/original-run-info.json`:

```text
GPU              1 x NVIDIA H200, rented SXM offering
SGLang source    1f6ce4b0686232fce5feb0d0d841df7d9bcabc02
Python           3.12.3
PyTorch          2.13.0+cu130
CUDA runtime     13.0
Triton           3.7.1
Ray              2.58.0
sglang-kernel    0.4.7
flashinfer       0.6.18
```

The essential historical invocation was:

```bash
# Historical explanation only. Do not use this for the corrected validation.
python3 benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
  --model /path/to/model-config-directory --tp-size 1 --batch-size "$m" --tune
```

Our wrapper launched individual M values, checkpointed their results, resumed interrupted work and assembled the final table. SGLang's plain tuner, not the wrapper, selected the launch parameters.

The BF16 search space was a Cartesian product, not random JSON generation:

| Parameter | Candidate values |
| --- | --- |
| `BLOCK_SIZE_M` | 16, 32, 64, 128, 256 |
| `BLOCK_SIZE_N` | 32, 64, 128, 256 |
| `BLOCK_SIZE_K` | 64, 128, 256 |
| `GROUP_SIZE_M` | 1, 16, 32, 64 |
| `num_warps` | 4, 8 |
| `num_stages` | 2, 3, 4, 5 |

For each M, the tuner considered 1,920 combinations, skipped configurations that failed compilation or execution, and selected the lowest measured mean among successful candidates in that run. Across 20 M values, this was up to 38,400 candidate attempts. Random input data does not mean the configuration search itself was random.

We saved 20 single-row JSON files and merged them into one 20-row table. These are syntactically valid upstream-format tuning outputs, but they establish only the winners under the original benchmark workload. We do not have a complete raw timing/ranking matrix for all candidate attempts.

Execution spanned two rental sessions. Local copies preserve the model configuration, environment, combined table, all per-M winners and some logs. Compilation caches were reused during the work, but current remote cache/volume availability is unverified. The saved local table does not depend on those remote resources still existing.

The historical results record approximately 3.605 H200-hours of summed per-M wall time. The user's roughly USD 20 also included setup and idle time. Compilation and cache reuse plausibly explain some large differences between M runtimes, but we did not collect profiling sufficient to attribute every interval. Do not use full-search costs to estimate a two-configuration validation.

The candidate was subsequently committed on a newer SGLang base. The original tuning source commit and the configuration commit are different and must both be recorded. Corrected-routing kernel A/B, full-model serving A/B and accuracy acceptance were not completed before this handoff.

## 4. Why the search completed but the result was not yet valid for the target workload

### 4.1 Confirmed mismatch: the shape was right, the routing was not

Inspect these paths on the pinned checkout:

- `benchmark/kernels/fused_moe_triton/common_utils.py::get_model_config`
- `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py::benchmark_config` and its callers
- `python/sglang/srt/models/glm4_moe_lite.py`, particularly MoE/TopK construction and forward
- `python/sglang/srt/layers/moe/topk.py::biased_grouped_topk_impl` and the CUDA selector path

`get_model_config` adds the shared expert to obtain E=65, but its top-k logic includes:

```python
topk = config.num_experts_per_tok + (
    0 if disable_shared_experts_fusion or topk_ids_dir is None else 1
)
```

The plain tuner did not provide `topk_ids_dir`. Its CLI is not a captured-routing interface that can be fixed merely by appending that option. Consequently, E was 65 but total top-k remained four. The historical `shape.topk=4` in `original-run-info.json` confirms what actually ran.

| Property | Original tuning workload | Target GLM serving workload |
| --- | --- | --- |
| Experts in the weight tensors | 65 | 65 |
| Router logit columns | 65 | 64, for routed experts only |
| Assignments per token | Four randomly selected experts | Four routed experts plus the shared expert |
| `topk_ids` shape | `[M,4]` | `[M,5]` |
| Expert 64 | Selected only occasionally | Selected for every token |
| Total token-expert assignments | `4M` | `5M` |
| Shared-expert load | Approximately `4M/65` in expectation | Exactly `M` |

For M=1024, the old workload has 4096 assignments instead of 5120. Expert 64 receives approximately 63 tokens in expectation instead of 1024.

The old workload has 20% fewer assignments than serving. That is a count, not a prediction of a 20% latency difference or a 20% configuration regression. Tile dimensions, skew, padding and memory behavior also matter.

The JSON key does not encode top-k, hidden size or the routing distribution. A correct filename and version directory therefore do not establish that the benchmark measured the correct workload.

### 4.2 What we missed, and what this does not imply

The initial workflow treated "the existing tuner generated the expected key" as sufficient evidence that it matched the serving workload. Before spending time on the full search, we should have checked the model forward path, TopK semantics and tuner inputs together, and asserted the shared-expert column in a preflight run.

This was an audit and validation-design omission. It was not caused by the user renting too few GPUs, by single-GPU tuning being impossible, or by losing data while merging JSON files. Successful benchmark execution is not semantic or performance acceptance.

We do not need to start by changing SGLang's serving kernel, TopK runtime or config lookup. Those are the behaviors the test should match. The current narrower action is to validate the existing candidate using our own corrected helper. A future generic plain-tuner fix should be designed and tested separately as a benchmark-only change.

### 4.3 What can still be reused

The table, M coverage, metadata and saved results remain useful candidates and evidence. Some or all rows may beat the heuristic under the corrected workload; some may regress. Neither claim has yet been measured.

It is not justified to discard all work, and it is equally unjustified to claim the table is probably faster as if that were a result. Compare the existing candidates first. The money already spent did not produce a fully accepted serving-faithful result; do not hide the missing validation.

## 5. Corrections and qualifications to the review prompt

The supplied review correctly identified the shared-expert mismatch and lack of performance evidence. Still, verify its line numbers, commands and inferences rather than executing it verbatim.

1. Small-M tile changes and disagreement with H100 are not, by themselves, proof of noise. Repeated measurements and independent confirmation are needed.
2. The old timed graph contained ten MoE invocations. Tuning used ten timed graph replays, not ten total MoE calls. `prepare(i)` changes routing between replays, so the entire experiment did not use one fixed set of experts.
3. Repeating identical routing within a graph and flushing L2 only before the timing loop can create cache behavior unlike layer-by-layer serving. This is a measurement risk, not proof that every winner is wrong. The new helper's explicit eviction policy is also an experimental condition, not a perfect simulation of serving.
4. H100 rows are useful candidate references, not an H200 answer key. Similar large-M tiles/warps/stages do not imply all six fields match; `GROUP_SIZE_M` and M=6144 coverage differ. Compare complete dictionaries and record the source table's Triton version.
5. In the pinned GLM deployment snippet, TP1 is the default. The condition explicitly adding `--attention-backend triton` is B200, not H200. Attention and MoE backends are distinct. Explicitly test `--moe-runner-backend triton` and keep the attention backend identical between arms.
6. `benchmark/gsm8k/bench_sglang.py` does not exist in the pinned checkout. Stage F below uses an existing entry point and explains its limitations. Re-check `--help` if changing versions.
7. PRs #34150 and #35374 provide examples of methodology and reporting, not performance results for this table. H100 PR #23719 is a reference, not cross-device validation.
8. Completing all validation within an hour has not been measured here. Environment setup, compilation, model loading and evaluation output lengths can dominate. Time the smoke test before estimating.

References:

- https://github.com/sgl-project/sglang/pull/34150
- https://github.com/sgl-project/sglang/pull/35374
- https://github.com/sgl-project/sglang/pull/23719

## 6. What the new helper implements, and its current limits

### 6.1 Implemented behavior

- Restricted to this BF16, TP1, E65/N1536, four-routed-plus-one-shared target; not a general validator for every model.
- Synthetic sigmoid routing over 64 logit columns; selects four routed experts and appends shared expert 64 for every token. All generated routing samples are checked for `[M,5]` and the fixed shared column.
- The first four weights are normalized. The shared weight is `1 / routed_scaling_factor`; for this model, scale=1.8, giving effective shared weight one after kernel scaling.
- Synthetic correction bias is zero. Learned model correction bias and real routing distributions are not loaded.
- For the first routing sample of each M, it calls SGLang's actual `select_experts`, then compares IDs and weights after sorting by expert ID. The remaining samples receive routing-invariant checks.
- Calls the original `fused_moe` through normal configuration lookup, without Python `override_config`.
- Creates two private, complete config trees. A excludes this key's main and down files in every Triton-version directory. B adds the candidate to the same base. The original checkout is untouched.
- This prevents an already-present candidate, cross-version fallback or a hidden down table from contaminating A.
- A/B use matching seeds, inputs, synthetic weights and routing for the same round/M. Input and routing hashes are saved and checked. Weights use matching deterministic generation but do not currently have a saved tensor hash.
- One fused-MoE invocation per graph replay. The original input is restored before each replay so an inplace result is not fed back as the next input. Routing is updated outside the timing interval.
- Before every timed sample, a 256 MiB buffer is written outside the timing interval to encourage L2 eviction. This is an explicit cache-eviction experiment, not hardware proof of every cache's state.
- Compilation and warmup precede timing. CUDA events time each replay. Defaults: 100 samples per M/arm/round, three rounds in fresh arm processes, ordered A B A B A B.
- Atomic per-M checkpoints and resume support. Old top-4 tuning checkpoints cannot masquerade as corrected validation results.
- Changes to inputs, script, environment, configuration trees or GPU UUID require a new comparison directory rather than mixing evidence.

### 6.2 What has and has not been tested

Recorded local checks: all 16 CPU unittests passed on 2026-09-28 in WSL with PyTorch 2.7.0, including extraction of SGLang's Python routing reference for a CPU comparison. Python/Bash syntax and CLI/check-mode checks also passed. These do not execute CUDA and do not reproduce the original GPU environment.

**NOT RUN: new H200 smoke; corrected-workload kernel performance; real serving; model accuracy.**

The CPU reference comparison is not evidence that the actual CUDA selector has run. That check only happens when the GPU worker executes. The helper checks finiteness of the last kernel output; it does not save A/B outputs for numerical comparison and does not implement an independent PyTorch MoE gold reference. "Sixteen tests passed" must not be reported as "the GPU kernel and performance are validated."

Additional limitations:

- Fixed A-then-B ordering can have thermal/order bias; perform an independent reverse-order check.
- Routing/input seeds change across rounds. Round spread combines workload variation and timing variation; it is not a confidence interval.
- There are no `--arm-order` or `--no-cache-flush` options. Do not invent them. An independent cross-check harness is needed for those experiments.
- The helper does not support a separate down/TMA table. Adding such a table changes the experiment and requires corresponding validation changes.

## 7. Stage A: audit and preparation without GPU execution

### A1. Pin the source; reproduce before porting

Prefer an existing authorized idle H200 environment with a separate checkout/venv. Do not disrupt another running installation. Example for a new, unused directory:

```bash
git clone --no-checkout https://github.com/chrisyao19/sglang.git /workspace/sglang-glm47-h200-validation
git -C /workspace/sglang-glm47-h200-validation checkout --detach ffadbf56faded64f2666925c3146624f4cfba95e
git -C /workspace/sglang-glm47-h200-validation status --short
git -C /workspace/sglang-glm47-h200-validation show --stat --oneline HEAD
```

An existing clean checkout may be used, but record its actual commit and any relevant differences. A different commit is not an exact reproduction of this pinned base.

Do not simply `pip install -U sglang`, upgrade Triton, or rename the table's version directory. The current helper requires PyTorch 2.13.0, CUDA runtime 13.0, Triton 3.7.1, device name `NVIDIA H200`, and imports from the specified source checkout. Record a compatible driver; the driver-displayed CUDA capability does not have to be the same string as the PyTorch CUDA runtime.

Prepare dependencies according to the checkout's constraints, preferably using a matching existing environment. Installation details depend on the actual container/venv. If incompatible, report the mismatch rather than deleting assertions to make the test run. "Latest software" is not automatically the correct environment for a table generated on Triton 3.7.1.

If ultimately submitting against newer upstream, treat that as a separate porting stage: pin the new commit, inspect relevant runtime/benchmark/dependency diffs, and revalidate. Do not `git pull` halfway through an experiment.

### A2. Verify inputs and configuration lookup

Run `sha256sum -c SHA256SUMS` inside the extracted package. Confirm the candidate matches the GitHub commit exactly.

Read these functions and write a short workload contract:

```text
common_utils.py                       get_model_config / get_configs_compute_bound
tuning_fused_moe_triton.py             benchmark_config / prepare / tune
glm4_moe_lite.py                       shared fusion / TopK / routed scaling
topk.py                               select_experts / biased_grouped_topk_impl
fused_moe_triton_config.py             get_moe_configs / try_get_optimal_moe_config
fused_moe.py                           _prepare_fused_moe_run / activation / reduction
```

Include E/N/hidden/dtype/TP/EP, w1/w2 shapes, total top-k=5, shared ID=64, normalization and scaling, gate/up layout, inplace behavior, up/down lookup and the timing boundary.

The helper times the complete fused-MoE invocation, including its internal alignment, GEMMs, activation and reduction. It generates and copies routing outside the interval. Therefore label the result fused-MoE kernel time, not full layer time including router/TopK or end-to-end serving latency.

Inventory this key's main/down files across all `triton_*` directories. If newer source already provides a table for the key, the helper's A remains a deliberately stripped-key historical heuristic baseline, not that newer upstream's unmodified default. Add an actual-upstream baseline or clearly qualify the comparison.

`SGLANG_MOE_CONFIG_DIR` points to the parent containing `configs/`, not to `configs/` itself. A directory containing only the candidate JSON changes unrelated lookup/fallback behavior and is not a faithful serving A/B setup.

### A3. Extract the package, then perform CPU checks and dry-run

These are example Linux paths. Replace them with the actual paths and choose a GPU index authorized for this task. The extraction directory should be new; do not merge this package into an old results folder.

```bash
export HANDOFF=/workspace/sglang-triton-moe-glm-4-7-flash-h200-validation-handoff-20260928-en
mkdir -p "$HANDOFF"
unzip /path/to/sglang-triton-moe-glm-4-7-flash-h200-validation-handoff-20260928-en.zip -d "$HANDOFF"

export SGLANG_REPO=/workspace/sglang-glm47-h200-validation
export H200_PYTHON=python3
export MODEL_CONFIG="$HANDOFF/model-config/config.json"
export CANDIDATE_JSON="$HANDOFF/E=65,N=1536,device_name=NVIDIA_H200.json"
export TRITON_CACHE_DIR=/workspace/glm47-h200-validation-cache
export CUDA_VISIBLE_DEVICES=0
export RUN_ROOT=/workspace/glm47-h200-4plus1-smoke-r1
SCRIPT="$HANDOFF/sglang_triton_moe_glm_4_7_flash_bf16_tp1_h200_tune.sh"

cd "$HANDOFF"
sha256sum -c SHA256SUMS
bash -n "$SCRIPT"
"$H200_PYTHON" -m unittest -v test_sglang_triton_moe_glm_4_7_flash_h200_validate
bash "$SCRIPT" check
```

CPU tensor tests require PyTorch. The upstream-reference test also requires `SGLANG_REPO`. Inspect skipped-test counts; an OK containing skipped checks does not establish full coverage.

`check` does not import CUDA, create the output directory, or verify GPU compatibility. It cannot replace the GPU smoke test. The source checkout must have no tracked edits. If changing a helper, record its new hash and use a new RUN_ROOT.

## 8. Stage B: H200 smoke and numerical cross-validation

### B1. Resource and environment preflight

Confirm resource authorization before execution; this document is not authorization to rent GPUs. Inspect `nvidia-smi`: model, UUID, driver, clocks, power and other compute processes. Use an otherwise idle GPU. On an eight-GPU node, expose one H200 and keep TP1/EP1. Switching the model to TP8 changes N and the lookup key; it does not validate this table.

Record Python, `torch.__version__`, `torch.version.cuda`, `triton.__version__`, kernel/FlashInfer package versions, `sglang.__file__`, source commit and exact commands. Passing the CPU check does not establish dependency compatibility.

### B2. Minimal smoke

```bash
# Continue with the environment from A3 and its dedicated smoke directory.
bash "$SCRIPT" smoke
```

This runs M=1 and M=1024, one round each for the two arms. Inspect the logs and artifacts, not just the exit status:

- Expected GPU and software versions.
- `[M,5]`, shared ID=64 and shared visits=M for each M.
- Actual SGLang selector cross-check passes.
- A resolves no up/down table; B resolves the candidate for both projections without cross-version fallback.
- Saved up/down configs correspond to the intended M/arm.
- Four measurement files exist, paired input/routing hashes match, and timings are positive and finite.
- A one-round smoke summary is intentionally `incomplete`: it is not three-round performance acceptance.

Stop on failure. Preserve logs and determine whether the cause is imports/versions, routing, config isolation or kernel execution. Do not suppress selector/lookup assertions and proceed through every M.

### B3. Required numerical cross-check: not implemented in the supplied helper

Add a small independent local check, not a SGLang runtime change. Use the pinned `test/registered/moe/test_fused_moe.py::torch_naive_moe` as a structural reference, but do not reuse its softmax/top-k assumptions for GLM's sigmoid plus fixed-shared semantics.

Requirements:

1. Test at least M=1,16,1024 with multiple fixed seeds and the actual target tensor shapes. Include the always-busy shared expert and a skewed routed-expert case.
2. Fix/save identical inputs, weights, IDs and routing weights for A, B and the reference, with tensor hashes. Do not independently regenerate different data for each arm.
3. Independently compute `x @ W1[e].T`, the correct gate/up SiLU-and-mul, `@ W2[e].T`, and weighted/scaled combination in PyTorch. Calling the same fused kernel again is not a gold reference.
4. Compare A/reference, B/reference and A/B; report maximum absolute error, relative/normalized error and failure locations. Check all tested outputs for NaN/Inf, not only the last replay.
5. BF16 does not require bitwise equality. Predefine tolerances and explain intermediate rounding. Existing BF16 tests can inform tolerance selection, but do not widen tolerances after observing failures merely to pass. Add an FP32 reference if needed for diagnosis.
6. Restore unmodified inputs before every inplace call, and explicitly verify that the shared expert's effective weight is one.

Complete this small numerical check before expanding the performance run. If pinned APIs and the helper disagree, fix the smallest helper issue, preserve the old version and record a new hash. Do not silently modify the serving, TopK or lookup implementations used as references.

## 9. Stage C: compare all 20 existing rows without re-tuning

After smoke and numerical checks succeed, use a new result directory:

```bash
export RUN_ROOT=/workspace/glm47-h200-4plus1-kernel-ab-r1
unset M_LIST_OVERRIDE
export VALIDATION_ROUNDS=3
export VALIDATION_ITERS=100
bash "$SCRIPT" validate
```

The default work is 20 M values x 2 arms x 3 rounds = 120 measurement units, with 100 samples each. This compares two configurations per M, not 38,400 candidate attempts. Wall time still includes compilation, process setup, GPU data preparation and cache eviction; do not estimate it as just 12,000 times a single kernel latency.

Result layout:

```text
run-manifest.json
environment.json
stock/configs/...
tuned/configs/...
logs/round-NN-stock.log
logs/round-NN-tuned.log
measurements/round-NN-ARM-M=xxx.json
summary.json
summary.csv
```

Each per-M checkpoint records the seed, input/routing hashes, resolved up/down configurations, all 100 raw timing samples, median/mean and wall time. Preserve raw files, not just screenshots or averages.

Resume by rerunning the same `validate` command with the same inputs, helper, GPU/environment and RUN_ROOT. Completed M units are retained; an interrupted unit is redone. A new machine/GPU requires a new comparison directory rather than mixing its samples into an old round. Compilation caches may be copied and reused when compatible, but measurement provenance must remain separate.

```bash
# Rebuild summaries from saved files; no GPU execution.
bash "$SCRIPT" summarize
```

Smoke and validation use different result directories but may share `TRITON_CACHE_DIR`. Caches are disposable accelerators; measurements and metadata are evidence. Back up completed files locally during the run and verify hashes before releasing resources. A remote volume's existence is not a permanent backup guarantee.

### C1. How summaries are computed and interpreted

For each M/arm/round, take the median of 100 samples. Then take the median of the three round medians.

```text
speedup = stock_median_us / tuned_median_us
latency_reduction_pct = 100 * (stock - tuned) / stock
spread_pct = 100 * (max(round_medians) - min(round_medians)) / median(round_medians)
helper noise_band_pct = max(1%, stock_spread_pct, tuned_spread_pct)
```

Report M, all A/B round values, A/B medians, speedup, latency reduction, both spreads and disposition. A 1.20x speedup is not a 20% latency reduction.

`faster`, `regression` and `within_noise` are screening labels, not statistical proofs. In particular, a wide spread can hide a regression; `within_noise` is not an automatic pass. Reduce uncontrolled variation or add independent repeats. Since seeds change across rounds, also repeat a fixed workload to estimate pure timing variability.

Never compare the historical top-4 baseline log directly with the new top-5 measurements. Both arms of any ratio must use the same method and workload.

## 10. Stage D: independent cross-validation

Mark each of the following as passed, failed or NOT RUN. Repeating the same helper twice is useful for repeatability, but is not an independent check of that helper's implementation.

### D1. Independent semantics and numerical implementation

B3's PyTorch reference and the actual SGLang selector comparison provide different evidence: MoE arithmetic and routing, respectively. Neither is replaced by checking the helper against its own routing function.

Also inspect the actual GLM forward arguments: total top-k, shared count/ID, scaling, activation, inplace mode and weight layout. Learned correction bias is not covered by the zero-bias synthetic generator; check it when using real routing.

### D2. Independent timing of key M values

Preselect M=1,16,128,1024,8192, then add every regression/high-variance row discovered in Stage C. Use a separate small harness with normal A/B config lookup, the same GPU and matching data. Run at least three independent repetitions, with balanced/reversed order, for example AB, BA, AB, BA.

Independently inspect timing boundaries, synchronization, input restoration and resolved configs. Do not simply import the original `measure_m` into another file and call that an independent timing implementation.

Use two separately labeled cache policies:

- `explicit-eviction`: eviction outside the timing interval, comparable to the primary experiment.
- `no-explicit-eviction`: natural cache retention, with input restoration and routing updates still outside the timing interval.

Compare A/B only within each policy. Do not divide A from one method by B from another. Include fixed-routing repeats and varying-routing repeats to separate timing noise from workload differences.

If a gain appears only with artificial eviction, report the condition dependence instead of claiming a serving gain. If independent implementations disagree, explain the disagreement rather than keeping only the favorable result.

### D3. Real-model serving as a separate level of evidence

This needs complete BF16 model weights. The synthetic kernel stages do not. This table targets GLM Flash TP1; eight available GPUs do not justify changing to TP8.

Check that weights, KV cache and CUDA Graph allocations fit on one idle H200 for the chosen settings. Do not promise arbitrary concurrency/context lengths will fit. Start conservatively and apply any memory-setting changes identically to both arms. If weights or download/resource authorization are unavailable, deliver a kernel-level report with serving marked NOT RUN.

## 11. Stage E: serving A/B when resources and weights are available

### E1. Only the tables differ

Use Stage C's complete `stock` and `tuned` trees. Explicitly set the override directory for A as well: the feature checkout already contains the candidate, so unsetting `SGLANG_MOE_CONFIG_DIR` would incorrectly make A use B's table.

Fix weights/tokenizer revision, dtype, TP/EP, attention and MoE backends, graph settings, memory settings, prefix-cache policy, speculative settings, request data, seeds and sampling parameters. Do not introduce speculative decoding or unrelated optimizations in this first validation.

Example server command, run one server at a time, with a separate terminal for clients:

```bash
export MODEL_WEIGHTS=/models/GLM-4.7-Flash-fixed-snapshot
export KERNEL_RESULT=/workspace/glm47-h200-4plus1-kernel-ab-r1
export SERVING_ROOT=/workspace/glm47-h200-serving-ab-r1
export ARM=stock
export ROUND=1
mkdir -p "$SERVING_ROOT"
export PYTHONPATH="$SGLANG_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_MOE_CONFIG_DIR="$KERNEL_RESULT/$ARM"

"$H200_PYTHON" -m sglang.launch_server \
  --model-path "$MODEL_WEIGHTS" --tp-size 1 --dtype bfloat16 \
  --moe-runner-backend triton --host 127.0.0.1 --port 30000 \
  2>&1 | tee "$SERVING_ROOT/server-$ARM-round-$ROUND.log"
```

The example does not force an attention backend. Record the resolved backend and keep it identical. If using an existing recipe that explicitly sets it, use the same recipe for both arms. Do not add `--disable-shared-experts-fusion`. Check the pinned environment's `--help`; CLI examples were checked against source parsers, not executed on H200.

Use a fresh server for every arm/round, in ABABAB or a predefined balanced order. Wait for HTTP 200 from `/health` before requests, e.g. `curl --fail http://127.0.0.1:30000/health`. Bound readiness waiting with a timeout and save startup failures. Stop only the server you launched, and confirm its GPU processes exit before starting the next arm; do not broadly `pkill python` or restart a shared node.

Required evidence:

- B logs load `.../tuned/configs/triton_3_7_1/E=65,N=1536,device_name=NVIDIA_H200.json`.
- No B default or cross-version fallback for this key. A missing-down message that explicitly reuses the main table is expected here, not a failure.
- A uses the heuristic for this key; unrelated keys behave identically.
- Resolved server arguments match apart from documented operational differences such as log locations and the table override environment variable.
- No competing GPU work. Compilation, graph capture and warmup are excluded from steady-state metrics.

### E2. Performance cells

In the client terminal, set the same HANDOFF/SGLANG_REPO/H200_PYTHON/PYTHONPATH/SERVING_ROOT/ARM/ROUND/MODEL_WEIGHTS variables as appropriate. Prepare a fixed ShareGPT data file, set `SERVING_DATA=/datasets/sharegpt-fixed.json`, and record its hash. In this version, the `random` dataset samples that text and truncates/repeats it to length; it is not uniformly random token IDs. Missing data can trigger a download, so prepare it before paid measurement time.

```bash
# Decode: start here; add batch size 128 only after checking memory.
"$H200_PYTHON" -m sglang.benchmark.one_batch_server \
  --model None --base-url http://127.0.0.1:30000 \
  --batch-size 1 8 16 32 64 --input-len 128 --output-len 128 \
  --result-filename "$SERVING_ROOT/decode-$ARM-round-$ROUND.jsonl"

# Prefill: keep server graph behavior identical between arms.
"$H200_PYTHON" -m sglang.benchmark.one_batch_server \
  --model None --base-url http://127.0.0.1:30000 \
  --batch-size 1 --input-len 512 2048 4096 8192 --output-len 1 \
  --result-filename "$SERVING_ROOT/prefill-$ARM-round-$ROUND.jsonl"

# Fixed-length serving: concurrency 16 first; separately test 64 if feasible.
"$H200_PYTHON" -m sglang.benchmark.serving \
  --backend sglang --base-url http://127.0.0.1:30000 \
  --model "$MODEL_WEIGHTS" --dataset-name random --dataset-path "$SERVING_DATA" --num-prompts 256 \
  --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1 \
  --max-concurrency 16 --seed 20260928 --temperature 0 \
  --output-file "$SERVING_ROOT/serving-c16-$ARM-round-$ROUND.jsonl" --output-details
```

A fixed dataset seed does not replace testing actual, unmodified requests, nor does it guarantee identical model outputs. Report successful/failed request counts, actual input/output token counts, throughput, TTFT, TPOT, latency percentiles, and every cell's three-round median/spread. Do not report only the best cell or relabel kernel gains as model throughput gains.

### E3. Decode output and real-routing cross-checks

Predefine real text prompts, tokenizer and request bodies. Send the same requests to A/B `/generate` with temperature zero and identical output limits. Save raw responses and generated token IDs. Check the pinned response schema; do not retokenize decoded text and call it the original generated IDs. If using token IDs from returned logprob fields, document extraction.

For each request, report exact token agreement, first divergence, finish reason and output length. BF16 rounding changes can flip a greedy token and alter later routing. Divergence is not automatically a kernel correctness failure, but it makes the decode workloads different, so the entire throughput delta cannot automatically be attributed to tiles.

A small fixed-prompt agreement check is supplementary. To claim a timed decode cell is a matched-workload comparison, save and compare that cell's actual requests/outputs. Mark divergent cells as confounded and prioritize fixed-input prefill and kernel evidence.

Optional real-routing capture: in a separate temporary instrumentation checkout, save actual TopK IDs/weights, layer IDs, M and serving settings. Check the shared column and expert histograms across real-text decode/prefill. Do not include capture hooks or CPU copies in timed serving runs. Python hooks do not execute on every replay of an already-captured CUDA Graph; any graph changes needed for capture are capture-only and must be recorded.

The supplied helper has no captured-routing input option. An offline replay extension must be separately identified, versioned and documented. Do not claim the synthetic tests already used real routing, or automatically expand into separate up/down/TMA tuning.

## 12. Stage F: accuracy is distinct from performance

For complete model-level acceptance, evaluate GSM8K with fresh servers in A, B, A order. First run a small evaluator/template sanity check; run the full set only with an appropriate resource budget.

The pinned checkout still has the entry point below, although it is marked deprecated. It is used here for pinned-version reproducibility, not as a claim about future preferred tooling. Prepare a fixed 1319-row GSM8K test JSONL, set `GSM8K_DATA=/datasets/gsm8k-test-fixed.jsonl`, and record its hash.

The example explicitly uses zero-shot: this evaluator otherwise takes its first five rows from the same file as few-shot examples, leaving only 1314 scored test questions.

```bash
"$H200_PYTHON" -m sglang.test.run_eval \
  --base-url http://127.0.0.1:30000 --eval-name gsm8k --api chat \
  --gsm8k-data-path "$GSM8K_DATA" \
  --num-examples 1319 --num-threads 32 --num-shots 0 \
  --max-tokens 2048 --temperature 0 --repeat 1 \
  2>&1 | tee "$SERVING_ROOT/gsm8k-$ARM-round-$ROUND.log"
```

Concurrency 32 is a conservative example, not an accuracy criterion. Use identical concurrency if increasing it. Zero-shot absolute scores are not directly comparable with another PR's five-shot scores. For five-shot evaluation, fix separate nonoverlapping examples and verify the evaluator's split. Check the actual scored question count, not just the requested `--num-examples`.

Fix dataset/revision/hash, chat template, thinking settings, stop/EOS policy and maximum output length. Investigate whether 2048 tokens truncate reasoning and cause invalid answers; adjust consistently before formal A/B/A, not only for B.

The evaluator writes reports to the `/tmp/gsm8k_<model>.html` and `.json` paths shown in its logs. Later runs may overwrite them. Immediately copy each round's actual files into separate arm/round directories and retain per-question answers and logs. If switching to `sgl-eval`, verify its version/recipe and use one evaluator consistently; do not compare scores from different scorers as an A/B result.

Report accuracy, invalid/unparsable answers, request errors, truncations and per-question disagreements. The review's approximately 1.5-percentage-point accuracy margin and zero invalid answers can be predefined screening criteria, not proof of equivalence. If A already produces invalid answers, investigate template/evaluator/truncation instead of recording those as zero.

## 13. When to retain the table, selectively re-tune, or stop

Show Stage C/D evidence to the user before proposing the smallest necessary remediation.

| Observation | Next action |
| --- | --- |
| Routing, lookup or numerical check fails | Fix the helper or diagnose semantics first; do not continue large timing runs or hide the issue by editing rows. |
| All 20 rows are stably nonregressing, with clear gains in relevant cases | Keep the candidate and proceed to serving/accuracy if feasible; a full re-search is not mandatory. |
| A few M values show stable regressions | Prioritize those M values and confirm affected neighboring M/nearest-row boundaries. |
| Widespread regression or contradictory measurement methods | Withhold conclusions; investigate workload, cache, order and version before deciding on full re-tuning. |
| Only differences within unresolved noise | Report insufficient evidence rather than manufacturing a gain. |
| Faster kernel but no serving gain | Report both separately; inspect model bottlenecks and output differences. |

If re-tuning is approved:

1. First make a minimal benchmark-only fix or standalone tuner ensuring correct 4+1 IDs, weights and scaling both initially and in every `prepare`. Changing only the integer top-k is insufficient.
2. Initially compare the existing row, heuristic and corresponding H100 row as limited candidates. Measure H100-derived settings on H200. Some heuristic parameters lie outside the old 1,920 grid; the grid winner does not automatically include or beat the heuristic.
3. Expand to a full search only when warranted. Use persistent compilation caches and do not run competing timing jobs on the same GPU.
4. Separate search samples from acceptance samples, using new seeds/processes. Selecting a noisy winner among many candidates and reusing that same observation is not confirmation.
5. Save candidate rankings and raw measurements, not only winners; checkpoint each M atomically.
6. Revalidate changed/affected M values and neighboring lookup boundaries, then at least one complete 20-row comparison for the final table. Do not attach an old table's report to a new table hash.
7. Separate up/down/TMA tuning is optional later work, not a prerequisite to fix the current mismatch. It changes lookup behavior, test scope and BLOCK_SIZE_M constraints.

Do not add E64, other TP/EP postures, other precisions or other models without separate authorization. Multiple models may share this filename key, but this experiment covers GLM's hidden size/top-k/routing, not all possible consumers.

## 14. Acceptance, budget and stop conditions

A minimally credible kernel-level conclusion requires:

- Workload contract matches source semantics and the actual selector.
- Independent numerical comparison passes, not just a finite-output check.
- All 20 rows have matched A/B raw measurements and proof of config resolution.
- Regressions/high-variance cases are investigated, with independent timing on key M values.
- Synthetic-routing, cache-policy and untested serving/accuracy limitations are explicit.

For full serving acceptance, also require per-cell nonregression within predefined repeatability bounds, a clear gain in a relevant workload, disclosure of output confounders, and no significant GSM8K A/B/A anomaly. The review's 1% kernel and 2% serving margins are screening rules, not permission to accept noisy data by widening the noise band.

Resource rules:

- CPU audit and packaging need no rented GPU.
- Synthetic kernel validation needs one H200 and the model config, not full weights.
- Model-level GLM Flash TP1 validation needs a sufficiently idle H200 with suitable memory settings; eight GPUs are not required for this key.
- Confirm any paid-resource budget and time limit before provisioning or expanding work.
- Use measured smoke compilation/wall time to estimate the next stage. Old full-search cost per M is not the cost of this comparison.
- The helper does not receive cloud balance alerts, implement a spending cap or automatically stop cloud resources. Manage those separately under user authorization.
- Before releasing resources, back up result evidence and verify hashes. Caches are not the primary artifact.

Stop and report on GPU/version mismatch, routing/lookup/numerical disagreement, competing GPU jobs, result-directory/hash conflicts, persistent uncontrolled variance, insufficient budget, or a need for new download/provisioning authorization. Do not keep rerunning until favorable numbers appear.

## 15. Required deliverables

Produce a professionally named result bundle, for example `sglang-triton-moe-glm-4-7-flash-bf16-tp1-h200-validation-YYYYMMDD`, containing:

1. A README with conclusions, scope, reproducible commands, completed/skipped stages and limitations.
2. Original/final candidates and hashes, with per-M provenance and reasons for changes.
3. Environment metadata: source commit, GPU/UUID, driver, software and model/tokenizer revision.
4. Versioned helpers and tests, clearly separated from upstream runtime changes.
5. Main run manifest, all raw measurements, complete logs and per-M summaries.
6. Independent numerical-reference source/error reports and independent timing methods/data/results.
7. If executed: serving requests/results, table-loaded logs, token comparisons and all GSM8K rounds/per-question records.
8. Every skipped stage explicitly marked `NOT RUN`; never substitute another PR's gains.
9. A SHA256 manifest and verified bundle integrity.

The short final report must answer:

```text
Can the existing table be retained? Which rows have supporting evidence?
What is the per-M improvement/regression versus the heuristic?
What independent checks support that conclusion?
Were real-model speed and accuracy tested? If not, what is missing?
Is further paid tuning necessary? If so, which M values and why?
What is the smallest useful next action?
```

If preparing an upstream contribution, run the relevant config tests and required formatting checks on the final table. Keep benchmark fixes in a separate commit. Present the report and PR draft for review before any authorized push/PR creation. Do not rewrite the already-shared historical commit merely to obscure the original mismatch.

## 16. What your first response should establish

Before substantial GPU work, report:

1. Your pinned source/environment and whether the available GPU meets this task's requirements.
2. Whether you independently confirm the original top-4 versus serving 4+1 mismatch, citing code and historical run evidence.
3. Any actual defects in the supplied validator that must be fixed first, clearly distinguished from tests not yet executed.
4. The smallest next stage and required resources. Do not default to re-searching all 20 M values without evidence.

Core principle: first establish that the benchmark measures the intended workload, then establish whether the table improves it. The old table is a candidate, and the new validator is a tool awaiting GPU verification; neither is the conclusion.
