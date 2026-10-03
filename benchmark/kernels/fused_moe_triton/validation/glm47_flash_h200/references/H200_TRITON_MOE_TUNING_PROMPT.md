# Task prompt: generic H200 Triton MoE tuning tables for SGLang

Copy the prompt below into a new agent task. Re-check every fact against the current upstream code
before acting.

```text
GOAL
Build SGLang's Triton fused-MoE tuning tables for NVIDIA H200 across the MoE shapes people commonly
serve on H200. A shape is a key (E, N, dtype, block_shape). Tune for the Triton version that H200
deployments of current SGLang use, prove the new tables do not regress anything, and prepare
upstream-ready commits and PR descriptions. Share the results with the requester before opening
any PR.

WHY
- On H200, `--moe-runner-backend auto` resolves to `triton` for bf16 and FP8 MoE models
  (per-tensor and 128x128 block quantization). The Triton kernel reads a per-shape JSON table of
  tile settings; without one it falls back to a generic heuristic. Measured cost of the heuristic
  on H200 at E=256,N=512 (upstream PR #34150): fused-MoE latency 650 -> 448 us at 1024 tokens
  (-31%), and decode-heavy serving throughput +7% and +12.8% once tuned.
- Coverage on upstream/main a1eb691ab5 (regenerate this yourself in step 0; don't trust these
  numbers):
  - 58 H200 keys exist, but only 4 are tuned for triton 3.7.1. The other 54 are older tables
    reached by version fallback, 24 of them from triton 3.1.0. The lookup code itself notes that
    a 3.1.0 table ran slower on 3.2.0.
  - 60 bf16/FP8 keys tuned for another Hopper or Blackwell GPU (H100, H20, H800, B200, ...) have
    no H200 table at all.
  - Only 7 H200 keys have a separate down-projection table.
- Priority models (in the T1 tier of step 0):
  - Qwen/Qwen3.5-35B-A3B and Qwen/Qwen3.5-35B-A3B-FP8 at TP1;
  - zai-org/GLM-4.7-Flash at TP1, with the default fused shared expert (E=65) and with
    --disable-shared-experts-fusion (E=64);
  - Qwen/Qwen3.5-397B-A17B-FP8 at TP8 EP8.

GROUND RULES
1. GPUs: use H200 GPUs that nothing else is using. Other GPU jobs on the same node distort
   kernel and serving timings, so check `nvidia-smi` before every timed run.
2. Branch from the latest upstream main. Ship JSON tables only; no runtime code changes. Small
   fixes to the tuning scripts under benchmark/kernels/fused_moe_triton/ are welcome as
   separate commits. Follow https://docs.sglang.io/developer_guide/contribution_guide.html
   (pre-commit, tests, benchmark evidence).
3. Tuning needs only each model's config.json (the tuner uses random weights). Download full
   weights only for the models you run end-to-end.
4. End-to-end runs:
   - Use the model's recommended H200 command from the SGLang cookbook, with CUDA graphs left on
     as the defaults set them.
   - If capture runs out of memory, lower --mem-fraction-static rather than disabling graphs.
   - Keep FlashInfer autotune on.
   - Wait for HTTP 200 on /health before sending requests.
   - Both A/B arms must be identical except for the tuning tables.
5. Report in plain language with concrete numbers. Say what you skipped and why.

HOW SGLANG PICKS A TABLE (verified on upstream/main a1eb691ab5; re-check on your base)
- Code: python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py
  (get_moe_configs, try_get_optimal_moe_config). Caller: fused_moe.py::_prepare_fused_moe_run.
- Directory: python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_<X_Y_Z>/, where
  X.Y.Z is the INSTALLED Triton version. Lookup order:
  1. the exact-version directory;
  2. every other triton_* directory, newest first (logs "Fallback to triton version ...");
  3. the generic heuristic (logs "Using default MoE kernel config").
- Which directory to target: check the Triton that ships with the torch pinned in
  python/pyproject.toml (`python3 -c "import triton; print(triton.__version__)"` in that
  environment). Recent NVIDIA tables in upstream are under triton_3_7_1/.
- File name:
  `E={local experts},N={moe_intermediate_size / moe_tp_size},device_name=NVIDIA_H200[,dtype=fp8_w8a8][,block_shape=[128, 128]].json`
  - bf16 has no dtype part, and there is a space inside "[128, 128]".
  - A fused shared expert adds 1 to E (e.g. 257, 65).
  - The key does NOT record hidden_size or top-k, so one table serves every model that shares
    (E, N, dtype, block_shape).
- Rows are keyed by M = the number of tokens entering the MoE layer (hidden_states.shape[0]);
  the nearest row wins. A table with rows only at 256/512/1024 therefore serves every decode
  batch with the 256 row, and every 8k prefill chunk with the 1024 row.
- Down-projection table (`..._down.json`):
  - It is optional. Without it, the down GEMM reuses the up row (no TMA).
  - Both GEMMs share one alignment sort, so the runtime forces the down row's BLOCK_SIZE_M to
    the up row's (warning "down_moe config BLOCK_SIZE_M=... does not match").
  - `"USE_TMA": true` in a row enables TMA for that GEMM; TMA is opt-in per row.
  - TRAP: the _down lookup falls back across Triton versions on its own. So a new triton_3_7_1 up
    table next to an OLD _down file in another version directory gets paired with that old
    table. Every key that has a _down file anywhere also needs a triton_3_7_1 _down file.
- `SGLANG_MOE_CONFIG_DIR=<dir>` replaces the package directory for ALL lookups
  (<dir>/configs/triton_*/), fallbacks included. Copy the full stock configs/ tree into <dir>
  before adding new files.

EXISTING TOOLS (benchmark/kernels/fused_moe_triton/; see README.md there)
- tuning_fused_moe_triton.py ("plain tuner")
  - Times the whole fused_moe with ONE config for both GEMMs, uniform random routing, under
    CUDA graphs. It is model-agnostic, which suits keys shared by many models.
  - Needs only the model's config.json. common_utils.get_model_config derives
    E/N/topk/hidden/block_shape from it. For DeepSeek-V3- and GLM-4.x-style architectures it
    adds the fused shared expert (+1 E) unless --disable-shared-experts-fusion is passed.
  - FP8 needs `--dtype fp8_w8a8` explicitly; block_shape comes from the checkpoint config.
  - Gotcha: compressed-tensors configs make it build block_shape=[0, None], which breaks it. To
    tune a per-tensor FP8 key, point --model at the bf16 checkpoint of the same architecture and
    pass --dtype fp8_w8a8.
  - Flags: `--batch-sizes` sets an explicit M list (the default stops at 4096);
    `--search-space-file` takes a custom list of candidate configs (the default grid is about
    1920 configs for bf16 and 1280 for FP8-block).
  - It starts its own ray instance, spreads the M rows over every visible GPU, and writes the
    JSON into the current directory.
  - Without --tune, it times whatever table the runtime lookup finds (it respects
    SGLANG_MOE_CONFIG_DIR). Use that for before/after numbers; it still forces one config for
    both GEMMs.
  - common_utils.get_config_filename + get_model_config turn (model, tp, ep, dtype, fusion) into
    the exact file name. Use them to generate keys; never hand-write names.
- tuning_fused_moe_triton_sep.py ("up/down tuner")
  - Tunes the up and down GEMMs separately (same BLOCK_SIZE_M), tries TMA for down
    (`--enable-tune-up-tma` for up), and writes X.json AND X_down.json. The recent H200 PRs
    #34150 and #35374 shipped tables made this way.
  - It needs real routing: `--topk-ids-dir` with files topk_ids_layer{L}_idx{j}.pt. It loads 100
    samples as L = i % moe_layers + first_k_dense_replace, j = i // moe_layers, and slices the
    first M rows. So it needs the model's weights, and every capture needs at least as many rows
    as the largest M.
  - It tunes only the default M list (1..4096). A single `--batch-size` runs one worker and does
    NOT write files, so add a `--batch-sizes` flag (mirror the plain tuner) to reach 6144/8192.
  - `--cmp-configs a.json b.json` compares tables under the captured routing; each file's row is
    used for both GEMMs there.
- tuning_client.py + tuning_text.json drive a live server with a ~246k-character real-text prompt,
  long enough for 8192-token prefill chunks.
- test/registered/unit/layers/moe/test_fused_moe_triton_config.py is a CPU unit test that pins
  some shipped tables.
- H200 deployment postures per model: docs/src/snippets/autoregressive/*-deployment.jsx and
  docs/cookbook/autoregressive/**. Example from qwen35-deployment.jsx: "397B-A17B: H200 tp=8 ep=8",
  "35B-A3B: H200 tp=1".
- Serving benchmark (it flushes the cache before each case):
  `python3 -m sglang.benchmark.one_batch_server --model None --base-url http://127.0.0.1:<port> --batch-size <list> --input-len <list> --output-len <list> --result-filename <file>`
- Accuracy: benchmark/gsm8k/bench_sglang.py (used by #35374).

STEPS
Step 0, inventory and target list (0 GPUs)
0a. Set up the environment.
    - Install SGLang from your upstream branch with its pinned dependencies, and record the
      versions of torch, triton, sglang-kernel, flashinfer and the sglang commit.
    - Check that `import ray` works.
    - Check the GPU name: "NVIDIA H200" gives NVIDIA_H200 in file names.
    - Use one persistent TRITON_CACHE_DIR for every tuning and benchmark process; Triton
      compilation dominates tuning time.
0b. Build the model list.
    - Take every MoE model in the SGLang cookbook that has an H200 posture, plus the priority
      models, plus widely used MoE checkpoints the cookbook lacks.
    - Starter list: Qwen3-30B-A3B(-FP8), Qwen3-235B-A22B(-FP8), Qwen3-Next-80B-A3B,
      DeepSeek-V2-Lite, the DeepSeek-V3/R1 family, Kimi-K2, GLM-4.5, GLM-4.5-Air, GLM-4.6/4.7,
      Mixtral-8x7B/8x22B, Llama-4 Scout/Maverick, MiniMax-M2.
    - Keep only models whose MoE runs through the Triton runner on H200 (bf16, FP8 per-tensor,
      FP8 128x128 block). Skip NVFP4 (marlin on H200), MXFP4 (other runners), and recipes that
      pin a different MoE backend.
    - If get_model_config doesn't know an architecture (e.g. Inkling), either add a branch as a
      separate commit, or skip the model and say so.
0c. Choose postures. For each model and each precision it ships in, take:
    - the cookbook's H200 TP/EP;
    - the neighbouring TP values in {1, 2, 4, 8} that fit in memory (weights/TP at most about
      110 GB), keeping intermediate/TP divisible by 128 for FP8-block;
    - EP=TP where the H200 recipe uses EP;
    - the tuner's default shared-expert fusion, plus the disabled variant for the models listed
      above.
0d. Generate exact file names from config.json with the tuner's helpers. Fetch config.json only
    (hf_hub_download of that one file, or a local HF cache); never weights.
0e. Join the keys with the existing tables.
    - For each key, record: the models and postures that use it; whether an H200 up/_down
      table exists in the target Triton directory, only in an older one, or nowhere; and which
      other GPUs have it.
    - Existing H200 or other-GPU keys that no model in your list explains: try to identify the
      model (num_experts in {E, E-1 if fused}, moe_intermediate_size = N x moe_tp) and add it.
      Otherwise list the key as unexplained and skip it; the tuner needs hidden_size and top-k,
      which the key does not carry.
0f. Split the keys into tiers.
    - T1: the priority models, plus the cookbook's H200 default postures for the most-used
      families (DeepSeek-V3/R1, Kimi-K2, Qwen3/3.5/Next, GLM-4.5 to 4.7, GLM-4.5-Air,
      MiniMax-M2, Llama-4, Mixtral).
    - T2: other cookbook or starter postures, and keys already tuned on another Hopper GPU.
    - T3: remaining neighbouring TPs, and old-Triton H200 re-tunes not already in T1/T2.
    - Skip: A100/A800-only keys, int8_w8a16, E=1 keys, and anything H200's Triton runner does
      not serve.
0g. Share the inventory (key, models, current source, tier) and the step-2 pilot timings, with a
    projected total, with the requester before the full run.

Step 1, confirm names against real servers (models whose weights you have; 1 GPU per TP1 model,
8 per TP8/EP8 model)
- Start each server once with its cookbook H200 command:
  `python3 -m sglang.launch_server --model-path <model> --tp <tp> [--ep-size <ep>] ... --port <port>`.
  CUDA-graph capture at startup already triggers every table lookup.
- Run `grep -E "Using MoE kernel config from|Fallback to triton version|Using default MoE kernel config|Down MoE config file not found"`
  on the log. Every logged key must equal the helper-generated name for that posture. If one
  doesn't, fix the generator before tuning anything.

Step 2, pilot (8 GPUs recommended; run one tuner process at a time, since each starts its own
ray instance)
- Tune 3 T1 keys, one each for bf16, FP8-block and FP8 per-tensor. Tune each twice: once with
  the full search space, and once with a seeded space via --search-space-file. The seeded space
  is the union of configs chosen in existing Hopper tables (about 417 distinct configs today),
  plus their BLOCK_SIZE_N/K neighbours.
- Record wall time per key with a cold and a warm Triton cache.
- Use the seeded space for T2/T3 only if its best result is within 1% of the full space's best
  at every M, on all 3 keys.

Step 3, baseline (1-8 GPUs)
- For every key, time today's runtime pick (exact table, fallback table or heuristic) at every
  M:
  `python3 tuning_fused_moe_triton.py --model <path> --tp-size <tp> [--ep-size <ep>] [--dtype fp8_w8a8] [--disable-shared-experts-fusion] --batch-sizes <M list>`
  (no --tune).
- M list: 1 2 4 8 16 24 32 48 64 96 128 256 512 1024 1536 2048 3072 4096 6144 8192. It covers
  decode CUDA-graph sizes and prefill chunks up to 8192.

Step 4, tune by tier (8 GPUs recommended, 1 works; one tuner process at a time)
- Run the plain tuner for every key: the same arguments plus --tune, with the seeded space
  where the pilot allows it.
- Also run the up/down tuner for T1 keys whose weights you have, where routing skew matters
  (e.g. fused shared experts). Capture the routing like this:
  - In a throwaway local copy of SGLang (never committed), add a TP-rank-0 torch.save of
    `topk_output.topk_ids.cpu()` right after `topk_output = self.topk(...)` in the model's MoE
    block. Qwen3.5 uses Qwen2MoeSparseMoeBlock in models/qwen2_moe.py; GLM-4.7-Flash uses
    models/glm4_moe_lite.py. Cover every call site the posture uses.
  - Save only when hidden_states.shape[0] >= 8192, as topk_ids_layer{self.layer_id}_idx{j}.pt,
    for j < ceil(100 / number of MoE layers). Match the loader's layer numbering.
  - With fused shared experts, save the topk+1 ids.
  - For this data-collection run only, serve with --disable-radix-cache,
    --chunked-prefill-size 8192 and `--cuda-graph-backend-prefill disabled`, because Python
    does not run inside a replayed CUDA graph.
  - Drive the server with tuning_client.py. Use real text, never random token ids.
- EP caveat (e.g. Qwen3.5-397B at TP8 EP8): with --ep-size, both tuners route every token's
  top-k among the local experts. In serving, the standard dispatcher maps non-local pairs to -1
  (their blocks only write zeros), so each expert really sees about EP times fewer rows than
  the tuner times. Either tune at B = M/EP and write each row under key M = EP*B (reusing the
  B=1 row for smaller M), or build an EP-faithful routing emulation. The end-to-end run decides.
- Time the first few keys, and update the projection if it changes.

Step 5, assemble (0 GPUs)
- Put the new files in configs/<target triton dir>/, named exactly as the tuners wrote them.
  Keep the tuner's JSON format (indent 4, rows sorted by M, trailing newline). Only merge rows;
  never hand-invent configs.
- Keep an existing row wherever it is faster at its M (step 6 decides). Shared keys also serve
  other models.
- For every key that has a _down file in any version directory, also ship a _down file in the
  target directory. Use the up/down tuner's output if you have it; otherwise copy the new up
  rows as the down table, which reproduces exactly what the plain tuner measured.
- Check that each up row and its _down row have equal BLOCK_SIZE_M, and that USE_TMA appears
  only where it was measured faster.

Step 6, kernel-level validation of every key (1-8 GPUs)
- Rerun step 3 with SGLANG_MOE_CONFIG_DIR pointing at the stock configs/ plus the new files.
- For keys with a _down table, the faithful check is a copy of the tuner's benchmark_config
  with `with override_config(config):` removed. That way the runtime lookup (up + _down + TMA)
  runs as it does in serving. Reuse the tuner's worker setup
  (set_global_server_args_for_scheduler).
- Accept a key only if it is never slower than today's pick by more than about 1% at any M, and
  faster somewhere serving uses (M 1-64 for decode, 512-8192 for prefill).
- Log the max gain, median gain and worst-M result per key.

Step 7, end-to-end A/B on representative models (1 GPU per TP1 model, 8 per TP8/EP8 model; TP1
runs may share a node on separate GPUs)
- Which models: the priority models, plus at least one model per PR group whose weights you
  have. If a group has none, its PR says "kernel-level evidence only".
- Arms. Both use the same cookbook command and differ only in the tables:
  - STOCK: no SGLANG_MOE_CONFIG_DIR.
  - TUNED: SGLANG_MOE_CONFIG_DIR=<dir holding a full copy of the stock configs/ plus the new
    files>.
  - Use a fresh server for every run, in the order A, B, A, B, A, B (3 rounds).
- Cells:
  - decode: `--batch-size 1 8 16 32 64 --input-len 128 --output-len 128`
  - prefill: `--batch-size 1 --input-len 512 2048 4096 8192 --output-len 1`
  - add larger decode batches where the recipe serves them.
- Proof the tables loaded:
  - The TUNED log shows "Using MoE kernel config from <dir>/configs/<target triton dir>/<file>"
    for each tuned key (up and _down).
  - None of those keys shows "Using default MoE kernel config", "Fallback to triton version" or
    "does not match".
  - Both logs print identical server_args.
- Decode cells: a different tile can change rounding, flip a greedy token and send later steps
  to other experts. That changes speed without any kernel change. Send the same fixed prompts
  at temperature 0 to /generate on both arms, compare the output token ids, and trust decode
  deltas only where they match. Otherwise judge on the prefill cells and the geomean.
- Accept if no cell is slower than the STOCK arm's own round-to-round spread (2% if unclear).
  Report every cell's median per arm.

Step 8, accuracy (same GPUs as the server)
- For each end-to-end model, arms in A, B, A order:
  `python3 benchmark/gsm8k/bench_sglang.py --num-questions 1319 --parallel 128 --port <port>`
  (check its arguments and the GSM8K data file).
- Accept if TUNED is within about 1.5 points of STOCK and invalid answers stay at 0.

Step 9, package (0 GPUs)
- Run `python3 -m pytest test/registered/unit/layers/moe/test_fused_moe_triton_config.py -q` and
  `pre-commit run --files <changed files>`.
- Make one PR-sized commit per model family (or per dtype if families are small).
- Draft one PR description per group, modeled on #35374 and #34150. Include:
  - environment: GPU, triton, torch, sglang commit;
  - method: tuner, search space, M list, routing;
  - a generated coverage table: key, models/postures, old source, new;
  - kernel before/after per key;
  - end-to-end A/B and GSM8K for the group's headline model.
- Final report: keys tuned, keys skipped (with reasons), gains, end-to-end results, accuracy,
  remaining gaps, and the location of every artifact.

H200 GPUS PER STEP
- Step 0 inventory and key generation: 0 (only config.json files)
- Step 1 naming check against real servers: 1 per TP1 model, 8 per TP8/EP8 model (only models
  whose weights you have)
- Step 2 pilot: 8 recommended (1 works, slower)
- Step 3 baseline timings: 1-8 (the tool spreads the M rows over the visible GPUs)
- Step 4 tuning: 1 minimum, 8 recommended; one tuner process at a time. Tuning never needs more
  than one GPU per shape, because TP/EP shapes are tuned as the per-GPU shard. Extra GPUs only
  run M rows in parallel.
- Step 4 routing capture (up/down tuner only): 1 per TP1 model, 8 per TP8/EP8 model
- Step 5 assemble: 0
- Step 6 kernel validation: 1-8
- Step 7 end-to-end A/B: 1 per TP1 model, 8 per TP8/EP8 model
- Step 8 GSM8K: same as the server
- Step 9 packaging: 0

Budget guide, NOT measured: the existing tables alone give about 114 candidate keys (54
old-Triton H200 re-tunes plus 60 ports), before model-derived additions. With a warm Triton
cache, the full search space may take tens of minutes per key on 8 GPUs, which puts the whole
set at roughly 1-3 days of an 8-GPU node. The seeded space cuts that several-fold. The step 2
pilot gives the real numbers; share them with the requester before the full run.
```
