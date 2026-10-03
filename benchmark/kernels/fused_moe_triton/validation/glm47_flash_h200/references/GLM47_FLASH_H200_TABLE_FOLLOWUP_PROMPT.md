# Follow-up prompt: make the GLM-4.7-Flash H200 Triton MoE table solid

Copy the prompt below into a new agent task. Re-check every fact against the current upstream code
before acting.

```text
CONTEXT
You are following up on commit ffadbf56fa ("perf: add GLM-4.7-Flash H200 Triton MoE config") on
branch chrisyao19/sglang-triton-moe-glm-4-7-flash-h200-config. It adds one file,
python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=65,N=1536,device_name=NVIDIA_H200.json
It covers bf16, TP1, fused shared expert, 20 rows for M=1..8192, and was made with
benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py on 1x H200 (PyTorch 2.13.0+cu130,
Triton 3.7.1, sglang 1f6ce4b068). Goal of this follow-up: prove the table helps, fix the parts
that don't match serving, and turn it into a mergeable upstream PR. Code references below are
for upstream main 14ab74ef95; re-check them on your base.

WHAT WAS DONE WELL (keep it)
1. Right key for standard serving. GLM-4.7-Flash fuses its one shared expert by default on NVIDIA
   GPUs (SM80+) when expert parallelism is off (python/sglang/srt/models/glm4_moe_lite.py:899).
   The cookbook's H200 command (`--tp 1 --attention-backend triton`) keeps that default, so
   serving looks up E=65, N=1536, bf16.
2. It fills a real gap. Upstream has no H200 table for this key in any triton_* directory, so today
   the kernel runs the generic default config.
3. Right directory: triton_3_7_1/ matches the Triton that ships with the pinned torch 2.13, and the
   commit records torch, Triton and the sglang commit.
4. Full row coverage: 20 rows from 1 to 8192, including 6144 and 8192, which the tuner's default
   list does not reach.
5. A clean, config-only change in the tuner's own JSON format, with a commit message that states
   plainly what was NOT validated.
6. The large-batch rows (M >= 1536) match the tuned H100 table for the same key
   (128x256x64 tiles, 8 warps, 4 stages), which is a good sanity signal.
7. No stale down-table trap. No H200 `_down` file exists for this key in any directory, so the
   down GEMM reuses these rows, which is exactly what the tuner measured.

WHAT IS WRONG OR MISSING (fix it)
1. No performance evidence. There is no before/after kernel timing, no serving A/B and no accuracy
   check, so the commit cannot claim a speedup yet. For scale: tuning a similar H200 shape
   (E=256,N=512, PR #34150) cut fused-MoE time 31% at 1024 tokens and raised decode-heavy
   serving throughput 7-13%. This table's gain is unknown.
2. The tuner measured a different workload than serving.
   - Serving: with the fused shared expert, the TopK layer returns 5 experts per token. The
     last column is set to the shared expert (index 64) for EVERY token
     (python/sglang/srt/layers/moe/topk.py:1189).
   - Tuner: common_utils.get_model_config adds the extra expert (E=65), but keeps top-k at 4
     unless --topk-ids-dir is given (benchmark/kernels/fused_moe_triton/common_utils.py:129).
     It also routes uniformly at random over all 65 experts.
   - So the tuner timed 20% fewer routed rows than serving, and no always-busy expert. At
     serve time expert 64 gets M rows, which is the biggest single GEMM in the layer once M is
     in the hundreds or more. The table is probably still better than the default, but its
     rows were chosen for the wrong load.
3. The decode rows (M <= 512) look like noise.
   - Neighbouring rows switch tile shapes back and forth (BLOCK_N 32, 32, 128, 128, 64, 32 for
     M = 1, 2, 4, 8, 16, 24), and they disagree with the H100 table for the same key.
   - In --tune mode each config is timed with only 10 CUDA-graph replays
     (tuning_fused_moe_triton.py:362).
   - Each replay runs the same 10 invocations on the same experts (line 213), with a single
     L2 flush before the loop (line 224). At small M, much of the touched experts' weights can
     stay in H200's 50 MB L2 between invocations.
   - So small-M timings favour cache hits, and the gaps between configs are within noise.
4. Only one posture is covered (optional to fix). Deployments that pass
   --disable-shared-experts-fusion look up E=64, N=1536 instead, which still has no H200 table.
5. Recent H200 tables (#34150, #35374) ship separate up and down tables, with TMA for the down
   GEMM where it measured faster. The plain tuner cannot explore that; an optional step below
   covers it.

NEXT STEPS (in order)
Step 1: make the plain tuner match serving (benchmark tool only; separate commit)
- In benchmark/kernels/fused_moe_triton/, when the model fuses a shared expert (the case where
  get_model_config adds +1 to E and --disable-shared-experts-fusion is not set):
  - time with top-k = k+1;
  - set the last top-k column to the shared-expert index (E-1) for every token, mirroring
    topk.py:1189;
  - do this both in the initial select_experts call and in prepare(i)
    (tuning_fused_moe_triton.py:169-173).
- Leave every other model unchanged, and add a small check that prints the resulting top-k
  shape and last column.
- This is a real upstream improvement for every fused-shared-expert model (GLM-4.x,
  DeepSeek-V3 family), so describe it in the PR.

Step 2: kernel before/after (1 GPU)
- STOCK: stock tree with no H200 table for this key, i.e. the default config.
- TUNED: SGLANG_MOE_CONFIG_DIR=<dir> containing a full copy of the stock configs/ tree plus the
  new file. The variable replaces the whole lookup directory, fallbacks included, so the copy
  must be complete.
- For each setting, run the tuner in benchmark mode with the Step 1 routing, 3 times, in fresh
  processes:
  `python3 benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py --model zai-org/GLM-4.7-Flash --tp-size 1 --batch-sizes 1 2 4 8 16 24 32 48 64 96 128 256 512 1024 1536 2048 3072 4096 6144 8192`
  (no --tune)
- Report, per M: the median time for each setting, the speedup, and the spread across the 3
  runs. The spread is your noise band.

Step 3: re-tune with the corrected routing (1 GPU works; 8 GPUs are about 8x faster, because the
tuner spreads M rows over the visible GPUs; run one tuner process at a time)
- Same command plus --tune, full default search space, same 20 M values. Use a persistent
  TRITON_CACHE_DIR, since compilation dominates tuning time.
- For each M, pick the best of: the new row, the current branch row, the H100 row for this key,
  and the default config. Judge them in benchmark mode (3 runs each).
- Keep a tuned row only where it beats the default by more than the noise band. Anywhere else,
  ship the config that is never slower.
- Watch the decode rows in particular: adjacent M values should not flip between very
  different tiles unless the measurements clearly say so.

Step 4 (optional): separate up/down tables from real routing
- Capture routing from a GLM-4.7-Flash server using a throwaway local copy of sglang (never
  committed):
  - after `topk_output = self.topk(...)` in glm4_moe_lite.py, TP-rank 0 does
    torch.save(topk_output.topk_ids.cpu()), keeping the shared-expert column;
  - save only when hidden_states.shape[0] >= 8192, as topk_ids_layer{layer_id}_idx{j}.pt, for
    j < ceil(100 / MoE layers);
  - serve with --disable-radix-cache, --chunked-prefill-size 8192 and
    `--cuda-graph-backend-prefill disabled` (this data-collection run only; Python does not
    run inside a replayed graph);
  - drive it with benchmark/kernels/fused_moe_triton/tuning_client.py (real text).
- Then run tuning_fused_moe_triton_sep.py --topk-ids-dir <dir> --tune. It only accepts a single
  --batch-size, and single-size mode does not write files, so first add a --batch-sizes flag
  (mirror the plain tuner) to cover all 20 rows.
- It writes the main table and a `_down` table (same BLOCK_SIZE_M per row; USE_TMA only where it
  measured faster). Ship this pair only if it beats Step 3's table in the Step 5 A/B.

Step 5: serving A/B (1 GPU)
- Launch with the cookbook's H200 command, e.g.
  `python3 -m sglang.launch_server --model-path zai-org/GLM-4.7-Flash --tp 1 --attention-backend triton --port 30000`.
  Do not add --disable-shared-experts-fusion.
- STOCK: no SGLANG_MOE_CONFIG_DIR. TUNED: SGLANG_MOE_CONFIG_DIR as in Step 2.
- Start a fresh server for every run, in the order A B A B A B, and wait for HTTP 200 on
  /health before any request.
- Decode:
  `python3 -m sglang.benchmark.one_batch_server --model None --base-url http://127.0.0.1:30000 --batch-size 1 8 16 32 64 128 --input-len 128 --output-len 128`
- Prefill: the same with `--batch-size 1 --input-len 512 2048 4096 8192 --output-len 1`.
- Realistic serving: `python3 -m sglang.bench_serving` with the random dataset, e.g. input 1024 /
  output 1024 at concurrency 16 and 64 (check the flag names).
- Prove the table loaded. The TUNED log must show "Using MoE kernel config from
  .../triton_3_7_1/E=65,N=1536,device_name=NVIDIA_H200.json", with no "Using default MoE
  kernel config" or "Fallback to triton version" for this key. Both logs must print identical
  server_args.
- Decode check: a different tile can change rounding, flip a greedy token and send later
  steps to different experts, which moves speed with no kernel change. Send the same fixed
  prompts at temperature 0 to /generate on both arms, compare the output token ids, and trust
  decode deltas only where they match.
- Optional: repeat with the cookbook's speculative-decoding command. The draft layer uses the
  same key, and verify steps run at small M.
- Expectation: with --attention-backend triton, long prefill is mostly attention time. Expect
  the MoE table to help decode and short prefill most.

Step 6: accuracy (1 GPU)
- GSM8K per arm, in A B A order:
  `python3 benchmark/gsm8k/bench_sglang.py --num-questions 1319 --parallel 128 --port 30000`

Step 7 (optional): more coverage
- E=64,N=1536 for the --disable-shared-experts-fusion posture, with the same Steps 2-6.
- Other TP sizes only if H200 users run them; the cookbook's H200 posture is TP1.

Step 8: package the PR
- Commits: the table (plus the `_down` table if Step 4 won); the tuner fix from Step 1 as a
  separate commit; the --batch-sizes flag for the sep tuner if you added it.
- Run `python3 -m pytest test/registered/unit/layers/moe/test_fused_moe_triton_config.py -q`
  and pre-commit on the changed files.
- The PR description, modeled on #35374 and #34150, should cover:
  - environment: GPU, torch, Triton, sglang commit;
  - method: tuner, routing including the shared-expert column, search space, M list,
    repeats;
  - a kernel before/after table per M, with the noise band;
  - the serving A/B tables;
  - GSM8K;
  - what is not covered.

ACCEPTANCE CRITERIA
- Kernel: no row slower than the default config by more than the noise band (1% if unclear), and
  a clear gain somewhere in M 1-64 (decode) or 512-8192 (prefill).
- Serving: no cell slower than the STOCK arm's own round-to-round spread (2% if unclear); report
  every cell's median for both arms.
- Accuracy: GSM8K within about 1.5 points of STOCK, with 0 invalid answers.
- The TUNED log proves the table loaded.

GPUS NEEDED
Every step fits on one H200. Eight GPUs only make the re-tune in Step 3 (and Step 4) faster.
Budget roughly an hour for Steps 2, 5 and 6 together. Tuning time depends on the search space and
the cache: time the first run before planning the rest.
```
