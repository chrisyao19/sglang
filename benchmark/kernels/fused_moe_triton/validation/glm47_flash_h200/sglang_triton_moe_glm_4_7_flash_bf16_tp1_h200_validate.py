#!/usr/bin/env python3
"""Compare the existing GLM H200 table with the heuristic; never search configs.

Synthetic BF16/TP1 inputs: four routed experts among 0..63, plus shared expert
64 on every token. This is not captured model routing or a serving benchmark.
SGLang's normal up/down config lookup and fused_moe implementation are used
without patching SGLang or overriding its config selection in Python.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

M_VALUES = [
    1,
    2,
    4,
    8,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    256,
    512,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
]
FILENAME = "E=65,N=1536,device_name=NVIDIA_H200.json"
DOWN_FILENAME = FILENAME.replace(".json", "_down.json")
CONFIG_REL = Path("python/sglang/srt/layers/moe/moe_runner/triton_utils/configs")
TRITON_VERSION = "3.7.1"
SCHEMA = "glm47-h200-shared-4plus1-v1"
FIELDS = {
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "num_warps",
    "num_stages",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def tree_hash(root):
    return object_hash(
        {
            p.relative_to(root).as_posix(): file_hash(p)
            for p in sorted(Path(root).rglob("*.json"))
        }
    )


def check_model(model):
    expected = {
        "architectures": ["Glm4MoeLiteForCausalLM"],
        "n_routed_experts": 64,
        "n_shared_experts": 1,
        "num_experts_per_tok": 4,
        "hidden_size": 2048,
        "moe_intermediate_size": 1536,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "hidden_act": "silu",
    }
    for key, value in expected.items():
        require(model.get(key) == value, f"Unsupported model {key}: {model.get(key)!r}")
    require(
        model.get("dtype", model.get("torch_dtype")) == "bfloat16",
        "This validator only supports the BF16 checkpoint.",
    )
    require(
        not model.get("quantization_config"), "Quantized checkpoints are unsupported."
    )
    scale = model.get("routed_scaling_factor")
    require(
        isinstance(scale, (int, float)) and math.isfinite(scale) and scale > 0,
        "Missing/invalid routed_scaling_factor.",
    )


def check_candidate(table):
    require(
        isinstance(table, dict) and set(table) == {str(m) for m in M_VALUES},
        "Candidate must contain exactly the original 20 M rows.",
    )
    for m, row in table.items():
        require(
            isinstance(row, dict) and set(row) == FIELDS,
            f"M={m}: expected the six plain-tuner fields (no TMA/down table).",
        )
        require(
            all(type(v) is int and v > 0 for v in row.values()),
            f"M={m}: all config values must be positive integers.",
        )


def make_shared_routing(logits, scaling_factor):
    """Zero-bias, one-group sigmoid routing; works on CPU or CUDA tensors.

    Matches GLM's normalized routed weights plus shared weight 1/scale.
    The kernel receives routed_scaling_factor=scale, so the shared expert's
    effective weight is one. Logits have 64 columns, never 65.
    """
    import torch

    require(logits.shape[-1] == 64, "Routing logits must cover only experts 0..63.")
    require(
        math.isfinite(scaling_factor) and scaling_factor > 0, "Invalid scaling factor."
    )
    scores = logits.float().sigmoid()
    routed_weights, routed_ids = torch.topk(scores, 4, dim=-1, sorted=True)
    total = routed_weights.sum(dim=-1, keepdim=True)
    shared_ids = torch.full_like(routed_ids[..., :1], 64)
    ids = torch.cat((routed_ids, shared_ids), dim=-1).to(torch.int32)
    weights = torch.cat((routed_weights, total / scaling_factor), dim=-1)
    return ids, (weights / (total + 1e-20)).to(torch.float32)


def check_routing(ids, weights, scaling_factor):
    import torch

    require(ids.shape == weights.shape and ids.shape[-1] == 5, "Expected [..., M, 5].")
    require(
        bool((ids[..., -1] == 64).all()), "Every token must visit shared expert 64."
    )
    require(
        bool(((ids[..., :4] >= 0) & (ids[..., :4] < 64)).all()),
        "Routed columns must exclude the shared expert.",
    )
    ordered = ids[..., :4].sort(dim=-1).values
    require(
        bool((ordered[..., 1:] != ordered[..., :-1]).all()), "Duplicate routed experts."
    )
    require(bool(torch.isfinite(weights).all()), "Non-finite routing weights.")
    torch.testing.assert_close(
        weights[..., :4].sum(-1),
        torch.ones_like(weights[..., -1]),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        weights[..., -1],
        torch.full_like(weights[..., -1], 1 / scaling_factor),
        rtol=1e-6,
        atol=1e-6,
    )


def prepare_config_trees(source, output, candidate):
    """Copy all unrelated configs; omit this key in ALL versions in the A arm.

    The input checkout may already be the feature branch containing our table.
    Removing it only in a private copy prevents baseline contamination and
    prevents a different-version up/down fallback from entering either arm.
    """
    omitted = sorted(
        p.relative_to(source).as_posix()
        for p in Path(source).rglob("*.json")
        if p.name in {FILENAME, DOWN_FILENAME}
    )
    stock = output / "stock" / "configs"
    tuned = output / "tuned" / "configs"
    shutil.copytree(
        source, stock, ignore=shutil.ignore_patterns(FILENAME, DOWN_FILENAME)
    )
    shutil.copytree(stock, tuned)
    target = tuned / "triton_3_7_1" / FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate, target)
    return omitted


def source_revision(repo):
    marker = repo / ".sglang-source-commit"
    if marker.is_file():
        return marker.read_text().strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    require(
        not dirty.strip(), "Use a checkout with no tracked edits for reproducible A/B."
    )
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def prepare_run(args):
    repo, root = args.sglang_repo.resolve(), args.output.resolve()
    model_path, candidate = args.model_config.resolve(), args.candidate.resolve()
    check_model(read_json(model_path))
    check_candidate(read_json(candidate))
    require(
        args.rounds >= 1 and args.iterations >= 10, "Need >=1 round and >=10 samples."
    )
    require(args.seed >= 0, "Seed must be non-negative.")
    require(
        len(set(args.batch_sizes)) == len(args.batch_sizes)
        and set(args.batch_sizes) <= set(M_VALUES),
        "Invalid/duplicate M values.",
    )
    require(
        (repo / CONFIG_REL).is_dir(), "SGLANG_REPO does not contain the config tree."
    )
    signature = {
        "schema": SCHEMA,
        "sglang_repo": str(repo),
        "sglang_commit": source_revision(repo),
        "script_sha256": file_hash(__file__),
        "model_sha256": file_hash(model_path),
        "candidate_sha256": file_hash(candidate),
        "source_configs_sha256": tree_hash(repo / CONFIG_REL),
        "batch_sizes": args.batch_sizes,
        "rounds": args.rounds,
        "iterations": args.iterations,
        "seed": args.seed,
        "triton": TRITON_VERSION,
        "timing": "one-invocation CUDA graph; restore input and flush 256MiB L2 before EACH sample",
        "routing": "synthetic sigmoid, zero correction bias, 4 routed + fixed shared 64",
    }
    if args.check_only:
        print(json.dumps(signature, indent=2))
        print(
            "Input checks passed. No GPU test was run and no output directory was created."
        )
        return None
    manifest_path = root / "run-manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        require(
            manifest["signature"] == signature,
            "Run inputs/settings changed. Use a NEW output directory; keep the old results.",
        )
        verify_copies(root, manifest)
        return manifest
    require(
        not root.exists() or not any(root.iterdir()),
        "Use a new/empty output directory. Old tuning checkpoints cannot be resumed here.",
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "model-config").mkdir()
    shutil.copy2(model_path, root / "model-config" / "config.json")
    omitted = prepare_config_trees(repo / CONFIG_REL, root, candidate)
    manifest = {
        "signature": signature,
        "run_id": object_hash(signature),
        "baseline_excluded_paths": omitted,
        "tree_hashes": {
            arm: tree_hash(root / arm / "configs") for arm in ("stock", "tuned")
        },
    }
    atomic_json(manifest_path, manifest)
    print(
        "Baseline excludes this key in every Triton version; source files are untouched."
    )
    print("Excluded source paths:", omitted or "none")
    return manifest


def verify_copies(root, manifest):
    require(
        file_hash(root / "model-config" / "config.json")
        == manifest["signature"]["model_sha256"],
        "Copied model config changed; use a new output directory.",
    )
    for arm in ("stock", "tuned"):
        require(
            tree_hash(root / arm / "configs") == manifest["tree_hashes"][arm],
            f"{arm} config copy changed; use a new output directory.",
        )


def result_path(root, round_index, arm, m):
    return root / "measurements" / f"round-{round_index:02d}-{arm}-M={m}.json"


def load_result(path, manifest, round_index, arm, m):
    result = read_json(path)
    for key, expected in {
        "run_id": manifest["run_id"],
        "round": round_index,
        "arm": arm,
        "M": m,
    }.items():
        require(result.get(key) == expected, f"Wrong {key} in {path}")
    samples = result["samples_us"]
    require(
        len(samples) == manifest["signature"]["iterations"]
        and all(math.isfinite(x) and x > 0 for x in samples),
        f"Invalid samples in {path}",
    )
    return result


def tensor_hash(*tensors):
    import torch

    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(
            tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        )
    return digest.hexdigest()


def measure_m(
    torch, fused_moe, runner, TopKOutput, x_base, w1, w2, logits, ids, weights
):
    """Route/input updates and the L2 flush are outside the timed CUDA events."""
    x = x_base.clone()
    topk = TopKOutput(weights[0].clone(), ids[0].clone(), logits[0].clone())
    flush_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    def prepare(i):
        # fused_moe is in-place: restore input so outputs are never fed back in.
        x.copy_(x_base)
        topk.topk_ids.copy_(ids[i])
        topk.topk_weights.copy_(weights[i])
        topk.router_logits.copy_(logits[i])

    def run():
        fused_moe(x, w1, w2, topk, moe_runner_config=runner)

    # Compile on a side stream before graph capture, as recommended by PyTorch.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            prepare(0)
            run()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    prepare(0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()  # ONE invocation, not ten repeated uses of the same routed weights.
    for _ in range(5):
        prepare(0)
        graph.replay()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(len(ids))]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(len(ids))]
    for i, (start, end) in enumerate(zip(starts, ends)):
        prepare(i)
        flush_buffer.zero_()
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()
    require(bool(torch.isfinite(x).all()), "Non-finite MoE output; timing rejected.")
    samples = [start.elapsed_time(end) * 1000 for start, end in zip(starts, ends)]
    graph.reset()
    return samples


def worker(args):
    root = args.output.resolve()
    manifest = read_json(root / "run-manifest.json")
    sig = manifest["signature"]
    require(
        file_hash(__file__) == sig["script_sha256"], "Validator changed during the run."
    )
    require(1 <= args.round_index <= sig["rounds"], "Invalid round.")
    verify_copies(root, manifest)
    repo = Path(sig["sglang_repo"])
    require(source_revision(repo) == sig["sglang_commit"], "SGLang source changed.")
    os.environ["SGLANG_MOE_CONFIG_DIR"] = str(root / args.arm)
    sys.path.insert(0, str(repo / "python"))
    import torch
    import triton

    require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "Expose exactly one otherwise idle H200 with CUDA_VISIBLE_DEVICES.",
    )
    require(
        torch.cuda.get_device_name(0) == "NVIDIA H200",
        "This table requires NVIDIA H200 (not NVL/H100).",
    )
    require(
        triton.__version__ == TRITON_VERSION,
        "Reproduce Triton 3.7.1; do not rename its table directory.",
    )
    require(
        torch.__version__.startswith("2.13.0") and torch.version.cuda == "13.0",
        "Reproduce the original PyTorch 2.13.0 / CUDA 13.0 environment.",
    )
    torch.cuda.set_device(0)
    import sglang
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.triton_utils import (
        fused_moe_triton_config as cfg,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils import (
        get_config,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
    from sglang.srt.layers.moe.topk import (
        StandardTopKOutput,
        TopKConfig,
        TopKOutputFormat,
        select_experts,
    )
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    require(
        Path(sglang.__file__).resolve().is_relative_to(repo / "python"),
        "Wrong SGLang installation.",
    )
    model = read_json(root / "model-config" / "config.json")
    check_model(model)
    server_args = ServerArgs(
        model_path=str(root / "model-config"),
        tp_size=1,
        ep_size=1,
        dtype="bfloat16",
        moe_runner_backend="triton",
    )
    set_global_server_args_for_scheduler(server_args)
    require(get_config() is None, "Unexpected Python config override.")
    require(
        cfg.get_config_file_name(65, 1536, None) == FILENAME,
        "Runtime generated a different lookup key.",
    )
    cfg.get_moe_configs.cache_clear()
    up_table = cfg.get_moe_configs(65, 1536, None)
    down_table = cfg.get_moe_configs(65, 1536, None, down_moe=True)
    expected = read_json(root / "tuned" / "configs" / "triton_3_7_1" / FILENAME)
    expected = {int(m): row for m, row in expected.items()}
    if args.arm == "stock":
        require(
            up_table is None and down_table is None,
            "Baseline is contaminated by a tuning table.",
        )
    else:
        require(
            up_table == expected and down_table == expected,
            "Tuned runtime did not load the candidate for both projections.",
        )
    print(
        f"Verified {args.arm} lookup: {os.environ['SGLANG_MOE_CONFIG_DIR']}", flush=True
    )
    snapshot = subprocess.run(
        ["nvidia-smi"], capture_output=True, text=True, check=True
    )
    print(snapshot.stdout, flush=True)
    environment = {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "python": sys.version,
        "sglang_file": str(sglang.__file__),
    }
    for package in ("sglang-kernel", "sgl-kernel", "flashinfer-python"):
        try:
            environment[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            environment[package] = None
    driver = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={environment['gpu_uuid']}",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    environment["driver"] = driver
    env_path = root / "environment.json"
    if env_path.exists():
        require(
            read_json(env_path) == environment,
            "GPU/software changed; start a new comparison directory.",
        )
    else:
        atomic_json(env_path, environment)

    scale = float(model["routed_scaling_factor"])
    runner = MoeRunnerConfig(
        num_experts=65,
        num_local_experts=65,
        top_k=5,
        num_fused_shared_experts=1,
        inplace=True,
        routed_scaling_factor=scale,
    )
    weight_generator = torch.Generator(device="cuda").manual_seed(sig["seed"])
    w1 = (
        torch.randn(
            65,
            3072,
            2048,
            dtype=torch.bfloat16,
            device="cuda",
            generator=weight_generator,
        )
        * 0.02
    )
    w2 = (
        torch.randn(
            65,
            2048,
            1536,
            dtype=torch.bfloat16,
            device="cuda",
            generator=weight_generator,
        )
        * 0.02
    )
    for m in sig["batch_sizes"]:
        target = result_path(root, args.round_index, args.arm, m)
        if target.exists():
            load_result(target, manifest, args.round_index, args.arm, m)
            print(
                f"Resume: keep completed round={args.round_index} {args.arm} M={m}",
                flush=True,
            )
            continue
        started = time.monotonic()
        # Same seed per (round, M) in both arms, independent of skipped rows.
        seed = sig["seed"] + args.round_index * 100_000 + m
        generator = torch.Generator(device="cuda").manual_seed(seed)
        x_base = torch.randn(
            m, 2048, dtype=torch.bfloat16, device="cuda", generator=generator
        )
        logits = torch.randn(
            sig["iterations"],
            m,
            64,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        ids, weights = make_shared_routing(logits, scale)
        check_routing(ids, weights, scale)
        # Check the initial routing against SGLang's actual selector, with a
        # zero synthetic correction bias (we do not load learned model weights).
        topk_config = TopKConfig(
            top_k=5,
            use_grouped_topk=True,
            topk_group=1,
            num_expert_group=1,
            num_fused_shared_experts=1,
            correction_bias=torch.zeros(64, device="cuda"),
            routed_scaling_factor=scale,
            output_format=TopKOutputFormat.STANDARD,
        )
        actual = select_experts(x_base, logits[0], topk_config)
        own_order = ids[0].argsort(dim=-1)
        actual_order = actual.topk_ids.argsort(dim=-1)
        torch.testing.assert_close(
            ids[0].gather(-1, own_order), actual.topk_ids.gather(-1, actual_order)
        )
        torch.testing.assert_close(
            weights[0].gather(-1, own_order),
            actual.topk_weights.gather(-1, actual_order),
            rtol=1e-5,
            atol=1e-6,
        )
        print(
            f"M={m}: topk shape=({m}, 5), shared column=[64], shared visits={m}; selector check passed",
            flush=True,
        )
        input_digest = tensor_hash(x_base)
        routing_digest = tensor_hash(ids, weights)
        resolved_up, (resolved_down, _) = cfg.try_get_optimal_moe_config(
            tuple(w1.shape), tuple(w2.shape), 5, None, m, return_down_config=True
        )
        samples = measure_m(
            torch,
            fused_moe,
            runner,
            StandardTopKOutput,
            x_base,
            w1,
            w2,
            logits,
            ids,
            weights,
        )
        result = {
            "run_id": manifest["run_id"],
            "round": args.round_index,
            "arm": args.arm,
            "M": m,
            "seed": seed,
            "input_sha256": input_digest,
            "routing_sha256": routing_digest,
            "topk_shape": [m, 5],
            "shared_expert": 64,
            "up_config": resolved_up,
            "down_config": resolved_down or resolved_up,
            "samples_us": samples,
            "median_us": statistics.median(samples),
            "mean_us": statistics.mean(samples),
            "wall_seconds": time.monotonic() - started,
        }
        atomic_json(target, result)
        print(
            f"round={args.round_index} {args.arm} M={m}: {result['median_us']:.3f} us; saved {target.name}",
            flush=True,
        )
        del x_base, logits, ids, weights, actual


def compare_times(stock, tuned, complete):
    if not stock or not tuned:
        return {"status": "incomplete"}
    a, b = statistics.median(stock), statistics.median(tuned)
    spread_a, spread_b = (
        100 * (max(stock) - min(stock)) / a,
        100 * (max(tuned) - min(tuned)) / b,
    )
    reduction = 100 * (a - b) / a
    band = max(1.0, spread_a, spread_b)
    status = "incomplete"
    if complete:
        status = (
            "faster"
            if reduction > band
            else "regression"
            if reduction < -band
            else "within_noise"
        )
    return {
        "stock_median_us": a,
        "tuned_median_us": b,
        "speedup": a / b,
        "latency_reduction_pct": reduction,
        "stock_spread_pct": spread_a,
        "tuned_spread_pct": spread_b,
        "noise_band_pct": band,
        "status": status,
    }


def summarize(root):
    root = root.resolve()
    manifest = read_json(root / "run-manifest.json")
    sig = manifest["signature"]
    rows = []
    for m in sig["batch_sizes"]:
        times = {"stock": [], "tuned": []}
        for r in range(1, sig["rounds"] + 1):
            pair = {}
            for arm in times:
                path = result_path(root, r, arm, m)
                if path.exists():
                    pair[arm] = load_result(path, manifest, r, arm, m)
                    times[arm].append(statistics.median(pair[arm]["samples_us"]))
            if len(pair) == 2:
                for field in ("seed", "input_sha256", "routing_sha256"):
                    require(
                        pair["stock"][field] == pair["tuned"][field],
                        f"Unmatched {field} in round {r}, M={m}; do not compare these samples.",
                    )
        complete = sig["rounds"] >= 3 and all(
            len(v) == sig["rounds"] for v in times.values()
        )
        rows.append(
            {
                "M": m,
                "stock_round_us": times["stock"],
                "tuned_round_us": times["tuned"],
                **compare_times(times["stock"], times["tuned"], complete),
            }
        )
    atomic_json(root / "summary.json", rows)
    fields = [
        "M",
        "stock_round_us",
        "tuned_round_us",
        "stock_median_us",
        "tuned_median_us",
        "speedup",
        "latency_reduction_pct",
        "stock_spread_pct",
        "tuned_spread_pct",
        "noise_band_pct",
        "status",
    ]
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("M     default us   candidate us   speedup   status")
    for row in rows:
        print(
            f"{row['M']:<5} {row.get('stock_median_us', float('nan')):>10.3f} "
            f"{row.get('tuned_median_us', float('nan')):>14.3f} "
            f"{row.get('speedup', float('nan')):>8.3f}   {row['status']}"
        )
    print(
        f"Saved {root / 'summary.csv'} (synthetic kernel measurements, not serving/accuracy evidence)."
    )


def run_comparison(args):
    manifest = prepare_run(args)
    if manifest is None:
        return
    root = args.output.resolve()
    (root / "logs").mkdir(exist_ok=True)
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("TRITON_CACHE_DIR", str(root / "triton-cache"))
    env["PYTHONUNBUFFERED"] = "1"
    for r in range(1, args.rounds + 1):
        for arm in ("stock", "tuned"):
            paths = [
                result_path(root, r, arm, m)
                for m in manifest["signature"]["batch_sizes"]
            ]
            if all(path.exists() for path in paths):
                for m, path in zip(manifest["signature"]["batch_sizes"], paths):
                    load_result(path, manifest, r, arm, m)
                print(
                    f"Resume: round {r} {arm} is already complete; no GPU process needed.",
                    flush=True,
                )
                continue
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "_worker",
                "--output",
                str(root),
                "--arm",
                arm,
                "--round-index",
                str(r),
            ]
            log_path = root / "logs" / f"round-{r:02d}-{arm}.log"
            print(f"Starting round {r}/{args.rounds}: {arm}; {log_path}", flush=True)
            with log_path.open("a", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                )
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    returncode = process.wait()
                except KeyboardInterrupt:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
            summarize(root)
            require(
                returncode == 0,
                f"Worker failed; see {log_path}. Completed M results are saved; rerun the same command to resume.",
            )
    summarize(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser(
        "run", help="Compare the heuristic and the existing table on one H200."
    )
    run.add_argument("--sglang-repo", type=Path, required=True)
    run.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Local config.json only; no weights.",
    )
    run.add_argument("--candidate", type=Path, required=True)
    run.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New results directory, or the SAME run to resume.",
    )
    run.add_argument("--batch-sizes", type=int, nargs="+", default=M_VALUES)
    run.add_argument("--rounds", type=int, default=3)
    run.add_argument("--iterations", type=int, default=100)
    run.add_argument("--seed", type=int, default=20260927)
    run.add_argument(
        "--check-only",
        action="store_true",
        help="Check inputs without importing CUDA/SGLang or writing output.",
    )
    summary = sub.add_parser(
        "summarize", help="Rebuild CSV from saved results; no GPU required."
    )
    summary.add_argument("--output", type=Path, required=True)
    child = sub.add_parser("_worker", help="Internal subprocess; use run instead.")
    child.add_argument("--output", type=Path, required=True)
    child.add_argument("--arm", choices=["stock", "tuned"], required=True)
    child.add_argument("--round-index", type=int, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        if args.command == "run":
            run_comparison(args)
        elif args.command == "summarize":
            summarize(args.output)
        else:
            worker(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
