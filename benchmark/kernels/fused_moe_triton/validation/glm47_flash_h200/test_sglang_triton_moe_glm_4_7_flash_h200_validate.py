"""CPU tests for our validator. No SGLang installation or CUDA execution needed.

Run: python3 -m unittest -v test_sglang_triton_moe_glm_4_7_flash_h200_validate
Set SGLANG_REPO to additionally compare routing with the upstream reference.
"""

import ast
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

import sglang_triton_moe_glm_4_7_flash_bf16_tp1_h200_validate as v

try:
    import torch
except ImportError:
    torch = None


MODEL = {
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
    "dtype": "bfloat16",
    "routed_scaling_factor": 1.8,
}
TABLE = {
    str(m): {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 16,
        "num_warps": 4,
        "num_stages": 2,
    }
    for m in v.M_VALUES
}


@unittest.skipIf(torch is None, "PyTorch is needed for CPU tensor tests")
class RoutingTests(unittest.TestCase):
    def test_all_twenty_m_shapes_have_exactly_one_shared_column(self):
        for m in v.M_VALUES:
            with self.subTest(M=m):
                generator = torch.Generator(device="cpu").manual_seed(m)
                logits = torch.randn(m, 64, generator=generator, device="cpu")
                ids, weights = v.make_shared_routing(logits, 1.8)
                v.check_routing(ids, weights, 1.8)
                self.assertEqual(tuple(ids.shape), (m, 5))
                self.assertEqual(int((ids == 64).sum()), m)
                self.assertEqual(ids.dtype, torch.int32)
                self.assertEqual(weights.dtype, torch.float32)

    def test_shared_expert_is_not_in_random_selection(self):
        logits = torch.arange(64, dtype=torch.float32).div(16).reshape(1, 64)
        ids, weights = v.make_shared_routing(logits, 1.8)
        self.assertEqual(ids.tolist(), [[63, 62, 61, 60, 64]])
        self.assertAlmostEqual(float(weights[0, 4] * 1.8), 1.0, places=6)
        with self.assertRaisesRegex(ValueError, "only experts"):
            v.make_shared_routing(torch.zeros(1, 65), 1.8)

    def test_every_update_in_a_sample_batch_is_checked(self):
        ids, weights = v.make_shared_routing(torch.randn(10, 16, 64), 1.8)
        v.check_routing(ids, weights, 1.8)
        ids[9, 15, 4] = 7
        with self.assertRaisesRegex(ValueError, "Every token"):
            v.check_routing(ids, weights, 1.8)

    def test_duplicate_or_wrong_routed_id_is_rejected(self):
        ids, weights = v.make_shared_routing(torch.randn(2, 64), 1.8)
        ids[0, 1] = ids[0, 0]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            v.check_routing(ids, weights, 1.8)

    def test_tensor_fingerprint_covers_values_and_handles_bf16(self):
        a = torch.arange(64, dtype=torch.bfloat16).reshape(2, 32)
        b = a.clone()
        self.assertEqual(v.tensor_hash(a), v.tensor_hash(b))
        b[0, 0] = 1
        self.assertNotEqual(v.tensor_hash(a), v.tensor_hash(b))

    def test_matches_upstream_reference_without_importing_sglang(self):
        repo = os.environ.get("SGLANG_REPO")
        if not repo:
            self.skipTest("Set SGLANG_REPO for the upstream reference cross-check")
        source = Path(repo) / "python/sglang/srt/layers/moe/topk.py"
        module = ast.parse(source.read_text())
        function = next(
            n
            for n in module.body
            if isinstance(n, ast.FunctionDef) and n.name == "biased_grouped_topk_impl"
        )
        function.decorator_list = []  # Execute the CPU reference, not torch.compile.
        namespace = {
            "torch": torch,
            "Optional": Optional,
            "_RENORMALIZE_SUM_EPSILON": 1e-20,
        }
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        reference = namespace[function.name]
        for m in [1, 16, 128]:
            generator = torch.Generator(device="cpu").manual_seed(73 + m)
            logits = torch.randn(m, 64, generator=generator)
            ids, weights = v.make_shared_routing(logits, 1.8)
            expected_weights, expected_ids = reference(
                torch.empty(m, 2048),
                logits,
                torch.zeros(64),
                topk=5,
                renormalize=True,
                num_expert_group=1,
                topk_group=1,
                num_fused_shared_experts=1,
                routed_scaling_factor=1.8,
            )
            torch.testing.assert_close(ids, expected_ids)
            torch.testing.assert_close(weights, expected_weights)


class FilesAndResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="glm47-validator-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        configs = self.repo / v.CONFIG_REL
        for version in ["triton_3_5_1", "triton_3_7_1"]:
            v.atomic_json(configs / version / v.FILENAME, TABLE)
            v.atomic_json(configs / version / v.DOWN_FILENAME, TABLE)
            v.atomic_json(configs / version / "unrelated.json", {"preserve": True})
        self.model = self.root / "config.json"
        self.candidate = self.root / "candidate.json"
        v.atomic_json(self.model, MODEL)
        v.atomic_json(self.candidate, TABLE)
        self.args = SimpleNamespace(
            sglang_repo=self.repo,
            output=self.root / "results",
            model_config=self.model,
            candidate=self.candidate,
            rounds=3,
            iterations=10,
            seed=0,
            batch_sizes=[1, 16],
            check_only=False,
        )

    def prepare(self):
        with (
            patch.object(v, "source_revision", return_value="test-revision"),
            redirect_stdout(io.StringIO()),
        ):
            return v.prepare_run(self.args)

    def test_stock_has_no_current_or_fallback_key_and_source_is_untouched(self):
        before = v.tree_hash(self.repo / v.CONFIG_REL)
        manifest = self.prepare()
        self.assertEqual(len(manifest["baseline_excluded_paths"]), 4)
        for path in (self.args.output / "stock").rglob("*.json"):
            self.assertNotIn(path.name, [v.FILENAME, v.DOWN_FILENAME])
        self.assertEqual(v.tree_hash(self.repo / v.CONFIG_REL), before)
        self.assertEqual(
            v.file_hash(self.candidate),
            v.file_hash(self.args.output / "tuned/configs/triton_3_7_1" / v.FILENAME),
        )
        self.assertTrue(
            (self.args.output / "stock/configs/triton_3_5_1/unrelated.json").exists()
        )

    def test_same_run_resumes_but_changed_settings_are_rejected(self):
        manifest = self.prepare()
        self.assertEqual(self.prepare(), manifest)
        self.args.seed = 99
        with self.assertRaisesRegex(ValueError, "NEW output"):
            self.prepare()

    def test_modified_override_cannot_silently_change_a_resumed_run(self):
        self.prepare()
        v.atomic_json(
            self.args.output / "stock/configs/triton_3_7_1" / v.FILENAME, TABLE
        )
        with self.assertRaisesRegex(ValueError, "config copy changed"):
            self.prepare()

    def test_old_tuning_markers_cannot_be_used_as_validation_results(self):
        v.atomic_json(self.args.output / "run-info.json", {"topk": 4})
        with self.assertRaisesRegex(ValueError, "Old tuning checkpoints"):
            self.prepare()

    def test_check_only_does_not_create_output(self):
        self.args.check_only = True
        self.assertIsNone(self.prepare())
        self.assertFalse(self.args.output.exists())

    def test_wrong_model_and_incomplete_table_are_rejected(self):
        with self.assertRaises(ValueError):
            v.check_model({**MODEL, "n_shared_experts": 0})
        with self.assertRaises(ValueError):
            v.check_candidate({"1": TABLE["1"]})

    def test_partial_summary_and_mismatched_routes(self):
        manifest = self.prepare()
        for arm, latency in [("stock", 10.0), ("tuned", 8.0)]:
            result = {
                "run_id": manifest["run_id"],
                "round": 1,
                "arm": arm,
                "M": 1,
                "samples_us": [latency] * 10,
                "seed": 100001,
                "input_sha256": "same-input",
                "routing_sha256": "same-routing",
            }
            v.atomic_json(v.result_path(self.args.output, 1, arm, 1), result)
        with redirect_stdout(io.StringIO()):
            v.summarize(self.args.output)
        rows = v.read_json(self.args.output / "summary.json")
        self.assertEqual(rows[0]["status"], "incomplete")
        self.assertEqual(rows[0]["speedup"], 1.25)
        self.assertEqual(rows[1]["status"], "incomplete")
        path = v.result_path(self.args.output, 1, "tuned", 1)
        result = v.read_json(path)
        result["routing_sha256"] = "different-routing"
        v.atomic_json(path, result)
        with self.assertRaisesRegex(ValueError, "Unmatched routing"):
            v.summarize(self.args.output)

    def test_full_summary_is_computed_from_raw_samples(self):
        manifest = self.prepare()
        for m in self.args.batch_sizes:
            for r in range(1, 4):
                for arm, latency in [("stock", 10.0), ("tuned", 8.0)]:
                    result = {
                        "run_id": manifest["run_id"],
                        "round": r,
                        "arm": arm,
                        "M": m,
                        "samples_us": [latency] * 10,
                        "median_us": 9999,
                        "seed": r * 100000 + m,
                        "input_sha256": f"input-{r}-{m}",
                        "routing_sha256": f"routing-{r}-{m}",
                    }
                    v.atomic_json(v.result_path(self.args.output, r, arm, m), result)
        with redirect_stdout(io.StringIO()):
            v.summarize(self.args.output)
        for row in v.read_json(self.args.output / "summary.json"):
            self.assertEqual(row["status"], "faster")
            self.assertEqual(row["stock_median_us"], 10)
            self.assertEqual(row["tuned_median_us"], 8)
            self.assertEqual(row["speedup"], 1.25)
            self.assertEqual(row["latency_reduction_pct"], 20)

    def test_resume_rejects_a_record_from_a_different_run(self):
        manifest = self.prepare()
        path = v.result_path(self.args.output, 1, "stock", 1)
        v.atomic_json(
            path,
            {
                "run_id": "old-top4-run",
                "round": 1,
                "arm": "stock",
                "M": 1,
                "samples_us": [10.0] * 10,
            },
        )
        with self.assertRaisesRegex(ValueError, "Wrong run_id"):
            v.load_result(path, manifest, 1, "stock", 1)


class SummaryTests(unittest.TestCase):
    def test_gain_regression_and_noise(self):
        self.assertEqual(v.compare_times([10] * 3, [8] * 3, True)["status"], "faster")
        self.assertEqual(
            v.compare_times([10] * 3, [12] * 3, True)["status"], "regression"
        )
        self.assertEqual(
            v.compare_times([10] * 3, [9.95] * 3, True)["status"], "within_noise"
        )
        self.assertEqual(
            v.compare_times([9, 10, 11], [9, 9, 9], True)["status"], "within_noise"
        )
        self.assertEqual(v.compare_times([10], [8], False)["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
