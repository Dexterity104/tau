from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from swebench_crown_benchmark import (
    AgentIdentity,
    BenchmarkManifest,
    ScoreSummary,
    aggregate_usage,
    build_comparison,
    current_king_from_state,
    load_manifest,
    merge_benchmark_into_dashboard,
    mini_swe_agent_filter,
    predictions_include_patch,
    mini_swe_agent_prediction_patch,
    pi_baseline_cache_dir,
    pi_model_arg,
    read_pi_pin,
    reset_prediction_outputs,
    restore_pi_baseline,
    save_pi_baseline,
    queue_latest,
    should_benchmark_king,
    write_pi_pin,
)


class SwebenchCrownBenchmarkTest(unittest.TestCase):
    def test_detects_current_king_from_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.json"
            state_path.write_text(
                json.dumps({"current_king": {"commit_sha": "abc", "repo_full_name": "owner/repo"}}),
                encoding="utf-8",
            )

            king = current_king_from_state(state_path)

            self.assertEqual(king["commit_sha"], "abc")

    def test_should_skip_completed_king(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job_dir = root / "abc"
            pi_dir = job_dir / "pi"
            pi_dir.mkdir(parents=True)
            (job_dir / "job.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
            (job_dir / "comparison.json").write_text(json.dumps({"baseline_name": "pi"}), encoding="utf-8")
            (pi_dir / "predictions.jsonl").write_text(
                json.dumps({"instance_id": "a", "model_patch": "diff --git a/a b/a\n"}) + "\n",
                encoding="utf-8",
            )

            self.assertFalse(should_benchmark_king(king={"commit_sha": "abc"}, benchmark_root=root))
            self.assertTrue(should_benchmark_king(king={"commit_sha": "def"}, benchmark_root=root))

    def test_should_rerun_completed_empty_patch_baseline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline_dir = root / "abc" / "mini-swe-agent"
            baseline_dir.mkdir(parents=True)
            (root / "abc" / "job.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
            (root / "abc" / "comparison.json").write_text(
                json.dumps({"baseline_name": "mini-swe-agent"}),
                encoding="utf-8",
            )
            (baseline_dir / "predictions.jsonl").write_text(
                json.dumps({"instance_id": "a", "model_patch": ""}) + "\n",
                encoding="utf-8",
            )

            self.assertTrue(should_benchmark_king(king={"commit_sha": "abc"}, benchmark_root=root))

    def test_queue_latest_replaces_pending_king(self):
        pending = {"commit_sha": "old"}
        candidate = {"commit_sha": "new"}

        self.assertEqual(queue_latest(pending, candidate), candidate)
        self.assertEqual(queue_latest(candidate, {"commit_sha": "new"}), candidate)

    def test_manifest_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "dataset_name": "dataset",
                        "split": "test",
                        "seed": 66,
                        "count": 2,
                        "instance_ids": ["a", "a"],
                    },
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_manifest(manifest_path)

    def test_usage_cost_availability(self):
        usage = aggregate_usage(
            [
                {"request_count": 1, "total_tokens": 10, "cost": 0.0, "requests": [{"cost": None}]},
                {"request_count": 2, "total_tokens": 20, "cost": 0.0, "requests": []},
            ],
        )

        self.assertEqual(usage["request_count"], 3)
        self.assertEqual(usage["total_tokens"], 30)
        self.assertIsNone(usage["cost"])
        self.assertFalse(usage["cost_available"])

    def test_comparison_summary_computes_delta(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            job_dir = Path(temp_dir)
            (job_dir / "job.json").write_text(json.dumps({"started_at": "start"}), encoding="utf-8")
            for agent in ("king", "pi"):
                agent_dir = job_dir / agent
                agent_dir.mkdir()
                (agent_dir / "usage_summary.json").write_text(
                    json.dumps({"cost_available": False, "cost": None}),
                    encoding="utf-8",
                )
                (agent_dir / "solve_results.jsonl").write_text(
                    json.dumps({"elapsed_seconds": 1.5}) + "\n",
                    encoding="utf-8",
                )
            manifest = BenchmarkManifest(
                dataset_name="dataset",
                split="test",
                seed=66,
                instance_ids=("a", "b"),
                manifest_hash="hash",
            )
            king_agent = AgentIdentity("king", "repo", "king-sha", Path("/tmp/king.py"), {})
            pi_agent = AgentIdentity("pi", "repo", "pi-sha", Path("/tmp/pi.py"), {})

            comparison = build_comparison(
                king={"commit_sha": "king-sha"},
                king_agent=king_agent,
                baseline_agent=pi_agent,
                manifest=manifest,
                job_dir=job_dir,
                scores=(
                    ScoreSummary(2, 2, 1.0, "king-report"),
                    ScoreSummary(1, 2, 0.5, "pi-report"),
                ),
                total_elapsed_seconds=10,
                model="model",
                provider_only="provider",
                baseline_cached=False,
            )

            self.assertEqual(comparison["scores"]["delta_pass_rate"], 0.5)
            self.assertEqual(comparison["baseline_name"], "pi")
            self.assertFalse(comparison["pi_baseline_cached"])
            self.assertFalse(comparison["usage"]["cost_available"])

    def test_pi_baseline_cache_restores_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = BenchmarkManifest("dataset", "test", 66, ("a",), "manifest-hash")
            pi_agent = AgentIdentity("pi", "repo", "pi-sha", root / "agent.py", {})
            job_dir = root / "job"
            pi_dir = job_dir / "pi"
            scoring_dir = pi_dir / "official_scoring"
            scoring_dir.mkdir(parents=True)
            for name in ("predictions.jsonl", "solve_results.jsonl", "usage_summary.json", "score_summary.json"):
                (pi_dir / name).write_text("{}\n", encoding="utf-8")
            (pi_dir / "predictions.jsonl").write_text(
                json.dumps({"instance_id": "a", "model_patch": "diff --git a/a b/a\n"}) + "\n",
                encoding="utf-8",
            )
            (scoring_dir / "report.json").write_text("{}", encoding="utf-8")

            cache_dir = pi_baseline_cache_dir(
                benchmark_root=root / "benchmarks",
                pi_agent=pi_agent,
                manifest=manifest,
                model="model",
                provider_only="provider",
            )
            save_pi_baseline(cache_dir=cache_dir, job_dir=job_dir)

            restored_job = root / "restored"
            self.assertTrue(restore_pi_baseline(cache_dir=cache_dir, job_dir=restored_job, skip_scoring=False))
            self.assertTrue((restored_job / "pi" / "predictions.jsonl").exists())
            self.assertTrue((restored_job / "pi" / "official_scoring" / "report.json").exists())

    def test_pi_pin_persists_one_head(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pin_path = Path(temp_dir) / "pinned_head.json"

            written = write_pi_pin(pin_path, repo_url="https://example.test/pi", ref="main", commit_sha="abc123")
            loaded = read_pi_pin(pin_path)

            self.assertEqual(written["commit_sha"], "abc123")
            self.assertEqual(loaded["commit_sha"], "abc123")
            self.assertEqual(loaded["requested_ref"], "main")

    def test_pi_model_arg_uses_openrouter_prefix(self):
        self.assertEqual(pi_model_arg("minimax/minimax-m2.7"), "openrouter/minimax/minimax-m2.7")
        self.assertEqual(pi_model_arg("openrouter/minimax/minimax-m2.7"), "openrouter/minimax/minimax-m2.7")

    def test_mini_swe_agent_filter_is_exact(self):
        self.assertEqual(mini_swe_agent_filter("django__django-123"), "^django__django-123$")

    def test_mini_swe_agent_prediction_patch_reads_preds_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            predictions_path = Path(temp_dir) / "preds.json"
            predictions_path.write_text(
                json.dumps({"django__django-123": {"model_patch": "diff --git a/file b/file\n"}}),
                encoding="utf-8",
            )

            patch = mini_swe_agent_prediction_patch(predictions_path, "django__django-123")

            self.assertTrue(patch.startswith("diff --git"))

    def test_predictions_include_patch_rejects_empty_predictions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            predictions_path = Path(temp_dir) / "predictions.jsonl"
            predictions_path.write_text(
                "\n".join(
                    [
                        json.dumps({"instance_id": "a", "model_patch": ""}),
                        json.dumps({"instance_id": "b", "model_patch": "   "}),
                    ],
                ),
                encoding="utf-8",
            )

            self.assertFalse(predictions_include_patch(predictions_path))

            predictions_path.write_text(
                json.dumps({"instance_id": "a", "model_patch": "diff --git a/a b/a\n"}),
                encoding="utf-8",
            )

            self.assertTrue(predictions_include_patch(predictions_path))

    def test_reset_prediction_outputs_removes_stale_agent_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            agent_dir = Path(temp_dir) / "agent"
            (agent_dir / "official_scoring").mkdir(parents=True)
            (agent_dir / "logs").mkdir()
            for name in ("predictions.jsonl", "solve_results.jsonl", "usage_summary.json", "score_summary.json"):
                (agent_dir / name).write_text("{}\n", encoding="utf-8")

            reset_prediction_outputs(agent_dir)

            self.assertFalse((agent_dir / "predictions.jsonl").exists())
            self.assertFalse((agent_dir / "official_scoring").exists())

    def test_dashboard_merge_preserves_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dashboard_path = Path(temp_dir) / "dashboard_data.json"
            dashboard_path.write_text(json.dumps({"duels": [], "status": {}}), encoding="utf-8")

            merge_benchmark_into_dashboard(
                dashboard_path,
                {
                    "status": "completed",
                    "benchmark": "swebench_verified_sample_50",
                    "king_commit_sha": "abc",
                    "scores": {"king": {}, "pi": {}, "delta_pass_rate": 0.1},
                },
            )

            payload = json.loads(dashboard_path.read_text(encoding="utf-8"))
            latest = payload["benchmarks"]["swebench_verified"]["latest"]
            self.assertEqual(latest["king_commit_sha"], "abc")


if __name__ == "__main__":
    unittest.main()
