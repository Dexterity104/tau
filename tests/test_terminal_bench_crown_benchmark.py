from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from swebench_crown_benchmark import AgentIdentity
from terminal_bench_crown_benchmark import (
    TerminalBenchManifest,
    build_comparison,
    build_tb_command,
    load_manifest,
    parse_terminal_bench_score,
    save_terminal_baseline,
    restore_terminal_baseline,
    terminal_baseline_cache_dir,
    terminal_bench_harness_error,
    row_passed,
    terminal_bench_env,
)
from terminal_bench_mini_swe_agent import openrouter_model_name
from terminal_bench_tau_agent import run_command, tau_terminal_bench_env


class TerminalBenchCrownBenchmarkTest(unittest.TestCase):
    def test_manifest_rejects_duplicate_task_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "dataset": "terminal-bench-core==0.1.1",
                        "count": 2,
                        "seed": 66,
                        "task_ids": ["hello-world", "hello-world"],
                    },
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_manifest(manifest_path)

    def test_build_tb_command_uses_custom_agent_import_path(self):
        manifest = TerminalBenchManifest("tb", "terminal-bench-core==0.1.1", 66, None, ("hello-world",), "hash")
        agent = AgentIdentity("king", "https://example.test/repo.git", "abc", Path("/tmp/agent.py"), {})

        command = build_tb_command(
            agent=agent,
            manifest=manifest,
            output_dir=Path("/tmp/out"),
            model="model",
            workers=4,
        )

        self.assertIn("--agent-import-path", command)
        self.assertIn("terminal_bench_tau_agent:TauSubnet66Agent", command)
        self.assertIn("--task-id", command)
        self.assertIn("hello-world", command)

    def test_build_tb_command_uses_builtin_baseline_agent(self):
        manifest = TerminalBenchManifest("tb", "terminal-bench-core==0.1.1", 66, 10, (), "hash")
        agent = AgentIdentity("mini-swe-agent", "terminal-bench://builtin/mini-swe-agent", "builtin", Path("mini-swe-agent"), {})

        command = build_tb_command(
            agent=agent,
            manifest=manifest,
            output_dir=Path("/tmp/out"),
            model="model",
            workers=4,
        )

        self.assertIn("--agent-import-path", command)
        self.assertIn("terminal_bench_mini_swe_agent:TauMiniSweAgent", command)
        self.assertNotIn("--task-id", command)
        self.assertIn("--n-tasks", command)

    def test_mini_adapter_normalizes_openrouter_model(self):
        self.assertEqual(
            openrouter_model_name("minimax/minimax-m2.7"),
            "openrouter/minimax/minimax-m2.7",
        )
        self.assertEqual(
            openrouter_model_name("openrouter/minimax/minimax-m2.7"),
            "openrouter/minimax/minimax-m2.7",
        )

    def test_terminal_bench_score_parser_counts_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            (output / "results.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {"task_id": "a", "passed": True},
                            {"task_id": "b", "passed": False},
                        ],
                    },
                ),
                encoding="utf-8",
            )

            score = parse_terminal_bench_score(output)

            self.assertEqual(score.resolved_count, 1)
            self.assertEqual(score.total_count, 2)
            self.assertEqual(score.pass_rate, 0.5)

    def test_terminal_bench_score_parser_reads_aggregate_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            run_dir = output / "2026-05-30__00-00-00"
            run_dir.mkdir()
            (run_dir / "results.json").write_text(
                json.dumps({"n_resolved": 3, "n_unresolved": 2, "accuracy": 0.6}),
                encoding="utf-8",
            )

            score = parse_terminal_bench_score(output)

            self.assertEqual(score.resolved_count, 3)
            self.assertEqual(score.total_count, 5)
            self.assertEqual(score.pass_rate, 0.6)

    def test_terminal_bench_harness_error_detects_null_unknown_agent_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "2026-05-30__00-00-00"
            run_dir.mkdir()
            (run_dir / "results.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "task_id": "a",
                                "is_resolved": None,
                                "failure_mode": "unknown_agent_error",
                                "trial_started_at": None,
                            },
                        ],
                    },
                ),
                encoding="utf-8",
            )

            error = terminal_bench_harness_error(output_dir=Path(temp_dir), stderr="")

            self.assertEqual(error, "terminal-bench trials did not start")

    def test_row_passed_accepts_statuses_and_scores(self):
        self.assertTrue(row_passed({"status": "resolved"}))
        self.assertFalse(row_passed({"result": "failed"}))
        self.assertTrue(row_passed({"score": 1}))
        self.assertIsNone(row_passed({"status": "unknown"}))

    def test_comparison_computes_delta(self):
        manifest = TerminalBenchManifest("tb", "dataset", 66, None, ("a", "b"), "hash")
        king = AgentIdentity("king", "repo", "king-sha", Path("/tmp/agent.py"), {})
        baseline = AgentIdentity("terminus", "builtin", "builtin", Path("terminus"), {})

        comparison = build_comparison(
            king={"commit_sha": "king-sha"},
            king_agent=king,
            baseline_agent=baseline,
            manifest=manifest,
            scores=(
                parse_score(2, 2),
                parse_score(1, 2),
            ),
            total_elapsed_seconds=12.0,
            started_at="start",
            model="model",
        )

        self.assertEqual(comparison["scores"]["delta_pass_rate"], 0.5)
        self.assertEqual(comparison["baseline_name"], "terminus")

    def test_adapter_env_and_command_are_explicit(self):
        env = tau_terminal_bench_env()

        self.assertIn("TAU_AGENT_REPO_URL", env)
        self.assertIn("TAU_MODEL", env)
        self.assertTrue(run_command("hello").startswith("python /installed-agent/run_tau_agent.py "))

    def test_runner_env_sets_agent_identity(self):
        agent = AgentIdentity("king", "https://example.test/repo.git", "abc", Path("/tmp/agent.py"), {})
        env = terminal_bench_env(
            agent=agent,
            args=SimpleNamespace(model="model", api_base="base", agent_timeout_seconds=30),
        )

        self.assertEqual(env["TAU_AGENT_REPO_URL"], "https://example.test/repo.git")
        self.assertEqual(env["TAU_AGENT_REF"], "abc")
        self.assertEqual(env["TAU_MODEL"], "model")

    def test_terminal_baseline_cache_restores_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = TerminalBenchManifest("tb", "dataset", 66, None, ("a",), "hash")
            agent = AgentIdentity("mini-swe-agent", "builtin", "builtin", Path("mini-swe-agent"), {})
            job_dir = root / "job"
            baseline_dir = job_dir / "mini-swe-agent"
            (baseline_dir / "tb-run").mkdir(parents=True)
            for name in ("score_summary.json", "run.json", "stdout.txt", "stderr.txt"):
                (baseline_dir / name).write_text("{}\n", encoding="utf-8")
            (baseline_dir / "score_summary.json").write_text(
                json.dumps(
                    {
                        "resolved_count": 1,
                        "total_count": 1,
                        "pass_rate": 1.0,
                        "report_path": "report.json",
                        "error": None,
                    },
                ),
                encoding="utf-8",
            )
            (baseline_dir / "tb-run" / "results.json").write_text("{}", encoding="utf-8")
            cache_dir = terminal_baseline_cache_dir(
                benchmark_root=root / "benchmarks",
                baseline_agent=agent,
                manifest=manifest,
                model="model",
            )

            save_terminal_baseline(cache_dir=cache_dir, job_dir=job_dir, baseline_name="mini-swe-agent")
            restored_job = root / "restored"

            self.assertTrue(restore_terminal_baseline(cache_dir=cache_dir, job_dir=restored_job, baseline_name="mini-swe-agent"))
            self.assertTrue((restored_job / "mini-swe-agent" / "tb-run" / "results.json").exists())


def parse_score(resolved: int, total: int):
    from terminal_bench_crown_benchmark import TerminalBenchScore

    return TerminalBenchScore(resolved, total, resolved / total, "report.json")


if __name__ == "__main__":
    unittest.main()
