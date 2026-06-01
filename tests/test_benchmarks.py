from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchmarks import (
    benchmark_runs,
    runloop_benchmark_runs,
    selected_benchmark_specs,
    selected_runloop_specs,
    task_count_for_preset,
    write_benchmark_report,
)


class BenchmarkPlanTest(unittest.TestCase):
    def test_defaults_to_three_requested_benchmarks(self):
        specs = selected_benchmark_specs(None)

        self.assertEqual(
            [spec.name for spec in specs],
            ["rebench", "deepswe", "terminal-bench"],
        )

    def test_replaces_agent_and_appends_common_overrides(self):
        runs = benchmark_runs(
            names=["rebench"],
            agent="king",
            model="openai/gpt-5.2",
            n_tasks=5,
            sample_seed=66,
            output_root=Path("/tmp/tau-benchmarks"),
        )

        command = runs[0].command
        self.assertEqual(command[command.index("--agent") + 1], "king")
        self.assertIn("--model", command)
        self.assertIn("openai/gpt-5.2", command)
        self.assertIn("--n-tasks", command)
        self.assertIn("5", command)
        self.assertIn("--sample-seed", command)
        self.assertIn("66", command)

    def test_rejects_unknown_benchmark(self):
        with self.assertRaisesRegex(ValueError, "Unknown benchmark"):
            selected_benchmark_specs(["nope"])

    def test_report_writer_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "nested" / "report.json"

            write_benchmark_report(report_path, [{"name": "rebench"}])

            self.assertTrue(report_path.exists())
            self.assertIn("rebench", report_path.read_text(encoding="utf-8"))

    def test_runloop_defaults_to_swe_verified_and_terminal_bench(self):
        specs = selected_runloop_specs(None)

        self.assertEqual(
            [spec.name for spec in specs],
            ["swe-bench-verified", "terminal-bench"],
        )

    def test_runloop_expands_agent_and_baseline_across_benchmarks(self):
        runs = runloop_benchmark_runs(
            names=None,
            agent="king",
            baseline="pi",
            scenario_ids=("scn_1",),
            timeout=120,
            n_concurrent_trials=10,
            output_root=Path("/tmp/tau-benchmarks"),
        )

        self.assertEqual(len(runs), 2)
        commands = [run.command for run in runs]
        self.assertTrue(all(command[:3] == ("rli", "benchmark-job", "run") for command in commands))
        self.assertTrue(all("--scenarios" in command for command in commands))
        self.assertTrue(all("scn_1" in command for command in commands))
        self.assertTrue(all("--timeout" in command for command in commands))
        self.assertTrue(all("--n-concurrent-trials" in command for command in commands))
        self.assertTrue(all(command.count("--agent") == 2 for command in commands))

    def test_smoke_preset_caps_to_one_task(self):
        self.assertEqual(task_count_for_preset(preset="smoke", n_tasks=None), 1)
        self.assertEqual(task_count_for_preset(preset="smoke", n_tasks=2), 2)


if __name__ == "__main__":
    unittest.main()
