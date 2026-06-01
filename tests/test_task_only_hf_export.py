import json
import tempfile
import unittest
from pathlib import Path

from task_only_hf_export import (
    artifact_bundle_sha256,
    dedupe_task_rows,
    iter_jsonl_path,
    repo_files_matching,
    shard_path,
    task_only_row,
    write_jsonl,
)


class TaskOnlyHfExportTest(unittest.TestCase):
    def test_task_only_row_keeps_task_artifacts_and_drops_archive_state(self):
        row = {
            "archive_hour": "2026-05-31-01",
            "archive_reason": "rotation",
            "pool_label": "primary",
            "task_name": "validate-1",
            "task_root_name": "validate-1",
            "pool_task": {"king_lines": 10},
            "king": {"hotkey": "secret-ish"},
            "task_metadata": {"issue": "fix parser"},
            "commit_metadata": {"sha": "abc"},
            "artifacts": [
                {"path": "task/task.txt", "encoding": "utf-8", "content": "fix parser\n"},
                {"path": "task/reference.patch", "encoding": "utf-8", "content": "diff\n"},
                {"path": "solutions/king/solution.diff", "encoding": "utf-8", "content": "diff\n"},
            ],
        }

        projected = task_only_row(row, source_dataset="owner/old", source_path="tasks/primary/hour.jsonl")

        self.assertIsNotNone(projected)
        assert projected is not None
        self.assertEqual(projected["row_type"], "tau_task_only")
        self.assertEqual(projected["task_name"], "validate-1")
        self.assertEqual(projected["task_metadata"], {"issue": "fix parser"})
        self.assertEqual(projected["commit_metadata"], {"sha": "abc"})
        self.assertEqual(projected["source"]["dataset"], "owner/old")
        self.assertEqual([item["path"] for item in projected["artifacts"]], ["task/task.txt", "task/reference.patch"])
        self.assertEqual(projected["artifact_count"], 2)
        self.assertEqual(projected["artifact_bundle_sha256"], artifact_bundle_sha256(projected["artifacts"]))
        self.assertNotIn("pool_task", projected)
        self.assertNotIn("king", projected)
        self.assertNotIn("pool_label", projected)

    def test_task_only_row_skips_rows_without_task_artifacts(self):
        self.assertIsNone(
            task_only_row(
                {"task_name": "validate-1", "artifacts": [{"path": "solutions/king/solution.diff"}]},
                source_dataset="owner/old",
                source_path="tasks/primary/hour.jsonl",
            )
        )

    def test_repo_files_matching_uses_source_patterns(self):
        matched = repo_files_matching(
            ["README.md", "tasks/primary/2026.jsonl", "tasks/primary/2026.jsonl.gz", "rollouts/a.jsonl.gz"],
            ("tasks/**/*.jsonl", "tasks/**/*.jsonl.gz"),
        )

        self.assertEqual(matched, ("tasks/primary/2026.jsonl", "tasks/primary/2026.jsonl.gz"))

    def test_shard_path_uses_directory_or_jsonl_stem(self):
        self.assertEqual(shard_path("tasks", 3), "tasks/tasks-00003.jsonl.gz")
        self.assertEqual(shard_path("tasks.jsonl", 3), "tasks-00003.jsonl.gz")
        self.assertEqual(shard_path("tasks.jsonl.gz", 3), "tasks-00003.jsonl.gz")

    def test_jsonl_round_trip_and_dedupe(self):
        rows = [
            {"task_name": "task-a", "value": 1},
            {"task_name": "task-a", "value": 2},
            {"task_name": "task-b", "value": 3},
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tasks.jsonl"

            count = write_jsonl(path, rows)
            loaded = list(iter_jsonl_path(path))

        self.assertEqual(count, 3)
        self.assertEqual(loaded, rows)
        self.assertEqual(list(dedupe_task_rows(loaded)), [rows[0], rows[2]])

    def test_iter_jsonl_ignores_non_dict_rows(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "rows.jsonl"
            path.write_text(json.dumps([1, 2, 3]) + "\n" + json.dumps({"task_name": "task"}) + "\n")

            loaded = list(iter_jsonl_path(path))

        self.assertEqual(loaded, [{"task_name": "task"}])
