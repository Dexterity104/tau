import unittest
from unittest.mock import patch

import docker_solver


class DockerSolverCleanupTest(unittest.TestCase):
    def test_kill_container_is_best_effort(self):
        with patch("docker_solver._run", side_effect=RuntimeError("Command timed out after 30s: docker kill abc")):
            docker_solver._kill_container("abc")

    def test_remove_container_is_best_effort(self):
        with patch("docker_solver._run", side_effect=RuntimeError("Command timed out after 30s: docker rm -f")):
            docker_solver._remove_container("abc")

    def test_read_runner_events_returns_empty_on_docker_failure(self):
        with patch("docker_solver._run", side_effect=RuntimeError("Command timed out after 30s: docker exec abc")):
            self.assertEqual(docker_solver._read_runner_events_from_container(container_id="abc"), "")

    def test_collect_repo_patch_best_effort_returns_none_on_docker_failure(self):
        with patch(
            "docker_solver._collect_repo_patch_from_container",
            side_effect=RuntimeError("Command timed out after 120s: docker exec abc"),
        ):
            self.assertIsNone(docker_solver._collect_repo_patch_from_container_best_effort(container_id="abc"))


if __name__ == "__main__":
    unittest.main()
