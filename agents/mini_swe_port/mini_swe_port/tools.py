from __future__ import annotations

import subprocess


def run_bash(repo_path: str, command: str, *, timeout: int = 90, limit: int = 20000) -> str:
    proc = subprocess.run(
        ["bash", "-lc", command],
        cwd=repo_path,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    output = f"$ {command}\nexit={proc.returncode}\n\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    if len(output) <= limit:
        return output
    return output[: limit // 2] + "\n...[command output truncated]...\n" + output[-limit // 2 :]
