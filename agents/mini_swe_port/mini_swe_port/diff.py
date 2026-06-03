from __future__ import annotations

import subprocess
from pathlib import Path


def git_diff(repo_path: str, *, limit: int = 12000) -> str:
    proc = subprocess.run(
        ["git", "diff", "--", "."],
        cwd=repo_path,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    diff = proc.stdout or ""
    if len(diff) <= limit:
        return diff
    return diff[: limit // 2] + "\n...[diff truncated]...\n" + diff[-limit // 2 :]


def repo_files(repo_path: str, *, limit: int = 120) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_path,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    files = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return files[:limit]


def read_file(repo_path: str, relative_path: str, *, limit: int = 12000) -> str:
    path = (Path(repo_path) / relative_path).resolve()
    root = Path(repo_path).resolve()
    if root not in path.parents and path != root:
        return f"Refusing to read outside repository: {relative_path}"
    if not path.is_file():
        return f"File does not exist: {relative_path}"
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= limit:
        return text
    return text[: limit // 2] + "\n...[file truncated]...\n" + text[-limit // 2 :]
