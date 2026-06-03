from __future__ import annotations

from mini_swe_port.loop import run_agent


def solve(repo_path: str, issue: str, model: str, api_base: str, api_key: str) -> dict:
    return run_agent(
        repo_path=repo_path,
        issue=issue,
        model=model,
        api_base=api_base,
        api_key=api_key,
    )
