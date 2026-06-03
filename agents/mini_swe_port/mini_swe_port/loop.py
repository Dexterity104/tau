from __future__ import annotations

from mini_swe_port.client import chat_completion
from mini_swe_port.diff import git_diff, read_file, repo_files
from mini_swe_port.parser import parse_action
from mini_swe_port.prompts import SYSTEM_PROMPT, initial_user_prompt, observation_prompt
from mini_swe_port.tools import run_bash


def _step(
    *,
    repo_path: str,
    action: str,
    body: str,
) -> str:
    if action == "read":
        return read_file(repo_path, body)
    if action == "bash":
        return run_bash(repo_path, body)
    if action == "finish":
        return body or "finished"
    return f"Unknown action: {action}"


def _append_message(messages: list[dict[str, str]], role: str, content: str) -> list[dict[str, str]]:
    return [*messages, {"role": role, "content": content}]


def run_agent(
    *,
    repo_path: str,
    issue: str,
    model: str,
    api_base: str,
    api_key: str,
    max_steps: int = 24,
) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": initial_user_prompt(issue=issue, files=repo_files(repo_path))},
    ]
    last_observation = ""
    for _ in range(max_steps):
        response = chat_completion(
            api_base=api_base,
            api_key=api_key,
            model=model,
            messages=messages,
        )
        action, body = parse_action(response)
        output = _step(repo_path=repo_path, action=action, body=body)
        diff = git_diff(repo_path)
        messages = _append_message(messages, "assistant", response)
        last_observation = output
        if action == "finish" and diff.strip():
            return {"success": True, "message": output, "diff": diff}
        messages = _append_message(
            messages,
            "user",
            observation_prompt(action=action, body=body, output=output, diff=diff),
        )

    diff = git_diff(repo_path)
    return {
        "success": bool(diff.strip()),
        "message": "step limit reached" if diff.strip() else last_observation or "no changes made",
        "diff": diff,
    }
