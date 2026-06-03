from __future__ import annotations


SYSTEM_PROMPT = """You are a software engineering agent running inside a repository.
Fix the issue by inspecting files, editing code, and checking the final diff.
Use one action per response:
<read>path/to/file.py</read>
<bash>command</bash>
<finish>short reason</finish>
Prefer focused shell commands and direct Python edits. Do not install dependencies."""


def initial_user_prompt(*, issue: str, files: list[str]) -> str:
    file_list = "\n".join(files)
    return f"""Issue:
{issue.strip()}

Tracked files:
{file_list}

Start by inspecting the most relevant files."""


def observation_prompt(*, action: str, body: str, output: str, diff: str) -> str:
    return f"""Action: {action}
Input:
{body}

Observation:
{output}

Current diff:
{diff or "(no changes yet)"}

Choose the next action."""
