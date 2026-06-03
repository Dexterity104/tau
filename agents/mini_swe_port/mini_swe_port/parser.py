from __future__ import annotations

import re


_TAG_PATTERN = re.compile(r"<(?P<tag>bash|read|finish)>\s*(?P<body>.*?)\s*</(?P=tag)>", re.DOTALL)


def parse_action(text: str) -> tuple[str, str]:
    match = _TAG_PATTERN.search(text)
    if match is None:
        return "bash", "git diff -- ."
    return match.group("tag"), match.group("body").strip()
