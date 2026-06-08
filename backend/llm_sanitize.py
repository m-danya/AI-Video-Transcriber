"""Remove common LLM meta / closing phrases from model output."""
from __future__ import annotations

import re
from typing import Optional

# Trailing blocks (English + Chinese, escaped for source readability) often added despite instructions.
_PATTERNS = [
    re.compile(r"(?is)\n{1,2}let me know[\s\S]{0,800}\Z"),
    re.compile(r"(?is)\n{1,2}feel free to[\s\S]{0,800}\Z"),
    re.compile(r"(?is)\n{1,2}if you (?:need|have|would like)[\s\S]{0,800}\Z"),
    re.compile(r"(?is)\n{1,2}(?:happy to|please let me know|don't hesitate)[\s\S]{0,800}\Z"),
    re.compile(r"(?is)\n{1,2}(?:hope this helps|thanks for reading)[\s\S]{0,400}\Z"),
    re.compile(
        r"(?is)\n{1,2}(?:\u8bf7\u544a\u8bc9|\u5982\u6709\u9700\u8981|"
        r"\u5982\u9700|\u6b22\u8fce\u53cd\u9988|\u5e0c\u671b\u5bf9\u4f60|"
        r"\u4ee5\u4e0a(?:\u5185\u5bb9)?)[\s\S]{0,800}\Z"
    ),
]


def strip_llm_artifacts(text: Optional[str]) -> str:
    if not text or not isinstance(text, str):
        return (text or "").strip()
    t = text.strip()
    for _ in range(6):
        before = t
        for pat in _PATTERNS:
            t = pat.sub("", t).strip()
        if t == before:
            break
    lines = t.split("\n")
    while lines:
        last = lines[-1].strip()
        if not last:
            lines.pop()
            continue
        low = last.lower()
        if len(last) < 200 and any(
            x in low
            for x in (
                "let me know",
                "further adjustments",
                "feel free",
                "hope this helps",
                "\u8bf7\u544a\u8bc9\u6211",
                "\u5982\u9700\u8c03\u6574",
                "\u6b22\u8fce\u53cd\u9988",
            )
        ):
            lines.pop()
            continue
        break
    return "\n".join(lines).strip()
