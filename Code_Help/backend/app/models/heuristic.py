# app/models/heuristic.py
import re
from typing import Dict, Any

def heuristic_rate(code: str) -> Dict[str, Any]:
    lines = code.splitlines()
    long_lines = sum(1 for l in lines if len(l) > 120)

    depth = 0
    max_depth = 0
    for l in lines:
        depth += l.count("{") - l.count("}")
        max_depth = max(max_depth, depth)

    issues = []
    if long_lines:
        issues.append(f"{long_lines} lines exceed 120 characters")
    if max_depth >= 5:
        issues.append("Deep nesting detected")

    score = 10 - min(4, long_lines) - max(0, max_depth - 3)
    score = max(0, score)

    return {
        "model": "heuristic-fallback",
        "score": score,
        "summary": "Heuristic-based review (no LLM available).",
        "suggestions": [
            "Reduce nesting depth",
            "Split large functions",
            "Add comments where logic is complex",
            "Limit line length",
            "Add tests"
        ],
        "issues": issues
    }
