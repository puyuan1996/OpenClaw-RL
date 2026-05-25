from __future__ import annotations


def score_subject(subject: str) -> int:
    normalized = subject.lower()
    if "urgent" in normalized or "today" in normalized:
        return 3
    if "follow up" in normalized or "review" in normalized:
        return 2
    return 1
