from __future__ import annotations


def pick_earliest_slot(slots: list[str]) -> str | None:
    if not slots:
        return None
    return sorted(slots)[0]
