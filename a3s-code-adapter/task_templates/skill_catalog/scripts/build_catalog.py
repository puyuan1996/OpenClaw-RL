from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = ROOT / "skills"
OUTPUT_PATH = ROOT / "catalog" / "skills_index.json"

SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def _parse_sections(text: str) -> dict[str, str]:
    matches = list(SECTION_RE.finditer(text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group(1).strip().lower()] = text[start:end].strip()
    return sections


def _parse_bullets(section_text: str) -> list[str]:
    items: list[str] = []
    for line in section_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            items.append(stripped[2:].strip())
    return items


def load_skill(skill_dir: Path) -> dict[str, object]:
    skill_file = skill_dir / "SKILL.md"
    text = skill_file.read_text(encoding="utf-8")
    lines = [line.rstrip() for line in text.splitlines()]
    title = lines[0].lstrip("#").strip() if lines else skill_dir.name.replace("_", " ").title()
    sections = _parse_sections(text)
    return {
        "slug": skill_dir.name,
        "title": title,
        "summary": sections.get("summary", "").splitlines()[0].strip() if sections.get("summary") else "",
        "tags": _parse_bullets(sections.get("tags", "")),
        "scripts": _parse_bullets(sections.get("scripts", "")),
    }


def build_catalog() -> list[dict[str, object]]:
    if not SKILLS_DIR.exists():
        return []
    skills = [load_skill(skill_dir) for skill_dir in sorted(SKILLS_DIR.iterdir()) if (skill_dir / "SKILL.md").exists()]
    return skills


def write_catalog(entries: list[dict[str, object]], output_path: Path = OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    write_catalog(build_catalog())
    print(f"wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
