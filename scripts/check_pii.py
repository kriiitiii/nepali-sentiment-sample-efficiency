"""Scan files intended for release for PII patterns, secrets and local paths.

Usage:
    python scripts/check_pii.py [paths ...]   (defaults to the tracked public tree)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PATTERNS = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b", re.I),
    "phone_np": re.compile(r"(?<!\d)9[678]\d{8}(?!\d)"),
    "google_api_key": re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    "github_token": re.compile(r"gh[pousr]_[0-9A-Za-z]{30,}"),
    "windows_path": re.compile(r"\b[A-Z]:\\(?:Users|Kriti)\\", re.I),
    "secret_assign": re.compile(r"(api[_-]?key|secret|password|token)\s*[:=]\s*['\"][^'\"]{8,}", re.I),
}
SKIP_DIRS = {".venv", ".git", "data", "__pycache__", ".pytest_cache", "work"}
TEXT_EXT = {".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".json", ".csv", ".tex", ".cfg"}
# Synthetic values used in unit-test fixtures.
ALLOWED = {"a.b@example.com", "9841234567"}


def scan(paths):
    hits = []
    for root in paths:
        files = [root] if root.is_file() else root.rglob("*")
        for f in files:
            if f.is_dir() or f.suffix not in TEXT_EXT or SKIP_DIRS & set(f.parts):
                continue
            if f.name == "check_pii.py":
                continue
            for n, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                for name, pat in PATTERNS.items():
                    for m in pat.finditer(line):
                        if m.group(0) not in ALLOWED:
                            hits.append(f"{f}:{n}: {name}")
    return hits


if __name__ == "__main__":
    targets = [Path(p) for p in sys.argv[1:]] or [Path(".")]
    found = scan(targets)
    print("\n".join(found) or "no findings")
    sys.exit(1 if found else 0)
