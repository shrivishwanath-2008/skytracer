"""
Run smoke tests for phases 1–5 after any change.

Usage:
  python scripts/verify_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = [
    ("Phase 1 detection", [sys.executable, "scripts/smoke_test.py"]),
    ("Phase 2 visible stars", [sys.executable, "scripts/test_visible_stars.py"]),
    ("Phase 3 alignment", [sys.executable, "scripts/test_align.py"]),
    ("Phase 4 geometric match", [sys.executable, "scripts/test_geometric_match.py"]),
    ("Phase 5 annotate", [sys.executable, "scripts/test_annotate.py"]),
]


def main() -> None:
    failed = []
    for name, cmd in TESTS:
        print(f"--- {name} ---")
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            failed.append(name)
        print()

    if failed:
        print("FAILED:", ", ".join(failed))
        sys.exit(1)
    print("All phase checks passed.")


if __name__ == "__main__":
    main()
