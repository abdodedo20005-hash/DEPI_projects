"""
Run every test file and print one line for each.

Run:  python scripts/run_tests.py

A test file that skipped tests (the retrieval-quality tests need the embedding model) is
shown with a warning: a skipped test is not a pass.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ["test_layout", "test_tools", "test_orders", "test_rag", "test_agent_policy"]


def main() -> int:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    failed = 0
    for name in TESTS:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "tests" / f"{name}.py")],
            cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        lines = [l for l in proc.stdout.splitlines() if "passed" in l]
        errors = proc.stderr.strip().splitlines()
        summary = lines[-1].strip() if lines else (errors[-1] if errors else "no output")
        ok = proc.returncode == 0
        failed += not ok
        note = "   <- retrieval quality NOT verified (needs sentence-transformers)" if "skipped" in summary and ok else ""
        print(f"{'ok    ' if ok else 'FAILED'} {name:<20} {summary}{note}")
        if not ok:
            print("\n".join("       " + l for l in (proc.stdout.splitlines()[-6:] + errors[-3:])))
    print("\nAll test files passed." if not failed else f"\n{failed} test file(s) failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
