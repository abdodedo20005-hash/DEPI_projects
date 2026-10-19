"""
test_layout.py: checks the project layout itself.

Run:  python tests/test_layout.py

It catches the mistakes that moving files around causes: an old flat import left behind,
a path that depends on the folder you start from, a script that the README forgot to mention.
"""

import py_compile
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shopagent import config  # noqa: E402

SKIP_DIRS = {".venv", "venv", "__pycache__", "data", ".git", "invoices", "test_invoices", "eval_results"}
OLD_FLAT_IMPORT = re.compile(r"^\s*(import|from)\s+(db|tools|orders|queries|rag|agent)\b", re.MULTILINE)


def python_files():
    return [p for p in ROOT.rglob("*.py") if not (SKIP_DIRS & set(p.relative_to(ROOT).parts))]


def test_every_python_file_compiles():
    for path in python_files():
        py_compile.compile(str(path), doraise=True)


def test_no_file_still_uses_the_old_flat_imports():
    offenders = [str(p.relative_to(ROOT)) for p in python_files() if OLD_FLAT_IMPORT.search(p.read_text(encoding="utf-8"))]
    assert not offenders, f"old flat imports (use 'from shopagent import ...'): {offenders}"


def test_the_policy_file_is_where_the_code_looks_for_it():
    assert (config.POLICY_DIR / "return_policy.md").exists()


def test_paths_do_not_depend_on_the_current_directory():
    code = (
        f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
        "from shopagent import db, rag, orders; print(db.DB_PATH); print(db.CSV_PATH); "
        "print(rag.POLICY_DIR); print(orders.INVOICE_DIR)"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=Path.home(), capture_output=True, text=True)
    lines = out.stdout.split()
    assert len(lines) == 4, out.stdout + out.stderr
    for line in lines:
        assert Path(line).is_absolute() and str(ROOT) in line, f"not inside the project: {line}"


def test_the_readme_mentions_every_script_and_test():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    names = [p.name for p in (ROOT / "scripts").glob("*.py")] + [p.name for p in (ROOT / "tests").glob("test_*.py")]
    missing = [n for n in names if n not in readme]
    assert not missing, f"not mentioned in the README: {missing}"


def test_secrets_and_data_are_git_ignored():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").split()
    for pattern in (".env", "data/", "*.db", "invoices/"):
        assert pattern in ignore, f"{pattern} is missing from .gitignore"


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:
            failed += 1
            print(f"FAIL  {name}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    raise SystemExit(1 if failed else 0)
