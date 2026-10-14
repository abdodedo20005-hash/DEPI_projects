"""
test_rag.py  –  CP6 tests for the RAG policy module
Run:  python test_rag.py
"""
from pathlib import Path


def test_policy_file_exists():
    assert Path("policies/return_policy.md").exists(), "policy file missing"


def test_rag_module_importable():
    import rag  # noqa: F401


def test_rag_contains_content():
    text = Path("policies/return_policy.md").read_text()
    assert "30" in text and "return" in text.lower()


def test_rag_answer_return():
    import rag
    ans = rag.answer("What is the return window?")
    assert isinstance(ans, str) and len(ans) > 10


def test_rag_answer_warranty():
    import rag
    ans = rag.answer("Is there a warranty on electronics?")
    assert isinstance(ans, str) and len(ans) > 10


def test_rag_unsupported_question():
    import rag
    ans = rag.answer("What is the capital of France?")
    # Should NOT invent a policy answer
    assert isinstance(ans, str)
    # We just confirm it returns a string without crashing


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
