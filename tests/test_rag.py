"""
test_rag.py: tests for the policy retrieval module (Checkpoint 6)

Run:  python tests/test_rag.py

Two kinds of tests:
1. LOGIC tests (always run). They use a tiny stand-in for the embedding model, so they
   check the code (chunking, ranking, threshold, topic boost, rebuild on change, the
   evidence format, error handling) but NOT how well the real model understands questions.
2. QUALITY tests (marked [model]). They use the real sentence-transformers model and check
   that English and Arabic questions find the right section and that unrelated questions
   find nothing. They are SKIPPED when the library or the model is not available.
   A skipped quality test is NOT a pass: run them on a machine with the model installed,
   and run scripts/calibrate_rag.py to choose RAG_MIN_SCORE.
"""

import shutil
import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so "from shopagent import ..." works when this file is run directly

from shopagent import rag  # noqa: E402
from tests.helpers import hash_embedder  # noqa: E402


class Skipped(Exception):
    pass


# ------------------------------------------------------------------
# Test helpers
# ------------------------------------------------------------------
def isolated(fn, use_model=False):
    """Give the test a clean state and restore the module settings afterwards."""
    def wrapper():
        saved = (rag.POLICY_DIR, rag.MIN_SCORE, rag.TOPIC_BOOST)
        try:
            if use_model:
                try:
                    import sentence_transformers  # noqa: F401
                except ImportError:
                    raise Skipped("sentence-transformers is not installed")
                rag.set_embedder(None)
                try:
                    rag.search("warm up")
                except Exception as exc:  # no internet on the first run, out of memory, ...
                    raise Skipped(f"the model could not be loaded ({type(exc).__name__})")
            else:
                rag.set_embedder(hash_embedder)
            return fn()
        finally:
            rag.POLICY_DIR, rag.MIN_SCORE, rag.TOPIC_BOOST = saved
            rag.set_embedder(None)
    wrapper.__name__ = fn.__name__
    wrapper.uses_model = use_model
    return wrapper


def logic(fn):
    return isolated(fn, use_model=False)


def model(fn):
    return isolated(fn, use_model=True)


def titles_for(question, top_k=1):
    return [r["title"] for r in rag.search(question, top_k)]


def norm(title):
    return rag._normalise_heading(title)


# ------------------------------------------------------------------
# LOGIC tests (stand-in embedder)
# ------------------------------------------------------------------
@logic
def test_policy_file_exists():
    assert any(Path(rag.POLICY_DIR).glob("*.md")), "no policy file found"


@logic
def test_sections_split_at_headings_and_carry_their_source():
    sections = rag.load_sections()
    names = [norm(s["title"]) for s in sections]
    assert len(sections) >= 5
    assert "warranty" in names and "general return policy" in names
    assert all(s["source"].endswith(".md") and s["text"].startswith("## ") for s in sections)


@logic
def test_document_title_is_the_first_heading():
    assert all(s["document"] == "Return & Warranty Policy" for s in rag.load_sections())


@logic
def test_a_section_finds_itself():
    for section in rag.load_sections():
        top = rag.search(section["text"])
        assert top and top[0]["title"] == section["title"] and top[0]["score"] > 0.99


@logic
def test_gibberish_and_empty_questions_find_nothing():
    assert rag.search("zzzz qqqq xxxx") == []
    assert rag.search("") == [] and rag.search("   ") == []


@logic
def test_threshold_decides_on_the_raw_score():
    section = rag.load_sections()[0]
    half = " ".join(section["text"].split()[: len(section["text"].split()) // 2])
    assert rag.search(half), "half of a section should still match"
    rag.MIN_SCORE = 0.999
    assert rag.search(half) == []


@logic
def test_boost_cannot_create_a_match():
    # "return" triggers the general-return boost, but the raw similarity is far below the threshold
    assert rag.search("return zzzz qqqq xxxx wwww") == []


@logic
def test_boost_cannot_hide_a_match():
    # The boosted section is below the threshold; a better, unboosted section must still be returned
    rows = [
        {"title": "Boosted but weak", "score": 0.20, "boosted": 0.45},
        {"title": "Strong", "score": 0.40, "boosted": 0.40},
    ]
    original = rag.scores_for
    rag.scores_for = lambda question: rows
    try:
        assert [r["title"] for r in rag.search("anything")] == ["Strong"]
    finally:
        rag.scores_for = original


@logic
def test_topic_rules_priority_and_languages():
    assert rag.topic_heading("Who pays for return shipping?") == "shipping costs"
    assert rag.topic_heading("Is the warranty claim returnable?") == "warranty"
    assert rag.topic_heading("Can I return a customised product?") == "non-returnable items"
    assert rag.topic_heading("My item arrived broken") == "defective or damaged items"
    assert rag.topic_heading("What is the return window?") == "general return policy"
    assert rag.topic_heading("من يدفع الشحن؟") == "shipping costs"
    assert rag.topic_heading("ما مدة الضمان؟") == "warranty"
    assert rag.topic_heading("hello") == ""


@logic
def test_boost_matches_headings_with_a_non_breaking_hyphen():
    question = "Can I return a customised product?"
    assert rag.boost_for(question, "Non\u2011Returnable Items") == rag.TOPIC_BOOST   # U+2011, as in the file
    assert rag.boost_for(question, "Non-Returnable Items") == rag.TOPIC_BOOST       # plain hyphen
    assert rag.boost_for(question, "Warranty") == 0.0


@logic
def test_boost_changes_the_order_and_can_be_switched_off():
    question = "return shipping warranty"
    on = [r["title"] for r in rag.scores_for(question)]
    rag.TOPIC_BOOST = 0
    off = [r["title"] for r in rag.scores_for(question)]
    assert norm(on[0]) == "shipping costs"           # the first rule wins the boost
    assert on != off or norm(off[0]) == "shipping costs"
    assert all(r["boosted"] == r["score"] for r in rag.scores_for(question))


@logic
def test_evidence_has_the_source_title_file_marker_and_section():
    section = next(s for s in rag.load_sections() if norm(s["title"]) == "warranty")
    evidence = rag.answer(section["text"])
    assert 'Source title: "Return & Warranty Policy"' in evidence
    assert "[return_policy.md]" in evidence and "## Warranty" in evidence


@logic
def test_not_found_text():
    assert rag.answer("zzzz qqqq xxxx") == rag.NOT_FOUND


@logic
def test_many_calls_in_one_process():
    section = rag.load_sections()[0]
    for _ in range(5):
        assert rag.search(section["text"])[0]["title"] == section["title"]


@logic
def test_new_or_changed_policy_files_are_picked_up():
    original = rag.POLICY_DIR
    tmp = Path(tempfile.mkdtemp())
    try:
        shutil.copy(next(original.glob("*.md")), tmp / "return_policy.md")
        (tmp / "gifts.md").write_text("## Gift Wrapping\nWe offer free gift wrapping on request.\n", encoding="utf-8")
        rag.POLICY_DIR = tmp
        assert "Gift Wrapping" in titles_for("We offer free gift wrapping on request")
        assert rag.search("We offer free gift wrapping on request")[0]["document"] == "Gifts"  # no H1: title from the file name
        (tmp / "gifts.md").write_text("## Gift Wrapping\nGift wrapping costs 5 dollars.\n", encoding="utf-8")
        assert "5 dollars" in rag.search("Gift wrapping costs 5 dollars")[0]["text"]
    finally:
        rag.POLICY_DIR = original
        shutil.rmtree(tmp)


@logic
def test_settings_are_read_from_the_dotenv_file():
    # agent.py imports rag BEFORE it loads .env, so rag must load .env itself
    import os
    import subprocess
    import sys

    folder = Path(tempfile.mkdtemp())
    try:
        (folder / ".env").write_text("RAG_MIN_SCORE=0.77\nRAG_TOPIC_BOOST=0.11\n", encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if not k.startswith("RAG_")}
        env["PYTHONPATH"] = str(ROOT)
        out = subprocess.run(
            [sys.executable, "-c", "from shopagent import rag; print(rag.MIN_SCORE, rag.TOPIC_BOOST)"],
            cwd=folder, env=env, capture_output=True, text=True,
        )
        assert out.stdout.split() == ["0.77", "0.11"], out.stdout + out.stderr
    finally:
        shutil.rmtree(folder)


@logic
def test_missing_policy_folder_raises_a_clear_error():
    rag.POLICY_DIR = Path(tempfile.mkdtemp())
    try:
        try:
            rag.answer("What is the return window?")
        except FileNotFoundError:
            return
        raise AssertionError("expected FileNotFoundError")
    finally:
        shutil.rmtree(rag.POLICY_DIR)


# ------------------------------------------------------------------
# QUALITY tests (real model; skipped when it is not available)
# ------------------------------------------------------------------
@model
def test_model_return_window_and_refund_english():
    top = rag.search("What is the return window for an unused item?")[0]
    assert norm(top["title"]) == "general return policy" and "30 days" in top["text"]
    assert "general return policy" in [norm(t) for t in titles_for("When do I get my refund?", 2)]


@model
def test_model_defective_item_english():
    top = rag.search("My phone arrived broken, what should I do?")[0]
    assert norm(top["title"]) == "defective or damaged items" and "7 days" in top["text"]


@model
def test_model_warranty_english():
    top = rag.search("How many months is the warranty for an electronic product?")[0]
    assert norm(top["title"]) == "warranty" and "12" in top["text"]


@model
def test_model_other_sections_english():
    assert norm(titles_for("Can I return opened software?")[0]) == "non-returnable items"
    assert norm(titles_for("Who pays for return shipping?")[0]) == "shipping costs"


@model
def test_model_arabic_questions():
    assert norm(titles_for("ما مدة إرجاع منتج غير مستخدم بالأيام؟")[0]) == "general return policy"
    assert norm(titles_for("ما مدة ضمان المنتجات الإلكترونية بالشهور؟")[0]) == "warranty"
    assert norm(titles_for("وصلني المنتج تالف، ماذا أفعل؟")[0]) == "defective or damaged items"


@model
def test_model_unrelated_questions_find_nothing():
    for question in [
        "What is the capital of France?", "What is the weather today?", "Tell me a joke",
        "How do I cook rice?", "ما عاصمة فرنسا؟", "كيف حال الطقس اليوم؟",
    ]:
        found = rag.search(question)
        assert found == [], (
            f"matched {found[0]['title']!r} with score {found[0]['score']:.2f} for {question!r}; "
            "run scripts/calibrate_rag.py and raise RAG_MIN_SCORE"
        )


# ------------------------------------------------------------------
if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    passed = failed = skipped = 0
    for name, fn in tests:
        tag = "[model] " if getattr(fn, "uses_model", False) else ""
        try:
            fn()
            passed += 1
            print(f"PASS  {tag}{name}")
        except Skipped as why:
            skipped += 1
            print(f"SKIP  {tag}{name}: {why}")
        except Exception as exc:
            failed += 1
            print(f"FAIL  {tag}{name}: {exc!r}")
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped (of {len(tests)})")
    if skipped:
        print("NOTE: the skipped tests check the real model's understanding of questions.")
        print("      Retrieval quality is NOT verified until they pass (pip install sentence-transformers).")
    raise SystemExit(1 if failed else 0)
