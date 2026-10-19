"""
rag.py: policy retrieval for ShopAgent-AI (Checkpoint 6)

Finds the section of the policy documents (policies/*.md) that answers a customer's
question, so the agent can answer from real text instead of guessing.

How it works
1. Every "## " section of every policy file is one chunk.
2. Chunks and questions are embedded with a multilingual sentence-transformers model,
   so Arabic questions match the English policy without translation.
3. Search is a plain dot product with numpy (the vectors are normalised, so this is the
   cosine similarity). With a handful of sections no vector database is needed.
4. Sections whose raw similarity is below MIN_SCORE are dropped. If none is left, nothing is
   returned and the agent must say it cannot verify the policy.
5. A small keyword "topic boost" (see TOPIC_RULES) then orders the remaining sections. It
   only changes the ORDER: it can neither create a match nor hide one.

Settings (environment variables)
  RAG_EMBED_MODEL   embedding model (default: paraphrase-multilingual-MiniLM-L12-v2)
  RAG_MIN_SCORE     minimum cosine similarity (default 0.30). PROVISIONAL: measure it with
                    calibrate_rag.py on a machine that has the model, then set it here.
  RAG_TOPIC_BOOST   boost for the topic rules (default 0.25, use 0 to switch them off)

The first call downloads the embedding model (internet needed once) and loads it into memory.

Usage:
    python -m shopagent.rag "What is the return window?"
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

from shopagent.config import POLICY_DIR

try:  # the settings below may come from the .env file, so load it BEFORE reading them
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

EMBED_MODEL = os.getenv(
    "RAG_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.30"))
TOPIC_BOOST = float(os.getenv("RAG_TOPIC_BOOST", "0.25"))
NOT_FOUND = "I could not find information about that in our policy documents."


# ------------------------------------------------------------------
# Embeddings
# ------------------------------------------------------------------
_embedder = None  # callable(list[str]) -> normalised float32 matrix (rows = texts)
_index = {"key": None, "embedder": None, "sections": [], "vectors": None}


def set_embedder(fn) -> None:
    """Replace the embedding function (the tests use a small stand-in). None = real model."""
    global _embedder
    _embedder = fn
    _index["key"] = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer  # raises ImportError if missing

        model = SentenceTransformer(EMBED_MODEL)
        _embedder = lambda texts: np.asarray(  # noqa: E731
            model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True),
            dtype="float32",
        )
    return _embedder


# ------------------------------------------------------------------
# Loading the policy documents
# ------------------------------------------------------------------
def _document_title(text: str, fallback: str) -> str:
    """The first '# ' heading of a file is the title the customer sees as the source."""
    match = re.search(r"^# (.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else fallback


def load_sections() -> list[dict]:
    """[{title, text, source (file name), document (document title)}, ...] for every '## ' section."""
    files = sorted(POLICY_DIR.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No policy files (*.md) found in {POLICY_DIR}")

    sections = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        document = _document_title(text, path.stem.replace("_", " ").title())
        for part in re.split(r"(?=^## )", text, flags=re.MULTILINE):
            part = part.strip()
            if part.startswith("## "):
                sections.append(
                    {
                        "title": part.splitlines()[0][3:].strip(),
                        "text": part,
                        "source": path.name,
                        "document": document,
                    }
                )
    if not sections:
        raise FileNotFoundError(f"The policy files in {POLICY_DIR} have no '## ' sections")
    return sections


def _get_index() -> dict:
    """Embed the sections once; rebuild when a policy file changes or the embedder changes."""
    files = sorted(POLICY_DIR.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No policy files (*.md) found in {POLICY_DIR}")
    embed = _get_embedder()
    key = tuple((p.name, p.stat().st_mtime_ns) for p in files)

    if key != _index["key"] or _index["embedder"] is not embed:
        sections = load_sections()
        _index.update(
            key=key,
            embedder=embed,
            sections=sections,
            vectors=np.asarray(embed([s["text"] for s in sections]), dtype="float32"),
        )
    return _index


# ------------------------------------------------------------------
# Topic boost
# ------------------------------------------------------------------
# The FIRST rule with a matching keyword wins. Headings are compared in lower case and
# with the non-breaking hyphen (U+2011) turned into "-", so "Non‑Returnable Items" matches.
# These rules are written for the sample policy's headings; for other documents they simply
# never match (the boost does nothing). Set RAG_TOPIC_BOOST=0 to switch them off.
TOPIC_RULES = (
    ("shipping costs", ("shipping", "شحن")),
    ("warranty", ("warranty", "ضمان", "كفالة")),
    (
        "non-returnable items",
        (
            "final sale", "clearance", "non-returnable", "cannot return", "not returnable",
            "opened software", "digital download", "perishable", "personalised", "personalized",
            "customised", "customized", "غير قابل للإرجاع", "منتج مخصص", "قابل للتلف",
        ),
    ),
    (
        "defective or damaged items",
        ("defective", "damaged", "faulty", "broken", "تالف", "معيب", "عيب"),
    ),
    ("general return policy", ("return", "refund", "إرجاع", "ارجاع", "استرجاع")),
)


def _normalise_heading(heading: str) -> str:
    return heading.replace("\u2011", "-").strip().lower()


def topic_heading(question: str) -> str:
    """The heading (normalised) that the question's keywords point to, or '' if none."""
    text = question.lower()
    for heading, keywords in TOPIC_RULES:
        if any(word in text for word in keywords):
            return heading
    return ""


def boost_for(question: str, section_title: str) -> float:
    heading = topic_heading(question)
    return TOPIC_BOOST if heading and _normalise_heading(section_title) == heading else 0.0


# ------------------------------------------------------------------
# Search
# ------------------------------------------------------------------
def scores_for(question: str) -> list[dict]:
    """Every section with its raw cosine score and boosted score, best (boosted) first.
    Used by search() and by calibrate_rag.py."""
    index = _get_index()
    query = np.asarray(_get_embedder()([question]), dtype="float32")[0]
    raw = index["vectors"] @ query
    rows = [
        {
            **section,
            "score": float(raw[i]),
            "boosted": float(raw[i]) + boost_for(question, section["title"]),
        }
        for i, section in enumerate(index["sections"])
    ]
    return sorted(rows, key=lambda r: -r["boosted"])


def search(question: str, top_k: int = 1) -> list[dict]:
    """The best matching section(s), or [] when the best raw similarity is below MIN_SCORE."""
    if not question.strip():
        return []
    # The threshold looks at each section's RAW similarity first; the boost then only decides
    # the order among the sections that passed. So the boost can neither create a match nor hide one.
    passed = [r for r in scores_for(question) if r["score"] >= MIN_SCORE]
    return passed[:top_k]


def answer(question: str) -> str:
    """Evidence text for the agent: the best section with its source, or NOT_FOUND.
    Raises ImportError / FileNotFoundError when the embedding library or the policy files
    are missing (the agent's policy tool catches these)."""
    results = search(question)
    if not results:
        return NOT_FOUND
    best = results[0]
    return (
        f'Policy evidence (use ONLY this text). Source title: "{best["document"]}"\n\n'
        f'[{best["source"]}]\n{best["text"]}'
    )


if __name__ == "__main__":
    import sys

    try:
        print(answer(" ".join(sys.argv[1:]) or "What is the return window?"))
    except ImportError:
        raise SystemExit("sentence-transformers is not installed: pip install sentence-transformers")
