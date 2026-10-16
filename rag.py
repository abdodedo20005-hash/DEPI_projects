"""Retrieve source-labelled return and warranty policy text for the chat agent."""

from __future__ import annotations

import os
import re
from pathlib import Path


POLICY_DIR = Path(__file__).parent / "policies"
EMBED_MODEL = os.getenv(
    "RAG_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
MIN_SCORE = 0.30
_index_cache = None


def _load_sections():
    """Keep each heading and its rules together as one retrieval unit."""
    sections, sources = [], []
    for path in sorted(POLICY_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for section in re.split(r"(?=^## )", text, flags=re.MULTILINE):
            section = section.strip()
            if section.startswith("## "):
                sections.append(section)
                sources.append(path.name)
    if not sections:
        raise FileNotFoundError(f"No policy sections found in {POLICY_DIR}")
    return sections, sources


def _get_index():
    """Build the local embedding index once per process."""
    global _index_cache
    if _index_cache is None:
        import faiss
        from sentence_transformers import SentenceTransformer

        sections, sources = _load_sections()
        model = SentenceTransformer(EMBED_MODEL)
        vectors = model.encode(sections, convert_to_numpy=True, normalize_embeddings=True)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        _index_cache = model, index, sections, sources
    return _index_cache


def _topic_heading(question):
    """Prefer the matching heading when nearby sections have similar scores."""
    text = question.lower()
    if any(word in text for word in ("shipping", "شحن")):
        return "## Shipping Costs"
    if any(word in text for word in ("warranty", "ضمان", "كفالة")):
        return "## Warranty"
    if any(word in text for word in (
        "final sale", "clearance", "non-returnable", "cannot return", "not returnable",
        "opened software", "digital download", "perishable", "personalised", "personalized",
        "customised", "customized", "غير قابل للإرجاع", "منتج مخصص", "قابل للتلف",
    )):
        return "## Non‑Returnable Items"
    if any(word in text for word in ("defective", "damaged", "faulty", "broken", "تالف", "معيب", "عيب")):
        return "## Defective or Damaged Items"
    if any(word in text for word in ("return", "refund", "إرجاع", "ارجاع", "استرجاع")):
        return "## General Return Policy"
    return ""


def answer(question: str) -> str:
    """Return the best policy section with its file name, or a clear no-match reply."""
    if not question.strip():
        return "I could not find information about that in our policy documents."

    model, index, sections, sources = _get_index()
    query = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    scores, positions = index.search(query, len(sections))
    heading = _topic_heading(question)
    candidates = [
        (float(score) + (0.25 if heading and sections[pos].startswith(heading) else 0),
         float(score), int(pos))
        for score, pos in zip(scores[0], positions[0]) if pos >= 0
    ]
    _, score, position = max(candidates)
    if score < MIN_SCORE:
        return "I could not find information about that in our policy documents."

    source = sources[position]
    return f"Policy evidence (cite {source} in the final answer):\n\n[{source}]\n{sections[position]}"


if __name__ == "__main__":
    import sys

    print(answer(" ".join(sys.argv[1:]) or "What is the return window?"))
