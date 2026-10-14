"""
rag.py  –  CP6 RAG module for ShopAgent-AI
Returns and warranty policy Q&A grounded in policies/*.md documents.

Dependencies (lightweight, no paid API):
  sentence-transformers  (local embeddings)
  faiss-cpu              (vector search)

Install:
  uv pip install sentence-transformers faiss-cpu
  (or: pip install sentence-transformers faiss-cpu)

Usage:
  import rag
  print(rag.answer("What is the return window?"))
"""

from __future__ import annotations

import os
import pickle
import textwrap
from pathlib import Path

# ── tunables ──────────────────────────────────────────────────────────
POLICY_DIR   = Path(__file__).parent / "policies"
CACHE_FILE   = Path(__file__).parent / ".rag_cache.pkl"
EMBED_MODEL  = os.getenv("RAG_EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE   = 300   # characters per chunk
CHUNK_OVERLAP = 60
TOP_K        = 3
MIN_SCORE    = 0.30  # cosine similarity threshold below which we say "not found"
# ─────────────────────────────────────────────────────────────────────


def _chunk(text: str) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end].strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c) > 30]


def _load_documents() -> tuple[list[str], list[str]]:
    """Load all *.md files from POLICY_DIR.
    Returns (chunks, source_labels).
    """
    chunks, labels = [], []
    for path in sorted(POLICY_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for chunk in _chunk(text):
            chunks.append(chunk)
            labels.append(path.name)
    if not chunks:
        raise FileNotFoundError(f"No .md files found in {POLICY_DIR}")
    return chunks, labels


def _build_index(chunks: list[str]):
    """Embed chunks and return a FAISS index."""
    import faiss
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product == cosine on normalised vecs
    index.add(embeddings)
    return model, index, embeddings


# Module-level lazy cache
_cache: dict = {}


def _get_index():
    """Return (model, index, chunks, labels) – build once, then reuse."""
    global _cache
    if _cache:
        return _cache["model"], _cache["index"], _cache["chunks"], _cache["labels"]

    # Try persistent disk cache first
    chunks, labels = _load_documents()
    mtime = max(p.stat().st_mtime for p in POLICY_DIR.glob("*.md"))
    if CACHE_FILE.exists():
        data = pickle.loads(CACHE_FILE.read_bytes())
        if data.get("mtime") == mtime and data.get("model") == EMBED_MODEL:
            _cache = data
            return data["model_obj"], data["index"], data["chunks"], data["labels"]

    model, index, _ = _build_index(chunks)
    _cache = {
        "model": EMBED_MODEL,
        "model_obj": model,
        "index": index,
        "chunks": chunks,
        "labels": labels,
        "mtime": mtime,
    }
    CACHE_FILE.write_bytes(pickle.dumps(_cache))
    return model, index, chunks, labels


def answer(question: str) -> str:
    """Answer a question grounded in the policy documents.
    Returns a plain-text answer, or a "not found" message if the
    documents do not contain relevant information.
    """
    try:
        model, index, chunks, labels = _get_index()
    except ImportError as exc:
        return (
            "RAG dependencies are not installed. "
            f"Run: uv pip install sentence-transformers faiss-cpu  ({exc})"
        )
    except FileNotFoundError as exc:
        return str(exc)

    import numpy as np

    q_vec = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(q_vec, TOP_K)

    top_scores = scores[0]
    top_idx    = indices[0]

    if float(top_scores[0]) < MIN_SCORE:
        return (
            "I'm sorry, I could not find information about that in our "
            "return and warranty policy documents."
        )

    # Build context from top-k chunks
    context_parts = []
    for score, idx in zip(top_scores, top_idx):
        if score >= MIN_SCORE and idx >= 0:
            context_parts.append(f"[{labels[idx]}]\n{chunks[idx]}")
    context = "\n\n".join(context_parts)

    # Try to use the LLM from agent.py if available; otherwise return the context
    try:
        from langchain_groq import ChatGroq
        from dotenv import load_dotenv, find_dotenv
        load_dotenv(find_dotenv(usecwd=True))

        llm_model = os.getenv("GROQ_FAST_MODEL", "llama3-8b-8192")
        llm = ChatGroq(model=llm_model, temperature=0)
        prompt = (
            "You are a customer-support assistant. "
            "Answer the user's question using ONLY the policy text below. "
            "If the answer is not in the text, say the information is not available.\n\n"
            f"Policy text:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        resp = llm.invoke(prompt)
        return resp.content.strip()
    except Exception:
        # Fallback: return the most relevant chunk directly
        return (
            f"Based on our policy:\n\n"
            + "\n\n".join(context_parts[:2])
        )


# ── CLI helper ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What is the return window?"
    print(answer(q))
