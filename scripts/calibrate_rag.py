"""
calibrate_rag.py: measure how well the policy retrieval separates policy questions from
unrelated ones, and suggest a value for RAG_MIN_SCORE.

Run it on a machine that has the embedding model (pip install sentence-transformers):

    python scripts/calibrate_rag.py

What it prints:
- for every policy question: the section it found and the raw cosine similarity
- for every unrelated question: the best raw similarity (it should be LOW)
- the lowest score of a policy question and the highest score of an unrelated one.
  If the first is higher than the second, any threshold between them separates the two
  groups, and the script suggests the middle.
- how many policy questions land on the right section WITH and WITHOUT the topic boost,
  so you can see whether the boost is needed.

The question lists are small and hand-written: a threshold chosen on them is a starting
point, not a guarantee. Add your own questions to the lists below and run it again.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so "from shopagent import ..." works when this file is run directly

from shopagent import rag  # noqa: E402

# (question, expected section heading)
POLICY_QUESTIONS = [
    ("What is the return window for an unused item?", "General Return Policy"),
    ("How many days do I have to return something?", "General Return Policy"),
    ("When do I get my refund?", "General Return Policy"),
    ("Do I need a receipt to return an item?", "General Return Policy"),
    ("My phone arrived broken, what should I do?", "Defective or Damaged Items"),
    ("How long do I have to report a damaged product?", "Defective or Damaged Items"),
    ("How many months is the warranty for an electronic product?", "Warranty"),
    ("Does the warranty cover water damage?", "Warranty"),
    ("Can I return a customised product?", "Non-Returnable Items"),
    ("Can I return opened software?", "Non-Returnable Items"),
    ("Who pays for return shipping?", "Shipping Costs"),
    ("How do I contact support about a return?", "Contact"),
    ("ما مدة إرجاع منتج غير مستخدم بالأيام؟", "General Return Policy"),
    ("ما مدة ضمان المنتجات الإلكترونية بالشهور؟", "Warranty"),
    ("وصلني المنتج تالف، ماذا أفعل؟", "Defective or Damaged Items"),
    ("من يدفع تكلفة شحن الإرجاع؟", "Shipping Costs"),
    ("هل يمكنني إرجاع برنامج تم فتحه؟", "Non-Returnable Items"),
]

UNRELATED_QUESTIONS = [
    "What is the capital of France?",
    "What is the weather today?",
    "Tell me a joke",
    "How do I cook rice?",
    "Who won the football match yesterday?",
    "Recommend a good book",
    "Show me USB cables under 500 rupees",
    "Add the second product to my cart",
    "Do you sell laptops?",
    "What is the price of product 12?",
    "ما عاصمة فرنسا؟",
    "كيف حال الطقس اليوم؟",
    "كيف أطبخ الأرز؟",
    "اعرض لي كابلات شحن",
]


def norm(title: str) -> str:
    return rag._normalise_heading(title)


def analyse() -> dict:
    """Scores for both question lists. Uses whichever embedder rag is currently set to."""
    policy = []
    for question, expected in POLICY_QUESTIONS:
        ranked = rag.scores_for(question)                                   # with the boost
        best_raw = max(ranked, key=lambda r: r["score"])
        policy.append(
            {
                "question": question,
                "expected": expected,
                "found": ranked[0]["title"],
                "score": ranked[0]["score"],
                "best_raw_score": best_raw["score"],
                "right": norm(ranked[0]["title"]) == norm(expected),
                "right_without_boost": norm(best_raw["title"]) == norm(expected),
            }
        )
    unrelated = []
    for question in UNRELATED_QUESTIONS:
        best = max(rag.scores_for(question), key=lambda r: r["score"])
        unrelated.append({"question": question, "found": best["title"], "score": best["score"]})

    lowest_policy = min(p["best_raw_score"] for p in policy)
    highest_unrelated = max(u["score"] for u in unrelated)
    return {
        "policy": policy,
        "unrelated": unrelated,
        "lowest_policy": lowest_policy,
        "highest_unrelated": highest_unrelated,
        "separable": lowest_policy > highest_unrelated,
        "suggested": round((lowest_policy + highest_unrelated) / 2, 2),
    }


def main() -> int:
    try:
        rag.search("warm up")  # loads the model
    except ImportError:
        print("sentence-transformers is not installed: pip install sentence-transformers")
        return 1
    except Exception as exc:
        print(f"The embedding model could not be loaded ({type(exc).__name__}: {exc}).")
        print("The first run needs internet to download it.")
        return 1

    result = analyse()
    print(f"Model: {rag.EMBED_MODEL}\nCurrent RAG_MIN_SCORE: {rag.MIN_SCORE} | topic boost: {rag.TOPIC_BOOST}\n")

    print("POLICY QUESTIONS (raw cosine score of the best section)")
    for p in result["policy"]:
        mark = "ok " if p["right"] else "WRONG"
        print(f"  {p['best_raw_score']:.2f} {mark:<5} {p['question'][:58]:<58} -> {p['found']}")
    print("\nUNRELATED QUESTIONS (best raw score; should be low)")
    for u in result["unrelated"]:
        print(f"  {u['score']:.2f}       {u['question'][:58]:<58} -> {u['found']}")

    right = sum(p["right"] for p in result["policy"])
    right_off = sum(p["right_without_boost"] for p in result["policy"])
    total = len(result["policy"])
    print(f"\nRight section with the topic boost : {right}/{total}")
    print(f"Right section WITHOUT the boost    : {right_off}/{total}   (set RAG_TOPIC_BOOST=0 to use it)")
    print(f"Lowest policy-question score       : {result['lowest_policy']:.2f}")
    print(f"Highest unrelated-question score   : {result['highest_unrelated']:.2f}")

    if result["separable"]:
        print(f"\nThe two groups are separable. Suggested RAG_MIN_SCORE = {result['suggested']} "
              f"(any value between {result['highest_unrelated']:.2f} and {result['lowest_policy']:.2f} works).")
        print("Put it in your .env file, then run python test_rag.py again.")
    else:
        print("\nThe groups OVERLAP: no threshold separates them on these questions.")
        print("Do not claim that the policy tool refuses unrelated questions; look at the rows above,")
        print("and consider a stronger embedding model or adding a keyword check.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
