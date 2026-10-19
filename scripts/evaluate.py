"""Evaluate the real ShopAgent chat on policy questions and a product search.

Run ``python scripts/evaluate.py`` with the populated ``data/shop.db``, or ``python scripts/evaluate.py --demo``
for a temporary two-product catalog. Both modes need the project dependencies, the embedding
model (sentence-transformers) and a reachable Groq or Ollama model: the checks call the same
``agent.chat()`` that the Streamlit app uses.

Cases (8):
- return window and warranty length, in English and in Arabic
- a follow-up question ("How many months does it last?") that depends on the previous turn
- two unrelated questions (English and Arabic): no policy text may be retrieved or quoted
- a product search whose product ids and prices are checked against the database

Every policy check looks at the model's OWN reply for the policy fact and its unit, and at
the tool evidence for the retrieved source. The app adds a source label and, when an English
question gets an Arabic reply, shows the policy excerpt instead; those additions are reported
but are not counted as the model's work.

For the broader regression suite (23 scenarios, every model on its own) use scripts/eval_agent.py.
"""

import argparse
from contextlib import closing, contextmanager
import os
from pathlib import Path
import re
import sqlite3
import sys
from tempfile import TemporaryDirectory
import uuid


ROOT = Path(__file__).resolve().parents[1]
POLICY_FILE = ROOT / "policies" / "return_policy.md"
POLICY_SOURCE = "return_policy.md"

# Spelled-out forms of the numbers the policy uses (a correct answer may say "thirty days")
NUMBER_WORDS = {
    30: ("thirty", "ثلاثين", "ثلاثون"),
    12: ("twelve", "[اإأ]ثن[يىا]\\s+عشر"),
}

# (name, turns, expected number, language): every turn is sent in the same conversation
POLICY_CASES = [
    ("return_english", ["What is the return window for an unused item?"], 30, "en"),
    ("warranty_english", ["How many months is the warranty for an electronic product?"], 12, "en"),
    ("return_arabic", ["ما مدة إرجاع منتج غير مستخدم بالأيام؟"], 30, "ar"),
    ("warranty_arabic", ["ما مدة ضمان المنتجات الإلكترونية بالشهور؟"], 12, "ar"),
    ("warranty_follow_up",
     ["Is there a manufacturer warranty for electronic products?", "How many months does it last?"], 12, "en"),
]
UNRELATED_CASES = [
    ("unrelated_english", ["What is the capital of France?"]),
    ("unrelated_arabic", ["ما عاصمة فرنسا؟"]),
]


def require(condition, reason):
    if not condition:
        raise AssertionError(reason)


def policy_title():
    """The title the customer sees as the source: the first '# ' heading of the policy file."""
    match = re.search(r"^# (.+)$", POLICY_FILE.read_text(encoding="utf-8"), flags=re.MULTILINE)
    return match.group(1).strip() if match else POLICY_FILE.stem


@contextmanager
def evaluation_catalog(use_demo):
    """Keep optional demo data separate from the team's real database."""
    from shopagent import db

    if not use_demo:
        db_path = Path(db.DB_PATH).resolve()
        require(db_path.is_file(),
                f"Missing product database: {db_path}. Use --demo or provide the team's catalog.")
        print(f"Catalog: {db_path}")
        yield db_path
        return

    previous_path = db.DB_PATH
    with TemporaryDirectory(prefix="shopagent-cp6-") as directory:
        db_path = Path(directory) / "shop.db"
        db.DB_PATH = str(db_path)
        try:
            with closing(db.get_conn()) as conn:
                conn.executescript(db.SCHEMA)
                conn.executemany(
                    "INSERT INTO products "
                    "(id, name, category, brand, price, stock, description, rating, category_path, sub_category) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (1, "Demo USB-C Cable", "Electronics", "Demo", 299.0, 8,
                         "USB-C charging cable", 4.5, "Electronics|Cables", "Cables"),
                        (2, "Demo HDMI Cable", "Electronics", "Demo", 499.0, 5,
                         "HDMI display cable", 4.3, "Electronics|Cables", "Cables"),
                    ],
                )
                conn.commit()
            print("Catalog: temporary demo catalog (2 products; prices in INR)")
            yield db_path
        finally:
            db.DB_PATH = previous_path


def cable_products(db_path):
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, name, price FROM products "
            "WHERE name LIKE '%cable%' OR category_path LIKE '%cable%' "
            "ORDER BY rating DESC, id ASC LIMIT 5"
        ).fetchall()
    require(rows, "The product database needs a cable for the search scenario; try --demo.")
    return [dict(row) for row in rows]


def show_reply(result):
    print(f"  model: {result['model']} | tools: {', '.join(result['tools']) or '(none)'}")
    if "answer_policy" in result["tools"]:
        sources = ", ".join(result.get("policy_sources", [])) or "(none)"
        print(f"  policy sources: {sources} | source label added by app: "
              f"{result.get('source_label_added', False)} | policy excerpt shown instead of the reply: "
              f"{result.get('used_policy_excerpt', False)}")
    print(f"  model reply: {result.get('model_reply', result['reply'])}")
    if result.get("model_reply") != result["reply"]:
        print(f"  displayed reply: {result['reply']}")


def contains_policy_fact(text, expected_number):
    """The number AND its unit, as digits, Arabic-Indic digits or words ('30 days', '٣٠ يومًا', 'thirty days')."""
    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    text = re.sub(r"[\u064b-\u065f\u0670]", "", text)
    unit = r"(?:days?|يوما?|أيام)" if expected_number == 30 else r"(?:months?|شهرا?|شهور)"
    number = "|".join([str(expected_number), *NUMBER_WORDS.get(expected_number, ())])
    return re.search(rf"(?<!\d)(?:{number})\W*{unit}", text, re.IGNORECASE)


def check_policy(result, expected_number, language):
    require("model_reply" in result, "The chat did not expose its original model answer")
    model_reply = result["model_reply"]
    displayed_reply = result["reply"]
    require("answer_policy" in result["tools"], "The chat did not use the policy tool")
    require(POLICY_SOURCE in result.get("policy_sources", []), "No policy source was retrieved")
    require(contains_policy_fact(model_reply, expected_number),
            "The model's own answer missed the policy fact and its unit")
    require(contains_policy_fact(displayed_reply, expected_number),
            "The displayed answer missed the policy fact and its unit")
    require(policy_title().lower() in displayed_reply.lower(), "The displayed answer missed the source title")

    if language == "ar":
        require(len(re.findall(r"[\u0600-\u06ff]", displayed_reply)) >= 10,
                "The Arabic question was not answered in Arabic")
    else:
        unit = "days?" if expected_number == 30 else "months?"
        number = "|".join([str(expected_number), *NUMBER_WORDS.get(expected_number, ())])
        require(re.search(rf"\b(?:{number})\W*{unit}\b", displayed_reply, re.IGNORECASE),
                "The English question was not answered with the expected English fact")


def check_no_policy(result):
    """An unrelated question must not retrieve policy text or state a policy fact."""
    require(not result.get("policy_sources"), "Policy text was retrieved for an unrelated question")
    for number in (30, 12):
        require(not contains_policy_fact(result.get("model_reply", result["reply"]), number),
                "The reply states a policy fact for an unrelated question")


def check_product_search(result, products):
    require("search_products" in result["tools"], "The chat did not search the catalog")
    reply = result["reply"].translate(str.maketrans("٠١٢٣٤٥٦٧٨٩٬٫", "0123456789,."))
    reply = re.sub(r"[*_`#]", "", reply)
    id_markers = list(re.finditer(r"\b(?:product\s+)?id\s*[:=]?\s*", reply, re.IGNORECASE))
    require(id_markers, "No explicit product ID appeared in the answer")
    by_id = {int(row["id"]): row for row in products}
    amount = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    currency = r"(?:INR|Rs\.?|₹)"

    for index, marker in enumerate(id_markers):
        end = id_markers[index + 1].start() if index + 1 < len(id_markers) else len(reply)
        entry = reply[marker.end():end]
        product_id = re.match(r"(\d+)(?!\d)(?!\.\d)", entry)
        require(product_id, "An explicit product ID was missing its integer value")
        product_id = int(product_id.group(1))
        require(product_id in by_id, f"Product ID {product_id} is not a catalog cable")
        require(len(re.findall(r"\bprice\b", entry, re.IGNORECASE)) == 1,
                f"Product ID {product_id} needs exactly one labeled price")
        price = re.search(
            rf"\bprice\s*[:=]?\s*(?:{currency}\s*{amount}|{amount}\s*{currency})",
            entry,
            re.IGNORECASE,
        )
        require(price, f"Product ID {product_id} is missing a price in INR")
        value = float((price.group(1) or price.group(2)).replace(",", ""))
        require(abs(value - by_id[product_id]["price"]) < 0.01,
                f"Product ID {product_id} has the wrong catalog price")


def run_case(runtime, name, turns, check):
    print(f"\n{name}: {'  ->  '.join(turns)}")
    thread = f"cp6-eval-{uuid.uuid4()}"
    try:
        for turn in turns:
            result = runtime.chat(turn, thread_id=thread)
        show_reply(result)
        check(result)
    except Exception as exc:
        print(f"  FAIL: {type(exc).__name__}: {exc}")
        return False
    print("  PASS")
    return True


def build_cases(products):
    cases = [
        (name, turns, lambda result, n=number, lang=language: check_policy(result, n, lang))
        for name, turns, number, language in POLICY_CASES
    ]
    cases += [(name, turns, check_no_policy) for name, turns in UNRELATED_CASES]
    cases.append((
        "product_search",
        ["Find one cable in the store. Reply in English on one line using this format: "
         "Product ID: <id>, Price: <price> INR."],
        lambda result: check_product_search(result, products),
    ))
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo", action="store_true",
                        help="use a temporary demo catalog without changing data/shop.db")
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    try:
        with evaluation_catalog(args.demo) as db_path:
            products = cable_products(db_path)
            from shopagent import agent as runtime
            print("\nShopAgent CP6 live-chat evaluation")
            print("These checks call the same agent.chat() used by the Streamlit app.")
            results = [run_case(runtime, name, turns, check) for name, turns, check in build_cases(products)]
    except Exception as exc:
        print(f"Cannot run evaluation: {type(exc).__name__}: {exc}")
        return 1

    passed = sum(results)
    total = len(results)
    print(f"\nResult: {passed}/{total} passed ({passed / total:.0%} success rate)")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
