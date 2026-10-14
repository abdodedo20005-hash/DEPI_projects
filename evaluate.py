"""
evaluate.py  –  CP6 Evaluation Suite for ShopAgent-AI
Run: python evaluate.py
"""
import importlib, sys, traceback, shutil
from pathlib import Path

PASS = "PASS"; FAIL = "FAIL"
results = []

def check(name, fn):
    try:
        fn()
        results.append((PASS, name, ""))
        print(f"PASS  {name}")
    except Exception as exc:
        results.append((FAIL, name, repr(exc)))
        print(f"FAIL  {name}: {exc!r}")

# ---------- bootstrap: point at a test DB ----------
import db
REAL_DB = db.DB_PATH
TEST_DB = str(Path(REAL_DB).with_name("eval_shop.db"))
if not Path(REAL_DB).exists():
    print(f"[error] {REAL_DB} not found – run 'python db.py' first"); sys.exit(1)
shutil.copy(REAL_DB, TEST_DB)
db.DB_PATH = TEST_DB

import tools as store, orders as ord_
ord_.init_orders_schema()
ord_.INVOICE_DIR = "eval_invoices"

# ---------- helpers ----------
def raw(sql, params=()):
    import sqlite3
    conn = sqlite3.connect(TEST_DB); conn.row_factory = sqlite3.Row
    try: return conn.execute(sql, params).fetchall()
    finally: conn.close()

def fresh_cart(tid="eval"):
    import sqlite3
    conn = sqlite3.connect(TEST_DB)
    conn.execute("DELETE FROM cart_items"); conn.commit(); conn.close()

P1, P2 = [r["id"] for r in raw("SELECT id FROM products ORDER BY id LIMIT 2")]

# ============================================================
# A) Product Search
# ============================================================
def t_search_english():
    r = store.search_products("cable"); assert len(r) > 0

def t_search_arabic_keyword():
    # Arabic queries are translated to English by the agent; here we test
    # the underlying tool with the English translation.
    r = store.search_products("charger"); assert isinstance(r, list)

def t_search_max_price():
    r = store.search_products("cable", max_price=500)
    assert all(p["price"] <= 500 for p in r)

def t_search_min_rating():
    r = store.search_products("cable", min_rating=4.0)
    assert all(p["rating"] >= 4.0 for p in r)

def t_search_no_match():
    assert store.search_products("zzzzzznoproduct_xyz") == []

def t_search_limit():
    r = store.search_products("cable"); assert len(r) <= 5

# ============================================================
# B) Product Comparison
# ============================================================
def t_compare_two():
    ids = [p["id"] for p in store.search_products("cable")[:2]]
    res = store.compare_products(ids)
    assert "cheapest_id" in res and res["cheapest_id"] in ids

def t_compare_needs_two():
    assert "error" in store.compare_products([P1])

def t_product_details():
    d = store.get_product_details(P1); assert "description" in d

def t_product_details_missing():
    assert "error" in store.get_product_details(999999)

# ============================================================
# C) Tool Selection / Stock
# ============================================================
def t_check_stock_available():
    res = store.check_stock(P1, quantity=1)
    assert isinstance(res["available"], bool)

def t_check_stock_too_much():
    assert store.check_stock(P1, quantity=99999)["available"] is False

# ============================================================
# D) Conversational Context – cart operations
# ============================================================
def t_add_to_cart():
    fresh_cart()
    ord_.add_to_cart("eval", P1, 2)
    assert ord_.view_cart("eval")["items"][0]["quantity"] == 2

def t_cart_isolated_per_session():
    fresh_cart()
    ord_.add_to_cart("alice", P1, 1)
    assert ord_.view_cart("bob")["item_count"] == 0

def t_over_stock_refused():
    fresh_cart()
    assert "error" in ord_.add_to_cart("eval", P1, 99999)

def t_remove_from_cart():
    fresh_cart()
    ord_.add_to_cart("eval", P1, 3)
    ord_.remove_from_cart("eval", P1, 1)
    assert ord_.view_cart("eval")["items"][0]["quantity"] == 2

# ============================================================
# E) Response Correctness – order flow
# ============================================================
def t_create_order():
    fresh_cart()
    ord_.add_to_cart("eval", P1, 1)
    res = ord_.create_order("eval", customer_id=1)
    assert "order_id" in res

def t_empty_cart_order_refused():
    fresh_cart()
    assert "error" in ord_.create_order("eval")

def t_invoice_created():
    fresh_cart()
    ord_.add_to_cart("eval", P1, 1)
    res = ord_.create_order("eval", customer_id=1)
    if "order_id" in res:
        inv = ord_.generate_invoice(res["order_id"])
        assert Path(inv["invoice_path"]).exists()

# ============================================================
# F) RAG – return/warranty questions
# ============================================================
def t_rag_relevant_answer():
    try:
        import rag
        ans = rag.answer("What is the return policy?")
        assert ans and "30" in ans or "day" in ans.lower() or "return" in ans.lower()
    except ImportError:
        raise AssertionError("rag.py not found – RAG not yet installed")

def t_rag_warranty():
    try:
        import rag
        ans = rag.answer("Is there a warranty on electronics?")
        assert ans and len(ans) > 10
    except ImportError:
        raise AssertionError("rag.py not found")

def t_rag_unsupported_question():
    try:
        import rag
        ans = rag.answer("What is the capital of France?")
        # Must not invent a policy; should say information not found
        assert "not" in ans.lower() or "unavailable" in ans.lower() or "capital" not in ans.lower()
    except ImportError:
        raise AssertionError("rag.py not found")

# ============================================================
# Run all
# ============================================================
TESTS = [
    # Search
    ("search_english",       t_search_english),
    ("search_arabic_kw",     t_search_arabic_keyword),
    ("search_max_price",     t_search_max_price),
    ("search_min_rating",    t_search_min_rating),
    ("search_no_match",      t_search_no_match),
    ("search_limit_5",       t_search_limit),
    # Comparison
    ("compare_two",          t_compare_two),
    ("compare_needs_two",    t_compare_needs_two),
    ("product_details",      t_product_details),
    ("product_missing",      t_product_details_missing),
    # Stock
    ("stock_available",      t_check_stock_available),
    ("stock_too_much",       t_check_stock_too_much),
    # Cart
    ("add_to_cart",          t_add_to_cart),
    ("cart_isolated",        t_cart_isolated_per_session),
    ("over_stock_refused",   t_over_stock_refused),
    ("remove_from_cart",     t_remove_from_cart),
    # Order flow
    ("create_order",         t_create_order),
    ("empty_cart_refused",   t_empty_cart_order_refused),
    ("invoice_created",      t_invoice_created),
    # RAG
    ("rag_relevant",         t_rag_relevant_answer),
    ("rag_warranty",         t_rag_warranty),
    ("rag_unsupported",      t_rag_unsupported_question),
]

print("=" * 55)
print("ShopAgent-AI  –  CP6 Evaluation Suite")
print("=" * 55)
for name, fn in TESTS:
    check(name, fn)

passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
print("=" * 55)
print(f"Result: {passed}/{len(TESTS)} passed, {failed} failed")
print("=" * 55)
