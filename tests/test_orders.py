"""
test_orders.py: checks the CP4 cart / order / invoice logic.

Run:  python tests/test_orders.py

The tests run on a COPY of the database (data/test_shop.db), so your real
stock is never touched.
"""

import shutil
import sqlite3
from pathlib import Path

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so "from shopagent import ..." works when this file is run directly

from shopagent import db  # noqa: E402

REAL_DB = db.DB_PATH
TEST_DB = str(Path(REAL_DB).with_name("test_shop.db"))

# Point every module at the copy BEFORE importing orders
shutil.copy(REAL_DB, TEST_DB)
db.DB_PATH = TEST_DB

from shopagent import orders  # noqa: E402

orders.init_orders_schema()
orders.INVOICE_DIR = str(ROOT / "test_invoices")


def raw(sql, params=()):
    conn = sqlite3.connect(TEST_DB)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def run(sql, params=()):
    conn = sqlite3.connect(TEST_DB)
    conn.execute(sql, params)
    conn.commit()
    conn.close()


def stock_of(pid):
    return raw("SELECT stock FROM products WHERE id = ?", (pid,))[0]["stock"]


# Deterministic test products: two items with known stock
P1, P2 = [r["id"] for r in raw("SELECT id FROM products ORDER BY id LIMIT 2")]


def fresh():
    """Reset stock and carts before every test."""
    run("UPDATE products SET stock = 10 WHERE id IN (?, ?)", (P1, P2))
    run("DELETE FROM cart_items")


def test_add_and_view_cart():
    fresh()
    orders.add_to_cart("t1", P1, 2)
    cart = orders.view_cart("t1")
    assert cart["item_count"] == 1
    assert cart["items"][0]["quantity"] == 2
    assert cart["subtotal"] > 0


def test_add_same_product_twice_adds_up():
    fresh()
    orders.add_to_cart("t1", P1, 2)
    orders.add_to_cart("t1", P1, 3)
    assert orders.view_cart("t1")["items"][0]["quantity"] == 5


def test_add_more_than_stock_is_refused():
    fresh()
    result = orders.add_to_cart("t1", P1, 11)
    assert "error" in result
    assert orders.view_cart("t1")["item_count"] == 0


def test_add_unknown_product():
    fresh()
    assert "error" in orders.add_to_cart("t1", 999999, 1)


def test_carts_are_separate_per_thread():
    fresh()
    orders.add_to_cart("alice", P1, 1)
    assert orders.view_cart("bob")["item_count"] == 0


def test_empty_cart_message_is_not_an_empty_result():
    fresh()
    cart = orders.view_cart("nobody")
    assert cart["items"] == [] and "message" in cart


def test_remove_from_cart():
    fresh()
    orders.add_to_cart("t1", P1, 3)
    orders.remove_from_cart("t1", P1, 1)
    assert orders.view_cart("t1")["items"][0]["quantity"] == 2
    orders.remove_from_cart("t1", P1)
    assert orders.view_cart("t1")["item_count"] == 0


def test_create_order_deducts_stock_and_clears_cart():
    fresh()
    orders.add_to_cart("t1", P1, 3)
    orders.add_to_cart("t1", P2, 2)
    result = orders.create_order("t1", customer_id=1)
    assert "order_id" in result, result
    assert stock_of(P1) == 7 and stock_of(P2) == 8
    assert orders.view_cart("t1")["item_count"] == 0
    assert len(raw("SELECT * FROM order_items WHERE order_id = ?", (result["order_id"],))) == 2
    saved_total = raw("SELECT total FROM orders WHERE id = ?", (result["order_id"],))[0]["total"]
    assert saved_total == result["total"]


def test_create_order_with_empty_cart():
    fresh()
    assert "error" in orders.create_order("t1")


def test_create_order_unknown_customer():
    fresh()
    orders.add_to_cart("t1", P1, 1)
    assert "error" in orders.create_order("t1", customer_id=99999)
    assert stock_of(P1) == 10


def test_order_rolls_back_when_stock_changed():
    fresh()
    orders.add_to_cart("t1", P1, 5)
    run("UPDATE products SET stock = 2 WHERE id = ?", (P1,))  # someone else bought it
    orders_before = len(raw("SELECT id FROM orders"))
    result = orders.create_order("t1")
    assert "error" in result
    assert stock_of(P1) == 2  # untouched
    assert len(raw("SELECT id FROM orders")) == orders_before  # no half-saved order
    assert orders.view_cart("t1")["item_count"] == 1  # cart kept


def test_invoice_pdf_is_created():
    fresh()
    orders.add_to_cart("t1", P1, 1)
    order = orders.create_order("t1")
    invoice = orders.generate_invoice(order["order_id"])
    assert Path(invoice["invoice_path"]).exists()
    assert "test_invoices" in invoice["invoice_path"]  # tests must not write into invoices/
    assert Path(invoice["invoice_path"]).stat().st_size > 500


def test_invoice_unknown_order():
    assert "error" in orders.generate_invoice(999999)


if __name__ == "__main__":
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # report and keep going
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    raise SystemExit(1 if failed else 0)
