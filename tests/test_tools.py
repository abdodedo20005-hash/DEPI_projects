"""
test_tools.py: checks that the CP2 tools work on their own.

Run:  python tests/test_tools.py
(Requires data/shop.db from CP1.)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so "from shopagent import ..." works when this file is run directly

from shopagent.tools import check_stock, compare_products, get_product_details, search_products  # noqa: E402


def test_search_returns_results():
    results = search_products("cable")
    assert len(results) > 0, "search should find at least one cable"
    assert len(results) <= 5


def test_search_respects_max_price():
    results = search_products("cable", max_price=500)
    assert all(p["price"] <= 500 for p in results)


def test_search_respects_min_rating():
    results = search_products("cable", min_rating=4.0)
    assert all(p["rating"] >= 4.0 for p in results)


def test_search_no_match():
    assert search_products("zzzzzz-not-a-product") == []


def test_product_details():
    first = search_products("cable")[0]
    details = get_product_details(first["id"])
    assert details["id"] == first["id"]
    assert "description" in details


def test_product_details_missing():
    assert "error" in get_product_details(999999)


def test_compare_products():
    ids = [p["id"] for p in search_products("cable")[:2]]
    result = compare_products(ids)
    assert len(result["products"]) == 2
    assert result["cheapest_id"] in ids


def test_compare_needs_two_products():
    assert "error" in compare_products([1])


def test_check_stock():
    first = search_products("cable")[0]
    result = check_stock(first["id"], quantity=1)
    assert result["product_id"] == first["id"]
    assert isinstance(result["available"], bool)


def test_check_stock_too_much():
    first = search_products("cable")[0]
    assert check_stock(first["id"], quantity=10_000)["available"] is False


if __name__ == "__main__":
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # report the failure and keep going
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    raise SystemExit(1 if failed else 0)
