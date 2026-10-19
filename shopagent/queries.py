"""
queries.py: read-only queries for the catalog browser and the admin dashboard

Needs the category_path / sub_category columns (see migrate_categories.py).
"""

import re

from shopagent.db import get_conn

LOW_STOCK_DEFAULT = 5


def pretty(label: str) -> str:
    """'Computers&Accessories' -> 'Computers & Accessories', 'USBCables' -> 'USB Cables'."""
    label = label.replace("&", " & ")
    label = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", label)
    label = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", label)
    return re.sub(r"\s+", " ", label).strip()


# ------------------------------------------------------------------
# Catalog
# ------------------------------------------------------------------
def category_tree() -> dict[str, dict[str, int]]:
    """{main category: {type: number of products}} built from the category paths."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT category_path, COUNT(*) AS n FROM products "
            "WHERE category_path IS NOT NULL GROUP BY category_path"
        ).fetchall()

    tree: dict[str, dict[str, int]] = {}
    for row in rows:
        parts = row["category_path"].split("|")
        top = parts[0]
        sub = parts[1] if len(parts) > 1 else "Other"
        tree.setdefault(top, {})
        tree[top][sub] = tree[top].get(sub, 0) + row["n"]
    return tree


def products_in_group(top: str, sub: str, limit: int = 15) -> list[dict]:
    """Best rated products of one main category and type."""
    if sub == "Other":
        where, params = "category_path = ?", [top]
    else:
        where, params = "(category_path = ? OR category_path LIKE ?)", [f"{top}|{sub}", f"{top}|{sub}|%"]
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT id, name, price, stock, rating FROM products WHERE {where} "
            "ORDER BY rating DESC, id LIMIT ?",
            (*params, limit),
        ).fetchall()
    return [dict(r) for r in rows]


# ------------------------------------------------------------------
# Admin
# ------------------------------------------------------------------
def stock_overview(low_threshold: int = LOW_STOCK_DEFAULT) -> dict:
    with get_conn() as conn:
        p = conn.execute(
            "SELECT COUNT(*) AS products, COALESCE(SUM(stock), 0) AS units, "
            "COALESCE(SUM(stock = 0), 0) AS out_of_stock, "
            "COALESCE(SUM(stock > 0 AND stock <= ?), 0) AS low_stock FROM products",
            (low_threshold,),
        ).fetchone()
        o = conn.execute(
            "SELECT COUNT(*) AS orders, COALESCE(SUM(total), 0) AS revenue FROM orders"
        ).fetchone()
    return {**dict(p), **dict(o)}


def stock_table(search: str = "", category: str = "All", only_low: bool = False,
                low_threshold: int = LOW_STOCK_DEFAULT, limit: int = 500) -> list[dict]:
    sql = "SELECT id, name, category, sub_category, price, stock, rating FROM products WHERE 1=1"
    params: list = []
    for word in search.split():
        sql += " AND name LIKE ?"
        params.append(f"%{word}%")
    if category and category != "All":
        sql += " AND category = ?"
        params.append(category)
    if only_low:
        sql += " AND stock <= ?"
        params.append(low_threshold)
    sql += " ORDER BY stock ASC, id LIMIT ?"
    params.append(limit)
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(sql, params)]


def stock_by_category() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT category, SUM(stock) AS units FROM products "
            "GROUP BY category ORDER BY units DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def category_names() -> list[str]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT category FROM products WHERE category IS NOT NULL ORDER BY category"
        ).fetchall()
    return [r["category"] for r in rows]


def recent_orders(limit: int = 10) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT o.id, o.created_at, c.name AS customer, o.total, o.status, "
            "GROUP_CONCAT(p.name || ' x' || oi.quantity, ' | ') AS items "
            "FROM orders o JOIN customers c ON c.id = o.customer_id "
            "LEFT JOIN order_items oi ON oi.order_id = o.id "
            "LEFT JOIN products p ON p.id = oi.product_id "
            "GROUP BY o.id ORDER BY o.id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
