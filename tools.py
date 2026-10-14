"""
tools.py: store tools for ShopAgent-AI (Checkpoint 2)

Plain Python functions that talk to the database. In CP3 we will wrap
them with LangChain's @tool so the agent can call them.

Rules:
- Always use ? placeholders (never build SQL from user text directly)
- Every function returns plain dicts/lists, so the LLM can read the result
"""

from db import get_conn

MAX_RESULTS = 5


def search_products(
    query: str,
    max_price: float | None = None,
    min_rating: float | None = None,
    category: str | None = None,
) -> list[dict]:
    """Search store products by keyword. Optionally filter by max price,
    minimum rating and category. Returns up to 5 products, best rated first.

    Every word must appear in the product name OR in its category path, so
    "smartphones" finds phones even when the name never says "smartphone"."""
    sql = (
        "SELECT id, name, category, sub_category, price, stock, rating "
        "FROM products WHERE 1=1"
    )
    params: list = []

    for word in query.lower().split():
        if len(word) > 3 and word.endswith("s"):
            word = word[:-1]  # "cables" -> "cable"
        sql += " AND (name LIKE ? OR category_path LIKE ?)"
        params += [f"%{word}%", f"%{word}%"]

    if max_price is not None:
        sql += " AND price <= ?"
        params.append(max_price)
    if min_rating is not None:
        sql += " AND rating >= ?"
        params.append(min_rating)
    if category:
        sql += " AND category_path LIKE ?"
        params.append(f"%{category}%")

    sql += " ORDER BY rating DESC LIMIT ?"
    params.append(MAX_RESULTS)

    with get_conn() as conn:
        return [dict(row) for row in conn.execute(sql, params)]


def get_product_details(product_id: int) -> dict:
    """Return full details of one product, or an error dict if it doesn't exist."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, name, category, brand, price, stock, rating, description "
            "FROM products WHERE id = ?",
            (product_id,),
        ).fetchone()
    if row is None:
        return {"error": f"Product {product_id} not found"}
    return dict(row)


def compare_products(product_ids: list[int]) -> dict:
    """Compare 2 to 4 products by price, rating and stock.
    Also reports the cheapest and the best rated."""
    if not 2 <= len(product_ids) <= 4:
        return {"error": "Provide between 2 and 4 product ids"}

    placeholders = ",".join("?" for _ in product_ids)
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT id, name, price, rating, stock FROM products "
            f"WHERE id IN ({placeholders})",
            product_ids,
        ).fetchall()

    products = [dict(r) for r in rows]
    if len(products) != len(set(product_ids)):
        return {"error": "One or more products were not found"}

    cheapest = min(products, key=lambda p: p["price"])
    # Products without a rating are ignored when picking the best rated
    rated = [p for p in products if p["rating"] is not None]
    best_rated = max(rated, key=lambda p: p["rating"]) if rated else None

    return {
        "products": products,
        "cheapest_id": cheapest["id"],
        "best_rated_id": best_rated["id"] if best_rated else None,
    }


def check_stock(product_id: int, quantity: int = 1) -> dict:
    """Check whether the requested quantity of a product is in stock."""
    if quantity < 1:
        return {"error": "Quantity must be at least 1"}

    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, name, stock FROM products WHERE id = ?", (product_id,)
        ).fetchone()
    if row is None:
        return {"error": f"Product {product_id} not found"}

    return {
        "product_id": row["id"],
        "name": row["name"],
        "available": row["stock"] >= quantity,
        "in_stock": row["stock"],
        "requested": quantity,
    }
