"""
db.py: database layer for ShopAgent-AI (Checkpoint 1)

This file does 4 things:
1. Creates the SQLite tables (products, customers, orders, order_items)
2. Loads the Kaggle CSV into the products table
3. Generates stock levels and sample customers
4. Runs a test search query to verify everything works

Run:  python db.py
"""

import random
import re
import sqlite3
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None  # pandas is optional for test runs

# ------------------------------------------------------------------
# Settings: adjust to match your CSV file
# ------------------------------------------------------------------
DB_PATH = "data/shop.db"
CSV_PATH = "data/amazon.csv"

# CSV column name (left) -> our column name (right)
COLUMN_MAP = {
    "product_name": "name",
    "category": "category",
    "discounted_price": "price",
    "rating": "rating",
    "about_product": "description",
}

random.seed(42)  # same numbers on every run


# ------------------------------------------------------------------
# 1) Connection and schema
# ------------------------------------------------------------------
def get_conn() -> sqlite3.Connection:
    """Return a connection to the database. Rows behave like dicts."""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    category    TEXT,
    brand       TEXT,
    price       REAL NOT NULL,
    stock       INTEGER NOT NULL DEFAULT 0,
    description TEXT,
    rating      REAL,
    category_path TEXT,
    sub_category  TEXT
);

CREATE TABLE IF NOT EXISTS customers (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name  TEXT NOT NULL,
    email TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS orders (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    total       REAL NOT NULL,
    status      TEXT NOT NULL DEFAULT 'confirmed',
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS order_items (
    order_id   INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity   INTEGER NOT NULL,
    unit_price REAL NOT NULL
);
"""


def init_db() -> None:
    """Create the tables if they don't exist yet."""
    with get_conn() as conn:
        conn.executescript(SCHEMA)


# ------------------------------------------------------------------
# 2) Load products from the CSV
# ------------------------------------------------------------------
def _clean_price(value) -> float | None:
    """Turn '₹1,099' or '1,099' into 1099.0"""
    digits = re.sub(r"[^\d.]", "", str(value))
    return float(digits) if digits else None


def _guess_brand(name: str) -> str:
    """The brand is usually the first word of the product name (rough guess)."""
    return str(name).split()[0] if str(name).strip() else "Unknown"


def load_products() -> int:
    """Read the CSV and load products into the products table. Return the count.
    If pandas is not available, skip loading and return 0.
    """
    if pd is None:
        print("[warn] pandas not installed – skipping product load.")
        return 0
    df = pd.read_csv(CSV_PATH)
    df = df.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())]

    df["price"] = df["price"].apply(_clean_price)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    # Category looks like "A|B|C": keep the full path, the last part and the first part
    df["category_path"] = df["category"].astype(str)
    df["sub_category"] = df["category_path"].str.split("|").str[-1]
    df["category"] = df["category_path"].str.split("|").str[0]
    df["brand"] = df["name"].apply(_guess_brand)
    df["stock"] = [random.randint(0, 50) for _ in range(len(df))]

    df = df.dropna(subset=["name", "price"]).drop_duplicates(subset=["name"])

    with get_conn() as conn:
        conn.execute("DELETE FROM products")  # re-running won't duplicate data
        df.to_sql("products", conn, if_exists="append", index=False)
    return len(df)


# ------------------------------------------------------------------
# 3) Sample customers
# ------------------------------------------------------------------
def seed_customers(n: int = 20) -> None:
    """Generate n sample customers."""
    first = ["Ahmed", "Mona", "Omar", "Sara", "Youssef", "Nour", "Khaled", "Hana"]
    last = ["Hassan", "Ali", "Mahmoud", "Ibrahim", "Salem", "Fathy"]
    with get_conn() as conn:
        for i in range(n):
            name = f"{random.choice(first)} {random.choice(last)}"
            conn.execute(
                "INSERT OR IGNORE INTO customers (name, email) VALUES (?, ?)",
                (name, f"user{i}@example.com"),
            )


# ------------------------------------------------------------------
# 4) Test query (same shape as the tool we'll write in CP2)
# ------------------------------------------------------------------
def search_products(query: str, max_price: float | None = None, limit: int = 5):
    """Search by name, optionally with a maximum price."""
    sql = "SELECT id, name, price, stock, rating FROM products WHERE name LIKE ?"
    params: list = [f"%{query}%"]
    if max_price is not None:
        sql += " AND price <= ?"
        params.append(max_price)
    sql += " ORDER BY rating DESC LIMIT ?"
    params.append(limit)
    with get_conn() as conn:
        return [dict(row) for row in conn.execute(sql, params)]


# ------------------------------------------------------------------
# Run Checkpoint 1
# ------------------------------------------------------------------
if __name__ == "__main__":
    init_db()
    count = load_products()
    seed_customers()
    print(f"Loaded {count} products")

    # CP1 success criterion: this query must return results
    for row in search_products("cable", max_price=1000):
        print(row)
