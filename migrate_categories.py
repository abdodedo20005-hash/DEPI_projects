"""
migrate_categories.py: add the full category path to the existing database

Why: db.py originally kept only the FIRST part of each category, and the product
names often don't contain words like "smartphone", so searching by type failed.
This script adds two columns (category_path, sub_category) WITHOUT touching
product ids, stock, carts or orders.

Run once:  python migrate_categories.py
It is safe to run again.
"""

import pandas as pd

import db


def main() -> None:
    df = pd.read_csv(db.CSV_PATH, usecols=["product_name", "category"])
    df = df.dropna().drop_duplicates(subset=["product_name"])  # same rule as db.load_products
    path_by_name = dict(zip(df["product_name"], df["category"].astype(str)))

    with db.get_conn() as conn:
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(products)")}
        if "category_path" not in columns:
            conn.execute("ALTER TABLE products ADD COLUMN category_path TEXT")
        if "sub_category" not in columns:
            conn.execute("ALTER TABLE products ADD COLUMN sub_category TEXT")

        updated = 0
        for row in conn.execute("SELECT id, name FROM products").fetchall():
            path = path_by_name.get(row["name"])
            if path:
                conn.execute(
                    "UPDATE products SET category_path = ?, sub_category = ? WHERE id = ?",
                    (path, path.split("|")[-1], row["id"]),
                )
                updated += 1
        conn.commit()
        total = conn.execute("SELECT COUNT(*) AS n FROM products").fetchone()["n"]

    print(f"Updated {updated} of {total} products")
    if updated < total:
        print("Some products have no category path (their names were not found in the CSV).")


if __name__ == "__main__":
    main()
