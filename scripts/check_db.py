import sqlite3

c = sqlite3.connect("data/shop.db")
print("products:", c.execute("SELECT COUNT(*) FROM products").fetchone()[0])
print("customers:", c.execute("SELECT COUNT(*) FROM customers").fetchone()[0])
print("invalid prices:", c.execute("SELECT COUNT(*) FROM products WHERE price IS NULL OR price <= 0").fetchone()[0])
for row in c.execute("SELECT name, category, price, stock, rating FROM products LIMIT 3"):
    print(row)