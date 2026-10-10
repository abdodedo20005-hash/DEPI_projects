"""
orders.py: cart, orders and invoices for ShopAgent-AI (Checkpoint 4)

Plain Python functions (no LLM). In the notebook they are wrapped as tools.

Design rules:
- The cart lives in the database (table cart_items), keyed by thread_id,
  NOT in the LLM's memory (the LLM can forget or hallucinate).
- create_order runs in ONE transaction: check stock, write the order,
  deduct stock, clear the cart. Either everything happens or nothing does.
- Every function returns plain dicts and never an empty list, because some
  LLM APIs reject empty tool results.
"""

from datetime import datetime
from pathlib import Path

from db import get_conn

TAX_RATE = 0.0  # set e.g. 0.14 to add VAT to every order
INVOICE_DIR = "invoices"

CART_SCHEMA = """
CREATE TABLE IF NOT EXISTS cart_items (
    thread_id  TEXT    NOT NULL,
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity   INTEGER NOT NULL CHECK (quantity > 0),
    PRIMARY KEY (thread_id, product_id)
);
"""


def init_orders_schema() -> None:
    """Create the cart table if it doesn't exist (orders tables come from db.py)."""
    with get_conn() as conn:
        conn.executescript(CART_SCHEMA)


# ------------------------------------------------------------------
# Cart
# ------------------------------------------------------------------
def view_cart(thread_id: str) -> dict:
    """Return the cart contents with line totals and the cart total."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT p.id AS product_id, p.name, p.price AS unit_price, c.quantity "
            "FROM cart_items c JOIN products p ON p.id = c.product_id "
            "WHERE c.thread_id = ? ORDER BY p.id",
            (thread_id,),
        ).fetchall()

    items = []
    for r in rows:
        item = dict(r)
        item["line_total"] = round(item["unit_price"] * item["quantity"], 2)
        items.append(item)

    total = round(sum(i["line_total"] for i in items), 2)
    result = {"items": items, "item_count": len(items), "subtotal": total}
    if not items:
        result["message"] = "The cart is empty."
    return result


def add_to_cart(thread_id: str, product_id: int, quantity: int = 1) -> dict:
    """Add a product to the cart, refusing quantities above the available stock."""
    if quantity < 1:
        return {"error": "Quantity must be at least 1"}

    with get_conn() as conn:
        product = conn.execute(
            "SELECT id, name, stock FROM products WHERE id = ?", (product_id,)
        ).fetchone()
        if product is None:
            return {"error": f"Product {product_id} not found"}

        row = conn.execute(
            "SELECT quantity FROM cart_items WHERE thread_id = ? AND product_id = ?",
            (thread_id, product_id),
        ).fetchone()
        wanted = (row["quantity"] if row else 0) + quantity

        if wanted > product["stock"]:
            return {
                "error": "Not enough stock",
                "in_stock": product["stock"],
                "already_in_cart": row["quantity"] if row else 0,
            }

        conn.execute(
            "INSERT INTO cart_items (thread_id, product_id, quantity) VALUES (?, ?, ?) "
            "ON CONFLICT(thread_id, product_id) DO UPDATE SET quantity = excluded.quantity",
            (thread_id, product_id, wanted),
        )
    return {"added": product["name"][:60], "quantity_in_cart": wanted, "cart": view_cart(thread_id)}


def remove_from_cart(thread_id: str, product_id: int, quantity: int | None = None) -> dict:
    """Remove a product from the cart. If quantity is None, remove it completely."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT quantity FROM cart_items WHERE thread_id = ? AND product_id = ?",
            (thread_id, product_id),
        ).fetchone()
        if row is None:
            return {"error": f"Product {product_id} is not in the cart"}

        if quantity is None or quantity >= row["quantity"]:
            conn.execute(
                "DELETE FROM cart_items WHERE thread_id = ? AND product_id = ?",
                (thread_id, product_id),
            )
        elif quantity < 1:
            return {"error": "Quantity must be at least 1"}
        else:
            conn.execute(
                "UPDATE cart_items SET quantity = quantity - ? "
                "WHERE thread_id = ? AND product_id = ?",
                (quantity, thread_id, product_id),
            )
    return {"removed": True, "cart": view_cart(thread_id)}


def clear_cart(thread_id: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM cart_items WHERE thread_id = ?", (thread_id,))


# ------------------------------------------------------------------
# Order (one transaction)
# ------------------------------------------------------------------
def create_order(thread_id: str, customer_id: int = 1) -> dict:
    """Turn the cart into an order: check stock, save the order, deduct stock,
    clear the cart. All-or-nothing."""
    conn = get_conn()
    try:
        conn.execute("BEGIN IMMEDIATE")  # lock the database for the whole operation

        cart = conn.execute(
            "SELECT c.product_id, c.quantity, p.name, p.price, p.stock "
            "FROM cart_items c JOIN products p ON p.id = c.product_id "
            "WHERE c.thread_id = ?",
            (thread_id,),
        ).fetchall()
        if not cart:
            conn.rollback()
            return {"error": "The cart is empty"}

        if conn.execute("SELECT 1 FROM customers WHERE id = ?", (customer_id,)).fetchone() is None:
            conn.rollback()
            return {"error": f"Customer {customer_id} not found"}

        # Stock may have changed since the item was added to the cart
        problems = [
            {"product_id": r["product_id"], "name": r["name"][:60],
             "wanted": r["quantity"], "in_stock": r["stock"]}
            for r in cart if r["quantity"] > r["stock"]
        ]
        if problems:
            conn.rollback()
            return {"error": "Some items are no longer available in that quantity", "items": problems}

        subtotal = round(sum(r["price"] * r["quantity"] for r in cart), 2)
        tax = round(subtotal * TAX_RATE, 2)
        total = round(subtotal + tax, 2)

        order_id = conn.execute(
            "INSERT INTO orders (customer_id, total, status) VALUES (?, ?, 'confirmed')",
            (customer_id, total),
        ).lastrowid

        for r in cart:
            conn.execute(
                "INSERT INTO order_items (order_id, product_id, quantity, unit_price) "
                "VALUES (?, ?, ?, ?)",
                (order_id, r["product_id"], r["quantity"], r["price"]),
            )
            # The "stock >= ?" guard protects against a concurrent order
            updated = conn.execute(
                "UPDATE products SET stock = stock - ? WHERE id = ? AND stock >= ?",
                (r["quantity"], r["product_id"], r["quantity"]),
            )
            if updated.rowcount != 1:
                raise RuntimeError(f"Stock changed for product {r['product_id']}")

        conn.execute("DELETE FROM cart_items WHERE thread_id = ?", (thread_id,))
        conn.commit()
    except Exception as exc:
        conn.rollback()
        return {"error": f"Order failed and nothing was changed: {exc}"}
    finally:
        conn.close()

    return {"order_id": order_id, "status": "confirmed", "subtotal": subtotal, "tax": tax, "total": total}


def get_order(order_id: int) -> dict:
    """Return an order with its customer and line items."""
    with get_conn() as conn:
        order = conn.execute(
            "SELECT o.id, o.total, o.status, o.created_at, c.name AS customer_name "
            "FROM orders o JOIN customers c ON c.id = o.customer_id WHERE o.id = ?",
            (order_id,),
        ).fetchone()
        if order is None:
            return {"error": f"Order {order_id} not found"}
        items = conn.execute(
            "SELECT p.name, oi.quantity, oi.unit_price "
            "FROM order_items oi JOIN products p ON p.id = oi.product_id "
            "WHERE oi.order_id = ?",
            (order_id,),
        ).fetchall()
    return {**dict(order), "items": [dict(i) for i in items]}


# ------------------------------------------------------------------
# Invoice (PDF)
# ------------------------------------------------------------------
def _latin(text: str, limit: int = 60) -> str:
    """The built-in PDF fonts only support latin-1, so replace anything else."""
    text = str(text)[:limit]
    return text.encode("latin-1", "replace").decode("latin-1")


def generate_invoice(order_id: int, out_dir: str = INVOICE_DIR) -> dict:
    """Create a PDF invoice for an order and return its file path."""
    from fpdf import FPDF  # pip install fpdf2

    order = get_order(order_id)
    if "error" in order:
        return order

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, "ShopAgent-AI", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Invoice #{order['id']}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Date: {order['created_at']}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Customer: {_latin(order['customer_name'], 40)}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # Table header
    pdf.set_font("Helvetica", "B", 11)
    widths = (95, 20, 35, 40)
    for w, title in zip(widths, ("Item", "Qty", "Unit price", "Line total")):
        pdf.cell(w, 8, title, border=1)
    pdf.ln()

    # Table rows
    pdf.set_font("Helvetica", "", 10)
    subtotal = 0.0
    for item in order["items"]:
        line = round(item["unit_price"] * item["quantity"], 2)
        subtotal += line
        pdf.cell(widths[0], 8, _latin(item["name"]), border=1)
        pdf.cell(widths[1], 8, str(item["quantity"]), border=1)
        pdf.cell(widths[2], 8, f"{item['unit_price']:.2f}", border=1)
        pdf.cell(widths[3], 8, f"{line:.2f}", border=1)
        pdf.ln()

    # Totals (all amounts in INR, the currency of the dataset)
    tax = round(subtotal * TAX_RATE, 2)
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Subtotal: {subtotal:.2f} INR", align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Tax ({TAX_RATE:.0%}): {tax:.2f} INR", align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 9, f"Total: {order['total']:.2f} INR", align="R", new_x="LMARGIN", new_y="NEXT")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"invoice_{order_id}_{datetime.now():%Y%m%d_%H%M%S}.pdf"
    pdf.output(str(path))
    return {"invoice_path": str(path), "order_id": order_id, "total": order["total"]}
