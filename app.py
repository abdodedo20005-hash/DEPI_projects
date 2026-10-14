"""
app.py: Streamlit UI for ShopAgent-AI (Checkpoint 5)

Run:  python -m streamlit run app.py

Layout:
- Left column: the catalog (main categories -> types -> products with "Add to cart").
- Right column: the chat with the agent.
- Sidebar: customer, live cart, "Confirm order" button and invoice download.
- "Admin" page (menu in the sidebar): live stock dashboard (pages/1_Admin.py).

The agent can search, compare and fill the cart, but it CANNOT place an order.
The "Confirm order" button places it by calling orders.create_order directly.
"""

import uuid
from pathlib import Path

import streamlit as st

import agent
import db
import orders
import queries

st.set_page_config(page_title="ShopAgent-AI", page_icon="🛒", layout="wide")

# Each paragraph picks its own direction, so Arabic and English both look right
st.markdown(
    """
    <style>
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li { unicode-bidi: plaintext; text-align: start; }
    </style>
    """,
    unsafe_allow_html=True,
)

BACKUP = Path(db.DB_PATH).with_name("shop_backup.db")


# ------------------------------------------------------------------
# Helpers and callbacks (callbacks run BEFORE the page is redrawn)
# ------------------------------------------------------------------
def new_conversation():
    """Start a fresh conversation: new thread id (so a new cart) and empty history."""
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I'm ShopAgent. Tell me what you are looking for, in Arabic or English, "
                       "or browse the catalog on the left.",
        }
    ]
    st.session_state.last_order = None


def select_group(top, sub):
    st.session_state.browse = (top, sub)


def close_browse():
    st.session_state.browse = None


def add_from_catalog(product_id):
    result = orders.add_to_cart(st.session_state.thread_id, product_id, 1)
    st.session_state.flash = (
        "Added to your cart." if "error" not in result else f"Could not add: {result['error']}"
    )


def list_customers():
    with db.get_conn() as conn:
        rows = conn.execute("SELECT id, name FROM customers ORDER BY id").fetchall()
    return [(r["id"], r["name"]) for r in rows]


def reset_demo_data():
    """Restore the original stock and delete all carts and orders. Returns True on success."""
    if not BACKUP.exists():
        return False
    with db.get_conn() as conn:
        conn.execute("ATTACH DATABASE ? AS bk", (str(BACKUP),))
        conn.execute(
            "UPDATE products SET stock = (SELECT b.stock FROM bk.products b WHERE b.id = products.id)"
        )
        conn.execute("DELETE FROM order_items")
        conn.execute("DELETE FROM orders")
        conn.execute("DELETE FROM cart_items")
        conn.commit()
        conn.execute("DETACH DATABASE bk")
    return True


if "thread_id" not in st.session_state:
    new_conversation()
st.session_state.setdefault("browse", None)

flash = st.session_state.pop("flash", None)
if flash:
    st.toast(flash)

thread_id = st.session_state.thread_id


# ------------------------------------------------------------------
# Sidebar: customer, live cart, confirm button, invoice
# ------------------------------------------------------------------
with st.sidebar:
    st.title("🛒 ShopAgent-AI")

    customers = list_customers()
    if not customers:
        st.error("No customers found. Run db.py first.")
        st.stop()
    customer_names = dict(customers)
    customer_id = st.selectbox("Customer", list(customer_names), format_func=customer_names.get)

    st.divider()
    st.subheader("Your cart")
    cart = orders.view_cart(thread_id)
    if cart["items"]:
        for item in cart["items"]:
            st.write(f"{item['quantity']} × {item['name'][:32]} — {item['line_total']:.0f} INR")
        st.metric("Subtotal", f"{cart['subtotal']:.2f} INR")
    else:
        st.caption("The cart is empty.")

    if st.button("Confirm order", type="primary", disabled=not cart["items"], width="stretch"):
        result = orders.create_order(thread_id, customer_id)
        if "order_id" in result:
            invoice = orders.generate_invoice(result["order_id"])
            st.session_state.last_order = {**result, "invoice_path": invoice.get("invoice_path")}
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"Your order #{result['order_id']} is confirmed. "
                               f"Total: {result['total']:.2f} INR. Your invoice is ready in the sidebar.",
                }
            )
            st.rerun()
        else:
            st.error(result.get("error", "The order failed."))
            for problem in result.get("items", []):
                st.write(f"- {problem['name']}: wanted {problem['wanted']}, in stock {problem['in_stock']}")

    last = st.session_state.last_order
    if last:
        st.success(f"Order #{last['order_id']} confirmed: {last['total']:.2f} INR")
        path = Path(last["invoice_path"]) if last.get("invoice_path") else None
        if path and path.exists():
            st.download_button(
                "Download invoice (PDF)",
                data=path.read_bytes(),
                file_name=path.name,
                mime="application/pdf",
                width="stretch",
            )

    st.divider()
    if st.button("New conversation", width="stretch"):
        new_conversation()
        st.rerun()

    with st.expander("Demo tools"):
        st.caption("Restores the original stock and deletes all carts and orders.")
        if st.button("Reset demo data"):
            ok = reset_demo_data()
            new_conversation()
            st.session_state.flash = "Demo data reset." if ok else f"No backup found at {BACKUP}"
            st.rerun()


# ------------------------------------------------------------------
# Main area: catalog (left) and chat (right)
# ------------------------------------------------------------------
st.title("ShopAgent-AI")
st.caption(
    "Browse the catalog or ask the agent in Arabic or English. The agent fills your cart; "
    "you place the order with the Confirm button in the sidebar."
)

col_catalog, col_chat = st.columns([1, 2], gap="large")


def render_catalog():
    st.subheader("Catalog")
    try:
        tree = queries.category_tree()
    except Exception:  # the category columns are missing: migrate_categories.py was not run
        st.warning("Catalog unavailable. Run `python migrate_categories.py` once, then reload.")
        return
    if not tree:
        st.caption("No categories found.")
        return

    browse = st.session_state.browse
    if browse:
        top, sub = browse
        with st.container(border=True):
            st.markdown(f"**{queries.pretty(top)} › {queries.pretty(sub)}**")
            for p in queries.products_in_group(top, sub, limit=15):
                st.write(p["name"][:70])
                st.caption(
                    f"id {p['id']} · {p['price']:.0f} INR · rating {p['rating']} · in stock {p['stock']}"
                )
                st.button(
                    "Add to cart",
                    key=f"add_{p['id']}",
                    disabled=p["stock"] < 1,
                    on_click=add_from_catalog,
                    args=(p["id"],),
                )
            st.button("Close", key="close_browse", on_click=close_browse)

    for top in sorted(tree):
        types = tree[top]
        with st.expander(f"{queries.pretty(top)} ({sum(types.values())})"):
            for sub, n in sorted(types.items(), key=lambda kv: -kv[1]):
                st.button(
                    f"{queries.pretty(sub)} ({n})",
                    key=f"grp|{top}|{sub}",
                    on_click=select_group,
                    args=(top, sub),
                    width="stretch",
                )


with col_catalog:
    render_catalog()

with col_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                st.caption(msg["meta"])

prompt = st.chat_input("What are you looking for?")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with col_chat:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = agent.chat(prompt, thread_id)
                    reply = result["reply"].strip() or "(no reply)"
                    meta = f"{result['model']} · {result['task']} · {result['seconds']}s"
                except RuntimeError as exc:
                    reply = "Sorry, I could not reach any model right now. Please try again."
                    meta = str(exc)[:300]

    st.session_state.messages.append({"role": "assistant", "content": reply, "meta": meta})
    st.rerun()  # refresh the sidebar cart
