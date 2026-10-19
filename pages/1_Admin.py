"""
pages/1_Admin.py: admin dashboard for ShopAgent-AI (Checkpoint 5)

Streamlit shows this file as the "Admin" page in the sidebar menu.
The dashboard refreshes itself every few seconds, so stock updates after
every order without reloading the page.

Optional: set ADMIN_PASSWORD in .env to ask for a password before showing the data.
(This is a simple demo gate, not real security.)
"""

import hmac
import os
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import find_dotenv, load_dotenv

from shopagent import queries

load_dotenv(find_dotenv(usecwd=True))

st.set_page_config(page_title="Admin | ShopAgent-AI", page_icon="📦", layout="wide")
st.title("📦 Admin dashboard")

password = os.getenv("ADMIN_PASSWORD")
if password and not st.session_state.get("admin_ok"):
    entered = st.text_input("Admin password", type="password")
    if entered:
        if hmac.compare_digest(entered, password):
            st.session_state.admin_ok = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()


def stock_status(stock: int, low: int) -> str:
    if stock == 0:
        return "Out of stock"
    return "Low" if stock <= low else "OK"


@st.fragment(run_every=5)
def dashboard():
    """Everything in here re-runs every 5 seconds."""
    low = st.slider("Low-stock threshold (units)", 1, 20, queries.LOW_STOCK_DEFAULT)
    overview = queries.stock_overview(low)

    cols = st.columns(6)
    cols[0].metric("Products", overview["products"])
    cols[1].metric("Units in stock", overview["units"])
    cols[2].metric("Out of stock", overview["out_of_stock"])
    cols[3].metric("Low stock", overview["low_stock"])
    cols[4].metric("Orders", overview["orders"])
    cols[5].metric("Revenue (INR)", f"{overview['revenue']:.0f}")
    st.caption(f"Last updated {datetime.now():%H:%M:%S} (auto-refreshes every 5 seconds)")

    st.subheader("Stock by category")
    by_category = queries.stock_by_category()
    if by_category:
        chart = pd.DataFrame(
            {"Units": [r["units"] for r in by_category]},
            index=[queries.pretty(r["category"]) for r in by_category],
        )
        st.bar_chart(chart)

    st.subheader("Stock per product")
    c1, c2, c3 = st.columns([2, 2, 1])
    search = c1.text_input("Search by name")
    category = c2.selectbox("Category", ["All"] + queries.category_names(), format_func=queries.pretty)
    only_low = c3.checkbox("Low / out of stock only")

    rows = queries.stock_table(search, category, only_low, low)
    table = pd.DataFrame(
        [
            {
                "ID": r["id"],
                "Product": r["name"][:70],
                "Category": queries.pretty(r["category"] or ""),
                "Price (INR)": r["price"],
                "Stock": r["stock"],
                "Status": stock_status(r["stock"], low),
            }
            for r in rows
        ]
    )
    st.caption(f"{len(rows)} product(s), lowest stock first")
    st.dataframe(table, hide_index=True, width="stretch")

    st.subheader("Recent orders")
    orders_rows = queries.recent_orders(10)
    if orders_rows:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Order": r["id"],
                        "Time (UTC)": r["created_at"],
                        "Customer": r["customer"],
                        "Total (INR)": r["total"],
                        "Items": (r["items"] or "")[:120],
                    }
                    for r in orders_rows
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    else:
        st.caption("No orders yet.")


dashboard()
