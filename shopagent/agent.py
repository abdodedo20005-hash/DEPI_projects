"""
agent.py: ShopAgent-AI agent module (Checkpoint 5)

Ported from the CP4 notebook so the Streamlit app can import it.

Big change from CP4: the agent has NO ordering tools.
It can browse, compare and manage the cart, but it cannot place an order.
Orders are placed only by the "Confirm order" button in the app, which calls
orders.create_order directly (no LLM involved).

Model routing:
- simple tasks  -> groq-fast  -> groq-smart -> ollama
- complex tasks -> groq-smart -> groq-fast  -> ollama   (ollama = last-resort fallback)
"""

import os
import re
import time
from typing import Annotated

from dotenv import find_dotenv, load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import InjectedState, create_react_agent

from shopagent import orders, queries, rag
from shopagent import tools as store

load_dotenv(find_dotenv(usecwd=True))

GROQ_FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")   # simple tasks
GROQ_SMART_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")      # complex tasks
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")                 # last-resort fallback


# ------------------------------------------------------------------
# Tools
# ------------------------------------------------------------------
def shorten(text, limit=80):
    """Cut long text so tool results stay small for the LLM."""
    text = str(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def short_names(rows):
    return [{**row, "name": shorten(row["name"])} for row in rows]


def short_cart(result):
    """Shorten product names inside a cart (or inside a result that contains one)."""
    cart = result.get("cart", result)
    if "items" in cart:
        cart["items"] = short_names(cart["items"])
    return result


def _thread(config):
    return config["configurable"]["thread_id"]


@tool
def search_products(
    query: str,
    max_price: float | None = None,
    min_rating: float | None = None,
    category: str | None = None,
) -> dict:
    """Search store products by keyword. Product names are in English, so translate
    the customer's request to ONE or TWO short English keywords first (for example
    'cable', 'charger', 'headphones'). Only use max_price, min_rating or category
    if the customer explicitly asked for them. Prices are in INR, rating is 0-5.
    Returns up to 5 products with id, name, price, stock, rating."""
    if not query.strip():
        return {"count": 0, "products": [], "message": "Empty search. Provide a keyword."}
    products = short_names(store.search_products(query, max_price, min_rating, category))
    if not products:
        # Never return an empty list: some LLM APIs reject empty tool results
        return {
            "count": 0,
            "products": [],
            "message": "No products found. Try a shorter English keyword or remove the filters.",
        }
    return {"count": len(products), "products": products}


@tool
def get_product_details(product_id: int) -> dict:
    """Get full details (brand, price, stock, rating, description) of one product by its id."""
    details = store.get_product_details(product_id)
    if "error" not in details:
        details["name"] = shorten(details["name"])
        details["description"] = shorten(details.get("description") or "", 300)
    return details


@tool
def compare_products(product_ids: list[int]) -> dict:
    """Compare 2 to 4 products by id on price, rating and stock.
    Also tells which one is the cheapest and which one is the best rated."""
    result = store.compare_products(product_ids)
    if "products" in result:
        result["products"] = short_names(result["products"])
    return result


@tool
def check_stock(product_id: int, quantity: int = 1) -> dict:
    """Check if the requested quantity of a product (by id) is available in stock."""
    result = store.check_stock(product_id, quantity)
    if "name" in result:
        result["name"] = shorten(result["name"])
    return result


@tool
def add_to_cart(product_id: int, quantity: int = 1, config: RunnableConfig = None) -> dict:
    """Add a product (by id) to the customer's cart. Use it only after the customer
    chose the product. If they did not give a quantity, use 1. Refuses quantities
    above the available stock."""
    return short_cart(orders.add_to_cart(_thread(config), product_id, quantity))


@tool
def remove_from_cart(product_id: int, quantity: int | None = None, config: RunnableConfig = None) -> dict:
    """Remove a product (by id) from the cart. Give a quantity to remove only some
    units, or leave it empty to remove the product completely."""
    return short_cart(orders.remove_from_cart(_thread(config), product_id, quantity))


@tool
def view_cart(config: RunnableConfig = None) -> dict:
    """Show the customer's cart: items, quantities, line totals and the subtotal."""
    return short_cart(orders.view_cart(_thread(config)))


@tool
def list_categories(category: str | None = None) -> dict:
    """List what the store sells. Without arguments it returns the main categories
    with product counts. Give a main category name (for example 'Electronics') to see
    the types inside it. Use it when the customer asks what we sell."""
    tree = queries.category_tree()
    if not tree:
        return {"message": "The catalog is not available."}
    if category:
        key = re.sub(r"[^a-z]", "", category.lower())
        for top, subs in tree.items():
            if re.sub(r"[^a-z]", "", top.lower()) == key:
                return {"category": queries.pretty(top),
                        "types": {queries.pretty(s): n for s, n in sorted(subs.items())}}
        return {"error": f"No such category: {category}",
                "available": [queries.pretty(t) for t in sorted(tree)]}
    return {"categories": {queries.pretty(t): sum(s.values()) for t, s in sorted(tree.items())}}


POLICY_UNAVAILABLE = "Policy retrieval is unavailable. I cannot verify the policy right now."


def policy_query(messages, tool_question=""):
    """The text the policy is searched with: the customer's latest message, plus the one
    before it when the latest is a short follow-up such as "and how long does it last?"
    (searching a follow-up alone would lose its topic)."""
    humans = [
        m.content.strip()
        for m in messages
        if m.type == "human" and isinstance(m.content, str) and m.content.strip()
    ]
    if not humans:
        return tool_question.strip()
    if len(humans) > 1 and len(humans[-1].split()) < 6:
        return humans[-2] + " " + humans[-1]
    return humans[-1]


def policy_translation(query, translation):
    """Allow an English retrieval retry for the same explicit Arabic policy topic."""
    translation = translation.strip()
    topic = rag.topic_heading(query)
    if (re.search(r"[\u0600-\u06ff]", query)
            and not re.search(r"[\u0600-\u06ff]", translation)
            and len(re.findall(r"[A-Za-z]+", translation)) >= 2
            and topic in {"warranty", "general return policy", "defective or damaged items",
                          "non-returnable items"}
            and rag.topic_heading(translation) == topic):
        return translation
    return ""


@tool
def answer_policy(
    messages: Annotated[list, InjectedState("messages")],
    question: str = "",
) -> str:
    """Retrieve policy evidence using the original customer question first.
    For an Arabic question, pass its faithful English translation in question;
    preserve its meaning and constraints without adding facts or an answer.
    The translation is used only after an original no-match and a topic check.
    Use this for policy questions, not product searches."""
    try:
        query = policy_query(messages, question)
        if not query:
            return POLICY_UNAVAILABLE
        evidence = rag.answer(query)
        translation = policy_translation(query, question)
        if evidence == rag.NOT_FOUND and translation:
            return rag.answer(translation)
        return evidence
    except Exception as exc:
        print(f"[warn] policy retrieval failed: {type(exc).__name__}")
        return POLICY_UNAVAILABLE


TOOLS = [
    search_products, get_product_details, compare_products, check_stock,
    add_to_cart, remove_from_cart, view_cart, list_categories, answer_policy,
]


# ------------------------------------------------------------------
# Models and agents
# ------------------------------------------------------------------
SYSTEM_PROMPT = """You are ShopAgent, a helpful assistant for an online store.

Browsing rules:
- Always use the tools to look up products. Never invent products, prices or stock.
- Product names in the database are in English. Translate the customer's request
  to one or two short English keywords before searching, but reply in the
  customer's language (Arabic or English).
- When you list products, show the id, a short name, the price and the rating.
- When the customer says "the first", "the second" or "the last one", refer to the
  products you listed in your previous message and use their ids.
- Prices are in Indian rupees (INR) unless told otherwise.

Policy rules:
- For return, refund, damaged-item, warranty or related return-shipping questions, call
  answer_policy. For Arabic questions, pass a faithful English translation in
  question, including relevant context only for a genuine follow-up. Do not add
  facts, time periods, policy conditions or an inferred answer to the translation.
  The tool always searches the original customer message first and may retry the
  translation when the original has no match. For English questions, omit question.
- Answer only from the retrieved policy text. If it contains the answer, do not
  claim the information is missing. End the answer with the source title shown in the
  evidence, for example "Source: Return & Warranty Policy". Never mention file names.
- Match the latest customer's language. For Arabic questions, write the final
  answer in Arabic script (except for the source title and numbers).
- If the policy tool finds no answer or fails, say you cannot verify the policy.
  Never invent a return period, warranty term, exception or shipping time.

Cart rules:
- Use add_to_cart only after the customer chose a product. If no quantity is given, use 1.
- You CANNOT place orders. When the customer is ready to buy, call view_cart, show
  the items and the total, and tell them to press the "Confirm order" button in the
  sidebar to finish. Never say an order was placed or that you placed it.
- If a tool returns an error, explain it honestly and suggest a fix (for example
  a smaller quantity).
- If the question is not about the store, politely say you can only help with shopping.
"""


def _catalog_summary() -> str:
    tree = queries.category_tree()
    if not tree:
        return "(catalog unavailable)"
    return "; ".join(
        f"{queries.pretty(top)} ({sum(subs.values())} products)" for top, subs in sorted(tree.items())
    )


SYSTEM_PROMPT += (
    "\nCatalog rules:\n"
    "- The store sells ONLY these main categories: " + _catalog_summary() + ".\n"
    "- Never suggest or promise product types outside this catalog. If the customer asks\n"
    "  for something we do not sell, say so clearly and mention what we do sell.\n"
    "- When unsure whether we sell something, search first (the search also matches category names).\n"
    "- Use list_categories when the customer asks what we sell.\n"
)


def build_llms():
    """Create every model that can be created. A missing key just skips that model."""
    llms = {}

    try:
        from langchain_groq import ChatGroq

        llms["groq-fast"] = ChatGroq(model=GROQ_FAST_MODEL, temperature=0)
        llms["groq-smart"] = ChatGroq(model=GROQ_SMART_MODEL, temperature=0)
    except Exception as exc:
        print(f"[warn] groq not available: {exc}")

    try:
        from langchain_ollama import ChatOllama

        llms["ollama"] = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    except Exception as exc:
        print(f"[warn] ollama not available: {exc}")

    return list(llms.items())


orders.init_orders_schema()

MEMORY = MemorySaver()  # one memory shared by all agents (lives as long as the process)

AGENTS = [
    (name, create_react_agent(llm, TOOLS, checkpointer=MEMORY, prompt=SYSTEM_PROMPT))
    for name, llm in build_llms()
]


# ------------------------------------------------------------------
# Router
# ------------------------------------------------------------------
def norm(text):
    """Lowercase, drop Arabic diacritics and unify letter variants for forgiving matching."""
    text = re.sub(r"[\u064B-\u0652]", "", text.lower())
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ى", "ي"), ("ة", "ه")):
        text = text.replace(a, b)
    return text


def has_any(text, words):
    tokens = set(re.findall(r"\w+", text))
    return any((w in text) if (" " in w or len(w) >= 4) else (w in tokens) for w in words)


# Words that signal a task needing more reasoning
COMPLEX_WORDS = [norm(w) for w in (
    "قارن", "مقارنه", "الفرق", "الافضل", "افضل", "احسن", "رشح", "رشحلي", "انصحني",
    "compare", "difference", "versus", "vs", "best", "recommend", "better", "which one",
)]
# Words that point back to earlier results (the model must use the conversation memory)
REFERENCE_WORDS = [norm(w) for w in (
    "الاول", "التاني", "الثاني", "التالت", "الثالث", "الاخير", "ده", "دي",
    "first", "second", "third", "last one", "that one", "previous",
)]
# Words about changing the cart or buying: these need the stronger model
STATE_WORDS = [norm(w) for w in (
    "سله", "السله", "اشتر", "اطلب", "احذف", "شيل",
    "cart", "buy", "order", "checkout", "remove", "add",
)]


def classify(message):
    """Return 'complex' or 'simple' using cheap keyword rules (no LLM call)."""
    text = norm(message)
    if has_any(text, COMPLEX_WORDS) or has_any(text, REFERENCE_WORDS) or has_any(text, STATE_WORDS):
        return "complex"
    if len(text) > 120 or len(re.findall(r"\d+", text)) >= 2:
        return "complex"
    return "simple"


ROUTES = {
    "complex": ["groq-smart", "groq-fast", "ollama"],
    "simple": ["groq-fast", "groq-smart", "ollama"],
}


def route(message):
    """Return (task_kind, ordered list of available models). Ollama is always last."""
    kind = classify(message)
    available = {name for name, _ in AGENTS}
    return kind, [name for name in ROUTES[kind] if name in available]


# ------------------------------------------------------------------
# Chat with fallback
# ------------------------------------------------------------------
def as_text(content):
    """Some models return a list of content blocks instead of a plain string."""
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)


def policy_language_fallback(question, reply, evidence):
    """Show a policy excerpt when an English question gets a mostly Arabic reply."""
    arabic_count = len(re.findall(r"[\u0600-\u06ff]", reply))
    english_count = len(re.findall(r"[A-Za-z]", reply))
    if re.search(r"[\u0600-\u06ff]", question) or arabic_count <= english_count:
        return reply
    heading = next((line.strip() for line in evidence.splitlines() if line.startswith("## ")), "")
    bullets = [line.strip() for line in evidence.splitlines() if line.strip().startswith("- ")]
    if not bullets:
        return reply
    return "Relevant policy excerpt:\n" + "\n".join([heading, *bullets] if heading else bullets)


def chat(message, thread_id="demo", skip=()):
    """Send a message to the agent. Returns a dict with the reply and some debug info.
    Raises RuntimeError if every model fails."""
    config = {"configurable": {"thread_id": thread_id}}
    kind, order = route(message)
    agents = dict(AGENTS)
    errors = []

    for name in order:
        if name in skip:
            continue
        start = time.time()
        try:
            result = agents[name].invoke({"messages": [("user", message)]}, config)
        except Exception as exc:  # connection, rate limit, wrong key, ...
            errors.append(f"{name}: {type(exc).__name__}: {str(exc)[:150]}")
            print(f"[warn] {name} failed -> trying the next model")
            continue

        messages = result["messages"]
        last_human = max(i for i, m in enumerate(messages) if m.type == "human")
        tools_used = [
            call["name"]
            for m in messages[last_human + 1 :]
            for call in (getattr(m, "tool_calls", None) or [])
        ]
        model_reply = as_text(messages[-1].content)
        reply = model_reply
        policy_sources = []
        policy_titles = []
        used_policy_excerpt = False
        source_label_added = False
        if "answer_policy" in tools_used:
            evidence = "\n".join(
                as_text(m.content)
                for m in messages[last_human + 1 :]
                if m.type == "tool" and getattr(m, "name", None) == "answer_policy"
            )
            policy_sources = sorted(set(re.findall(r"\[([\w.-]+\.md)\]", evidence)))
            policy_titles = list(dict.fromkeys(re.findall(r'Source title: "([^"]+)"', evidence)))
            if policy_sources:
                reply = policy_language_fallback(message, reply, evidence)
                used_policy_excerpt = reply != model_reply
                labels = policy_titles or policy_sources
                if not any(x.lower() in reply.lower() for x in labels + policy_sources):
                    label = "المصدر" if re.search(r"[\u0600-\u06ff]", message) else "Source"
                    reply += "\n\n" + label + ": " + ", ".join(labels)
                    source_label_added = True
        return {
            "reply": reply,
            "model_reply": model_reply,
            "model": name,
            "task": kind,
            "seconds": round(time.time() - start, 1),
            "tools": tools_used,
            "policy_sources": policy_sources,
            "policy_titles": policy_titles,
            "used_policy_excerpt": used_policy_excerpt,
            "source_label_added": source_label_added,
        }

    detail = "\n".join(errors) if errors else "no models are available (check your .env and Ollama)"
    raise RuntimeError("All models failed:\n" + detail)
