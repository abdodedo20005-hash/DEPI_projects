# ShopAgent-AI

An Arabic/English AI shopping assistant built with LangGraph, LangChain, Groq and Streamlit.  
Live demo: https://drive.google.com/file/d/1GdK7IaI6nHsueSitul_NJib8QT48wJk2/view?usp=sharing

---

## Features

| CP | Feature |
|----|---------|
| CP1 | SQLite product catalogue loaded from Kaggle Amazon CSV |
| CP2 | Tool functions: search, details, compare, stock check |
| CP3 | LangGraph ReAct agent with multi-model routing (Groq fast/smart + Ollama fallback) |
| CP4 | Cart management + PDF invoice generation |
| CP5 | Streamlit UI with catalog browser, sidebar cart, admin stock dashboard |
| CP6 | Evaluation suite, Return/Warranty RAG, README, GitHub preparation |

---

## Architecture

```
User ──► Streamlit (app.py)
              │
              ├─► Catalog browser (queries.py → SQLite)
              │
              └─► Chat (agent.py)
                       │
                       ├─► classify() → route to groq-fast or groq-smart (Groq API)
                       │               fallback: Ollama (local)
                       │
                       └─► LangGraph ReAct agent
                                │  tools
                                ├─ search_products / get_product_details
                                ├─ compare_products / check_stock
                                ├─ add_to_cart / remove_from_cart / view_cart
                                ├─ list_categories
                                └─ answer_policy (RAG – rag.py)
```

Orders are placed via the **"Confirm order"** button, not by the LLM.

---

## Repository Structure

```
.
├── agent.py              # LangGraph agent + model routing
├── app.py                # Streamlit application (CP5)
├── db.py                 # Database init + CSV loader (CP1)
├── tools.py              # Store tool functions (CP2)
├── orders.py             # Cart, order, invoice logic (CP4)
├── queries.py            # Category tree helpers
├── migrate_categories.py # One-time category column migration
├── rag.py                # CP6 – RAG for return/warranty queries
├── evaluate.py           # CP6 – Full evaluation suite
├── test_tools.py         # CP2 tool unit tests
├── test_orders.py        # CP4 order/cart unit tests
├── test_rag.py           # CP6 RAG unit tests
├── policies/
│   └── return_policy.md  # Return & warranty policy document
├── pages/
│   └── 1_Admin.py        # Streamlit admin stock dashboard
├── requirements.txt
├── .env.example
└── data/                 # (git-ignored) shop.db, amazon.csv
```

---

## Installation & Setup

### Prerequisites
- Python 3.11+
- `uv` (recommended) or `pip`

### 1. Clone
```bash
git clone https://github.com/abdodedo20005-hash/DEPI_projects.git
cd DEPI_projects
```

### 2. Create virtual environment
```bash
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
uv pip install -r requirements.txt
```

### 4. Set environment variables
```bash
cp .env.example .env
# Edit .env and fill in your keys (see below)
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key (https://console.groq.com) |
| `GROQ_MODEL` | No | Smart model name (default: `llama3-70b-8192`) |
| `GROQ_FAST_MODEL` | No | Fast model name (default: `llama3-8b-8192`) |
| `OLLAMA_MODEL` | No | Local Ollama model (default: `qwen2.5:3b`) |
| `RAG_EMBED_MODEL` | No | Sentence-transformers model for RAG (default: `all-MiniLM-L6-v2`) |

---

## Database Initialisation

Place `amazon.csv` (Kaggle Amazon product dataset) in the `data/` folder, then run:

```bash
python db.py                  # creates data/shop.db, loads products, seeds customers
python migrate_categories.py  # adds category columns (run once)
```

---

## Running the Application

```bash
python -m streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Running the Evaluation Suite

```bash
python evaluate.py
```

The suite runs 22 test cases covering:
- Product search (English and Arabic-translated keywords)
- Product comparison and details
- Stock checking
- Cart operations (add, remove, isolation per session)
- Order creation and invoice generation
- RAG: return/warranty questions and unsupported questions

Output example:
```
=======================================================
ShopAgent-AI  –  CP6 Evaluation Suite
=======================================================
PASS  search_english
PASS  search_arabic_kw
...
=======================================================
Result: 19/22 passed, 3 failed
=======================================================
```

Individual test files can also be run directly:
```bash
python test_tools.py    # CP2 tool tests
python test_orders.py   # CP4 order/cart tests
python test_rag.py      # CP6 RAG tests
```

---

## RAG – Return & Warranty Policy

`rag.py` implements a lightweight local RAG pipeline:

1. **Documents** – Markdown files in `policies/` are loaded and chunked.
2. **Embeddings** – `sentence-transformers/all-MiniLM-L6-v2` (runs locally, free).
3. **Index** – FAISS `IndexFlatIP` (cosine similarity on normalised vectors).
4. **Retrieval** – Top-3 chunks above a 0.30 similarity threshold are retrieved.
5. **Generation** – If Groq is available the retrieved context is sent to the LLM; otherwise the raw chunks are returned.
6. **Grounding** – If no chunk exceeds the threshold, the module returns *"I could not find information about that in our policy documents"* – it never invents a policy.

### Install RAG dependencies
```bash
uv pip install sentence-transformers faiss-cpu
```

### CLI usage
```bash
python rag.py "What is the return window?"
python rag.py "Is there a warranty on electronics?"
```

### Integration with the agent
The `rag.answer()` function can be wrapped as a LangChain `@tool` and added to `TOOLS` in `agent.py` to let the agent answer policy questions.

---

## Known Limitations

- The agent cannot place orders (by design). Orders are placed via the Confirm button only.
- The Groq API requires an active internet connection and a valid key.
- The Ollama fallback requires a local Ollama installation with the configured model downloaded.
- RAG embeddings are computed locally on first run (may take 10–30 s); subsequent runs use a disk cache (`.rag_cache.pkl`).
- Product data must be loaded manually from the Kaggle CSV (`python db.py`).
- The invoice PDF uses `fpdf2` and only supports Latin characters; Arabic text in product names is transliterated.
