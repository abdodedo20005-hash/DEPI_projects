# ShopAgent-AI

An Arabic/English shopping assistant built by the DEPI course team. The project has a SQLite product catalog, shopping tools, a cart, and a Streamlit chat. The chat can answer return and warranty questions using a policy document.

[Existing live demo video](https://drive.google.com/file/d/1GdK7IaI6nHsueSitul_NJib8QT48wJk2/view?usp=sharing)

## How it works

`app.py` sends customer messages to `agent.chat()` in `agent.py`. The LangGraph agent can search products, check stock, compare items, manage the cart, and call `answer_policy`. That last tool retrieves a relevant section from `policies/return_policy.md` through `rag.py` and gives the section and source name to the same agent. The customer confirms an order with the button in Streamlit; the model itself does not place orders.

The configured models are `openai/gpt-oss-20b` for simple tasks, `openai/gpt-oss-120b` for complex tasks, and local `qwen2.5:3b` as the fallback. The policy search uses local multilingual embeddings and FAISS. No Llama model is configured in this code.

## Setup

Use Python 3.11 or newer. Create an environment and activate it before installing packages. On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, replace the activation line with `source .venv/bin/activate`. Copy `.env.example` to `.env` and enter a Groq API key for the cloud models. To use the local fallback, install Ollama and download `qwen2.5:3b` instead. Keep `.env` private.

Place the original Amazon CSV at `data/amazon.csv`, then build the product database:

```bash
python db.py
python -m streamlit run app.py
```

The application opens at `http://localhost:8501`. `data/`, `.env`, and generated invoices are ignored by Git.

## CP6: policy search and evaluation

The policy in `policies/return_policy.md` is **example course data**, not a verified store policy. `rag.py` divides it by heading, embeds each section, and searches for the closest section. The `answer_policy` tool passes that excerpt to the existing chat agent, which should answer in the customer's language and name the source file. Retrieval runs locally; the selected Groq or Ollama model writes the chat reply.

With a working Groq or Ollama model, run the evaluation with a temporary two-product catalog:

```bash
python evaluate.py --demo
```

`--demo` keeps the fixture database separate from `data/shop.db` and removes it after the run. To use your populated catalog, run `python evaluate.py` instead.

This script asks the actual `agent.chat()` used by Streamlit five fixed questions: returns and warranties in English and Arabic, plus one cable search. It checks the tools used, the policy fact and its unit in both the original model reply and displayed reply, the retrieved source, and each explicit product ID with its own catalog price. It prints a passed count and percentage and exits with an error if any case fails or no model is available. The result measures only these examples; model replies may vary between runs. Earlier database, cart, and order unit tests remain in `test_tools.py` and `test_orders.py`. Existing policy-module checks remain in `test_rag.py`; they are separate from the agent evaluation.

The report distinguishes the original model reply from the displayed customer reply. If an English policy question gets a mostly Arabic reply, the application displays the retrieved English policy excerpt. If the model omits the source filename, the application appends it from the tool evidence. These application additions do not establish that the model wrote a citation or translation itself.

The policy tool retrieves one section per question. Ask separate questions for different policy topics, and check generated wording against the cited excerpt before treating it as an exact policy statement.
