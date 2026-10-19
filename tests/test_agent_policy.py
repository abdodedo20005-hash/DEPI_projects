"""
test_agent_policy.py: tests for the agent's policy tool and the reply handling in agent.chat()
(Checkpoint 6)

Run:  python tests/test_agent_policy.py

No API key and no embedding model are needed:
- a scripted fake LLM decides when to call the policy tool and what the final reply says,
- a small stand-in replaces the embedding model (it matches words, not meaning),
- a temporary empty database is used, so your data/shop.db is never touched.

These tests check the plumbing: the question the tool searches with, the source label,
the language fallback, the failure handling. They do NOT test how well a real LLM or the
real embedding model behave (use scripts/evaluate.py and scripts/calibrate_rag.py for that).
"""

import sys
import tempfile
import uuid
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")  # create_react_agent is deprecated in LangGraph 1.x; it still works

from shopagent import db  # noqa: E402

# Use a throw-away database: importing agent reads the catalog and creates the cart table
_tmp = tempfile.mkdtemp(prefix="shopagent-test-")
db.DB_PATH = str(Path(_tmp) / "shop.db")
db.init_db()

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langgraph.prebuilt import create_react_agent  # noqa: E402

from shopagent import agent, rag  # noqa: E402
from tests.helpers import hash_embedder  # noqa: E402


class FakeModel(BaseChatModel):
    """Calls answer_policy once (unless call_tool is False), then replies with final_text
    or, with echo=True, with the evidence the tool returned."""

    final_text: str = ""
    call_tool: bool = True
    echo: bool = False

    @property
    def _llm_type(self):
        return "fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.call_tool and messages[-1].type != "tool":
            message = AIMessage(content="", tool_calls=[{"name": "answer_policy", "args": {}, "id": "call-1"}])
        elif messages[-1].type == "tool":
            message = AIMessage(content=messages[-1].content if self.echo else self.final_text)
        else:
            message = AIMessage(content=self.final_text)
        return ChatResult(generations=[ChatGeneration(message=message)])


def ask(question, thread=None, **model_kwargs):
    model = FakeModel(**model_kwargs)
    agent.AGENTS[:] = [
        ("groq-fast", create_react_agent(model, agent.TOOLS, checkpointer=agent.MEMORY, prompt=agent.SYSTEM_PROMPT))
    ]
    return agent.chat(question, thread_id=thread or str(uuid.uuid4()))


def with_standin(fn):
    def wrapper():
        rag.set_embedder(hash_embedder)
        try:
            return fn()
        finally:
            rag.set_embedder(None)
    wrapper.__name__ = fn.__name__
    return wrapper


Q_EN = "Customers may return items within 30 days of the delivery date?"
TITLE = "Return & Warranty Policy"


# ------------------------------------------------------------------
@with_standin
def test_tool_schema_hides_the_conversation_and_makes_the_question_optional():
    schema = agent.answer_policy.tool_call_schema.model_json_schema()
    assert list(schema["properties"]) == ["question"], "the LLM must not see the injected messages"
    assert "question" not in schema.get("required", [])


@with_standin
def test_policy_query_merges_short_follow_ups_only():
    ask_policy = agent.policy_query
    assert ask_policy([HumanMessage("What is the return window?")]) == "What is the return window?"
    short = [HumanMessage("Is there a warranty on electronics?"), AIMessage("Yes."), HumanMessage("how long does it last?")]
    assert ask_policy(short) == "Is there a warranty on electronics? how long does it last?"
    long_ = [HumanMessage("Is there a warranty on electronics?"), AIMessage("Yes."),
             HumanMessage("What is the return window for an unused item please?")]
    assert ask_policy(long_) == "What is the return window for an unused item please?"
    assert ask_policy([], "fallback question") == "fallback question"
    assert ask_policy([AIMessage("hello")], "fallback question") == "fallback question"


@with_standin
def test_reply_that_cites_the_title_is_left_alone():
    result = ask(Q_EN, final_text=f"Within 30 days. Source: {TITLE}")
    assert result["tools"] == ["answer_policy"] and result["policy_sources"] == ["return_policy.md"]
    assert result["policy_titles"] == [TITLE]
    assert result["source_label_added"] is False and result["reply"] == result["model_reply"]


@with_standin
def test_missing_source_gets_the_document_title_not_the_file_name():
    result = ask(Q_EN, final_text="You can return items within 30 days.")
    assert result["source_label_added"] is True
    assert result["reply"].endswith(f"Source: {TITLE}")
    assert "return_policy.md" not in result["reply"]
    assert result["model_reply"] == "You can return items within 30 days."  # the original is kept for evaluation


@with_standin
def test_arabic_question_gets_an_arabic_source_label():
    result = ask("ما هي Customers may return items within 30 days of the delivery date؟",
                 final_text="يمكنك الإرجاع خلال 30 يوما.")
    assert result["source_label_added"] is True and f"المصدر: {TITLE}" in result["reply"]


@with_standin
def test_english_question_with_a_mostly_arabic_reply_shows_the_policy_excerpt():
    result = ask(Q_EN, final_text="يمكنك إرجاع المنتجات خلال ثلاثين يوما من تاريخ التسليم والمنتج غير مستخدم تماما")
    assert result["used_policy_excerpt"] is True
    assert "Relevant policy excerpt:" in result["reply"] and "30 days" in result["reply"]
    assert "ثلاثين" in result["model_reply"]  # what the model wrote is still available


@with_standin
def test_a_follow_up_is_searched_together_with_the_previous_question():
    seen = []
    original = agent.rag.answer
    agent.rag.answer = lambda query: (seen.append(query), original(query))[1]
    try:
        thread = str(uuid.uuid4())
        ask("Is there a manufacturer warranty for electronic products?", thread, final_text="ok")
        ask("how long does it last?", thread, final_text="ok")
    finally:
        agent.rag.answer = original
    assert seen[0] == "Is there a manufacturer warranty for electronic products?"
    assert seen[1] == "Is there a manufacturer warranty for electronic products? how long does it last?"


@with_standin
def test_the_tool_searches_with_the_customers_words_not_the_models():
    result = ask("Is there a manufacturer warranty for electronic products?", echo=True)
    assert "## Warranty" in result["model_reply"]


@with_standin
def test_retrieval_failure_is_reported_not_raised():
    def broken(texts):
        raise ImportError("no embedding model")
    rag.set_embedder(broken)
    result = ask(Q_EN, final_text="I cannot verify the policy right now.")
    assert result["tools"] == ["answer_policy"] and result["policy_sources"] == []
    assert result["source_label_added"] is False and result["reply"] == result["model_reply"]


@with_standin
def test_no_policy_evidence_means_no_source_label():
    result = ask("zzzz qqqq xxxx wwww", final_text="I could not find that.")
    assert result["policy_sources"] == [] and result["source_label_added"] is False


@with_standin
def test_turns_without_the_policy_tool_are_untouched():
    result = ask("Show me cables", call_tool=False, final_text="Here are some cables.")
    assert result["tools"] == [] and result["reply"] == "Here are some cables."
    assert result["policy_sources"] == [] and result["used_policy_excerpt"] is False


@with_standin
def test_result_has_every_key_the_app_and_the_evaluation_use():
    result = ask(Q_EN, final_text="ok")
    for key in ("reply", "model_reply", "model", "task", "seconds", "tools", "policy_sources",
                "policy_titles", "used_policy_excerpt", "source_label_added"):
        assert key in result, key


# ------------------------------------------------------------------
if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:
            failed += 1
            print(f"FAIL  {name}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    raise SystemExit(1 if failed else 0)
