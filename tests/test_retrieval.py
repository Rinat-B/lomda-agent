"""Hebrew tokenizer, BM25 index, and the RAG answer node (fake model, no network)."""

import json
from pathlib import Path

import pytest

import agent.nodes as nodes
from agent.messages import msg
from agent.session import Session, SourceDoc
from agent.tools import ChunkIndex, make_retrieve_tool, normalize, prefix_variants, tokenize
from tests.conftest import make_result

DATA = json.loads((Path(__file__).parent / "fixtures" / "retrieval_he.json").read_text(encoding="utf-8"))


# ---- tokenizer ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", DATA["tokenize"], ids=[c["name"] for c in DATA["tokenize"]])
def test_tokenize_hebrew(case):
    tokens = tokenize(case["text"])
    for t in case.get("includes", []):
        assert t in tokens, (case["name"], t, tokens)
    for t in case.get("excludes", []):
        assert t not in tokens, (case["name"], t, tokens)
    if "exact" in case:
        assert tokens == case["exact"]


def test_prefix_variants_leave_non_hebrew_alone():
    assert prefix_variants("forklift") == ["forklift"]
    assert prefix_variants("2.5") == ["2.5"]


def test_normalize_lowercases_english():
    assert normalize("WMS Forklift") == "wms forklift"


# ---- index ---------------------------------------------------------------------------------
@pytest.fixture
def index():
    idx = ChunkIndex()
    idx.rebuild(DATA["chunks"])
    return idx


@pytest.mark.parametrize("q", DATA["queries"], ids=lambda q: q["expect_file"])
def test_retrieve_finds_the_right_chunk_first(index, q):
    top = index.retrieve(q["query"], k=5)[0]
    assert top["locator"]["file"] == q["expect_file"]
    for key in ("slide", "page", "paragraph"):
        if f"expect_{key}" in q:
            assert top["locator"][key] == q[f"expect_{key}"]


def test_retrieve_respects_k_and_returns_locators_and_scores(index):
    hits = index.retrieve(DATA["queries"][0]["query"], k=2)
    assert 1 <= len(hits) <= 2
    assert all({"text", "kind", "locator", "score"} <= set(h) for h in hits)
    assert hits == sorted(hits, key=lambda h: -h["score"])


def test_unrelated_query_returns_nothing(index):
    assert index.retrieve(DATA["unrelated_query"], k=5) == []


def test_empty_and_reset_index(index):
    assert ChunkIndex().retrieve("anything") == []
    index.reset()
    assert len(index) == 0 and index.retrieve(DATA["queries"][0]["query"]) == []


def test_retrieve_tool_wraps_the_index(index):
    tool = make_retrieve_tool(index)
    assert tool.name == "retrieve"
    hits = tool.invoke({"query": DATA["queries"][1]["query"], "k": 1})
    assert len(hits) == 1 and hits[0]["locator"]["file"] == "ethics.pdf"


def test_session_updates_index_on_add_and_clears_on_new(tmp_path):
    session = Session()
    f = tmp_path / "deck.pptx"
    f.write_bytes(b"x")
    session.add_file(f, SourceDoc("slide", 1, DATA["chunks"][:2]))
    assert len(session.index) == 2
    g = tmp_path / "safety.docx"
    g.write_bytes(b"x")
    session.add_file(g, SourceDoc("paragraph", 1, DATA["chunks"][2:3]))
    assert len(session.index) == 3  # rebuilt with all files
    session.reset()
    assert len(session.index) == 0


# ---- answer node with retrieval -------------------------------------------------------------
@pytest.fixture
def fake_llm(monkeypatch):
    """Replace the model: returns a canned reply and records the messages it got."""
    calls = []

    def install(reply: str):
        def fake_call_llm(messages):
            calls.append(messages)
            return make_result(text=reply)

        monkeypatch.setattr(nodes, "call_llm", fake_call_llm)
        return calls

    return install


def run_answer(question, index):
    config = {"configurable": {"index": index}}
    return nodes.answer({"question": question}, config)


def test_answer_uses_excerpts_and_appends_sources(index, fake_llm, output):
    calls = fake_llm("ANSWER-TEXT [1]. MORE [1, 2].")
    result = run_answer(DATA["queries"][0]["query"], index)
    prompt = calls[0][1].content
    assert "[1] (deck.pptx, slide 4, paragraph 2, body)" in prompt
    assert "ONLY the numbered excerpts" in calls[0][0].content
    hits = index.retrieve(DATA["queries"][0]["query"], 5)
    cited = hits[:2]  # [1] and [2]; numbers beyond the hit list are ignored
    expected_sources = msg(
        "rag.sources", sources=", ".join(dict.fromkeys(nodes.source_label(h["locator"]) for h in cited))
    )
    assert result["answer"] == "ANSWER-TEXT. MORE.\n\n" + expected_sources
    assert result["sources"] == [h["locator"] for h in cited]
    assert f"[INFO] Retrieved {len(hits)} chunks" in output()


def test_cited_hits_order_dedup_and_range():
    hits = [{"locator": {"file": f"f{i}", "paragraph": i}} for i in range(1, 4)]
    assert nodes.cited_hits("a [3] b [1, 3] c [9] d [2]", hits) == [hits[2], hits[0], hits[1]]
    assert nodes.strip_citations("one [1]. two [2, 3].") == "one. two."


def test_answer_not_found(index, fake_llm):
    fake_llm("NOT_FOUND")
    result = run_answer(DATA["queries"][0]["query"], index)
    assert result == {"answer": msg("rag.not_found"), "sources": []}


def test_answer_without_hits_does_not_call_the_model(index, fake_llm, output):
    calls = fake_llm("should not be used")
    result = run_answer(DATA["unrelated_query"], index)
    assert calls == [] and result["answer"] == msg("rag.not_found")
    assert "[INFO] Retrieved 0 chunks" in output()


def test_answer_without_citations_falls_back_to_top_hits(index, fake_llm):
    fake_llm("ANSWER")
    query = DATA["queries"][3]["query"]
    hits = index.retrieve(query, 5)
    result = run_answer(query, index)
    assert result["sources"] == [h["locator"] for h in hits[: nodes.FALLBACK_SOURCES]]


def test_answer_without_files_is_a_plain_answer(fake_llm, output):
    calls = fake_llm("PLAIN")
    result = nodes.answer({"question": "q"}, {"configurable": {"index": ChunkIndex()}})
    assert result == {"answer": "PLAIN", "sources": []}
    assert calls[0][0].content == nodes.ANSWER_PROMPT
    assert "Retrieved" not in output()


@pytest.mark.parametrize(
    "locator, key",
    [({"file": "a.pptx", "slide": 4, "paragraph": 1}, "slide"),
     ({"file": "a.pdf", "page": 2, "paragraph": 3}, "page"),
     ({"file": "a.docx", "paragraph": 7}, "paragraph")],
)
def test_source_labels(locator, key):
    number = locator.get(key, locator["paragraph"])
    assert nodes.source_label(locator) == msg(f"rag.source.{key}", file=locator["file"], number=number)


@pytest.fixture
def context_aware_model(monkeypatch):
    """A fake gatekeeper that follows the instructions: IN_SCOPE only when an excerpt matched."""
    calls = []

    def fake_call_llm(messages):
        calls.append(messages)
        no_excerpt = "(none: no excerpt matches the question)" in messages[1].content
        return make_result(text="OUT_OF_SCOPE" if no_excerpt else "IN_SCOPE")

    monkeypatch.setattr(nodes, "call_llm", fake_call_llm)
    return calls


def test_scope_check_always_calls_the_model_with_file_context(index, context_aware_model):
    question = DATA["queries"][1]["query"]  # the gift question
    result = nodes.scope_check({"question": question}, {"configurable": {"index": index}})
    assert result == {"in_scope": True}
    assert len(context_aware_model) == 1  # never skipped
    system, user = (m.content for m in context_aware_model[0])
    assert nodes.SCOPE_CHECK_WITH_SOURCES in system
    assert "Source files in this session: deck.pptx, ethics.pdf, safety.docx" in user
    assert "(ethics.pdf, page 2, paragraph 5)" in user  # the gift excerpt
    assert user.endswith(f"Question: {question}")


def test_scope_check_blocks_unrelated_question_even_with_files(index, context_aware_model):
    result = nodes.scope_check({"question": DATA["unrelated_query"]}, {"configurable": {"index": index}})
    assert result["in_scope"] is False and result["answer"] == msg("graph.out_of_scope")
    assert "(none: no excerpt matches the question)" in context_aware_model[0][1].content


def test_scope_check_without_files_has_no_file_context(context_aware_model):
    nodes.scope_check({"question": "q"}, {"configurable": {"index": ChunkIndex()}})
    system, user = (m.content for m in context_aware_model[0])
    assert system == nodes.SCOPE_CHECK_PROMPT and user == "q"


def test_scope_check_respects_model_refusal_even_when_excerpt_matches(index, fake_llm):
    """The model decides; a matching excerpt alone no longer forces IN_SCOPE."""
    fake_llm("OUT_OF_SCOPE")
    result = nodes.scope_check({"question": DATA["queries"][1]["query"]}, {"configurable": {"index": index}})
    assert result["in_scope"] is False
