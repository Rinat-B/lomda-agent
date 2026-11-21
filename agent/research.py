"""Web research for training lessons: open research, gaps against the owner's material,
verification. Compliance lessons skip it (only the owner's document counts there).

Order (docs/SPEC.md): open research on the subject (current guidance, data, incidents) ->
the owner's material as inspiration and to find gaps -> more ideas for the gaps ->
verification. Every search goes through OpenRouter's web plugin (Exa, 3 results) and the model
returns facts with an exact quote from a page it saw. Verification is ours: the page is
downloaded with httpx (Windows certificates via truststore), the quote must appear in it after
normalising spaces, niqqud and quote marks, and the page passes injection_guard. A page that
cannot be loaded makes the fact "unverified", never an error. Sources follow the hierarchy in
agent/assets/web_sources.json; blocked and unknown (commercial, unofficial) sites are rejected.

Verified facts become RAG chunks of kind "web" (locator: site, url, title); unverified facts go
to gate 2, marked, and enter the lesson only if the owner accepts them.
"""

import html
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import httpx

from agent.guards import SOURCE_DATA_RULE, scan_chunks, wrap_sources

SOURCES_PATH = Path(__file__).parent / "assets" / "web_sources.json"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
SEARCH_TIMEOUT_SECONDS = 180
FETCH_TIMEOUT_SECONDS = 20
MAX_PAGE_BYTES = 3_000_000
MIN_QUOTE_CHARS = 20
SOURCE_EXCERPT_CHARS = 6_000

VERIFIED, UNVERIFIED, REJECTED = "verified", "unverified", "rejected"


# ---- configuration and source policy -----------------------------------------------------------
def research_config() -> dict:
    from agent.llm import load_config

    defaults = {"engine": "exa", "max_results": 3, "max_searches": 12, "searches_per_idea": 2,
                "open_ideas": 3, "gap_ideas": 2}
    return {**defaults, **(load_config().get("research") or {})}


def _policy() -> dict:
    return json.loads(SOURCES_PATH.read_text(encoding="utf-8"))


def domain_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _matches(host: str, name: str) -> bool:
    return host == name or host.endswith("." + name)


def classify_source(url: str, kind: str = "") -> tuple[str | None, str]:
    """(tier name or None, reason). None = rejected."""
    if urlparse(url).scheme not in ("http", "https"):
        return None, "not a web page"
    host = domain_of(url)
    policy = _policy()
    if any(_matches(host, b) for b in policy["blocked"]):
        return None, f"{host}: social network, forum or user-edited site"
    for tier in policy["tiers"]:
        names = tier.get("domains", []) + tier.get("suffixes", [])
        if any(_matches(host, n) for n in names):
            if tier.get("only_for") and kind not in tier["only_for"]:
                return None, f"{host}: {tier['name']} is used only for {', '.join(tier['only_for'])}"
            return tier["name"], ""
    return None, f"{host}: not an official, academic or recognised safety/health source"


# ---- text normalisation and quote check --------------------------------------------------------
_NIQQUD = re.compile(r"[֑-ׇ]")
_QUOTES = str.maketrans({"“": '"', "”": '"', "„": '"', "״": '"', "’": "'", "‘": "'",
                         "׳": "'", "–": "-", "—": "-", "־": "-", " ": " "})
_TAGS = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>|<[^>]+>", re.DOTALL | re.IGNORECASE)


def normalize_text(text: str) -> str:
    text = html.unescape(text).translate(_QUOTES)
    text = _NIQQUD.sub("", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def page_text(raw_html: str) -> str:
    return normalize_text(_TAGS.sub(" ", raw_html))


def quote_in_page(quote: str, text: str) -> bool:
    q = normalize_text(quote).strip(" .\"'")
    return len(q) >= MIN_QUOTE_CHARS and q in text


def page_paragraphs(raw_html: str) -> list[str]:
    """The page as paragraphs (for injection_guard), split on block tags."""
    parts = re.split(r"</?(?:p|div|li|h[1-6]|br|tr|section|article)[^>]*>", _TAGS.sub(lambda m: m.group(0) if not m.group(1) else " ", raw_html), flags=re.IGNORECASE)
    out = []
    for part in parts:
        text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", part))).strip()
        if len(text) > 30:
            out.append(text)
    return out


# ---- fetching a page ---------------------------------------------------------------------------
class FetchError(Exception):
    """The page could not be loaded: the fact stays unverified."""


def _ssl_context():
    import ssl

    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)  # the Windows certificate store
    except Exception:  # noqa: BLE001 - fall back to certifi's bundle
        return True


def fetch_page(url: str) -> str:
    """The page's HTML. Raises FetchError for blocks, filtering, certificates and timeouts."""
    try:
        with httpx.Client(verify=_ssl_context(), timeout=FETCH_TIMEOUT_SECONDS, follow_redirects=True,
                          headers={"User-Agent": "Mozilla/5.0 (lesson-agent source check)"}) as client:
            response = client.get(url)
    except httpx.TimeoutException as e:
        raise FetchError("timeout") from e
    except httpx.ConnectError as e:
        reason = "certificate error" if "CERTIFICATE" in str(e).upper() else "connection failed"
        raise FetchError(reason) from e
    except httpx.HTTPError as e:
        raise FetchError(type(e).__name__) from e
    if response.status_code >= 400:
        raise FetchError(f"HTTP {response.status_code}")
    content_type = response.headers.get("content-type", "")
    if "html" not in content_type and "text" not in content_type:
        raise FetchError(f"not an HTML page ({content_type or 'unknown type'})")
    return response.text[:MAX_PAGE_BYTES]


# ---- searching ---------------------------------------------------------------------------------
@dataclass
class SearchResult:
    text: str  # the model's reply
    citations: list[dict]  # [{"url", "title", "content"}]
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0


def openrouter_search(messages: list[dict], model: str, config: dict | None = None) -> SearchResult:
    """One chat request with OpenRouter's web plugin = one search."""
    config = config or research_config()
    body = {"model": model, "messages": messages,
            "plugins": [{"id": "web", "engine": config["engine"], "max_results": config["max_results"]}]}
    response = httpx.post(CHAT_URL, json=body, timeout=SEARCH_TIMEOUT_SECONDS,
                          headers={"Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY', '')}"})
    response.raise_for_status()
    data = response.json()
    message = (data.get("choices") or [{}])[0].get("message") or {}
    citations = [{"url": a["url_citation"].get("url", ""), "title": a["url_citation"].get("title", ""),
                  "content": a["url_citation"].get("content", "")}
                 for a in message.get("annotations") or [] if a.get("type") == "url_citation"]
    usage = data.get("usage") or {}
    return SearchResult(str(message.get("content") or ""), citations, data.get("model", model),
                        usage.get("prompt_tokens", 0) or 0, usage.get("completion_tokens", 0) or 0,
                        float(usage.get("cost", 0) or 0))


IDEAS_PROMPT = (
    "You plan web research for a workplace training lesson in Israel. From the lesson subject "
    "below, propose {count} research ideas that would make the lesson current and convincing, one "
    "of each kind if possible: current official guidance or regulations (guidance), recent data or "
    "statistics (data), and real incidents or stories (incident). Think about the subject itself, "
    "not only what the material already says. For each idea give a search query in Hebrew aimed at "
    "Israeli official sources. Reply with only a JSON object: "
    '{{"ideas": [{{"idea": "<short English description>", "kind": "guidance" or "data" or "incident", '
    '"query": "<Hebrew search query>"}}]}}.'
) + SOURCE_DATA_RULE

GAP_PROMPT = (
    "You compare web findings with a content owner's training material. For every finding, say "
    "whether the material already says the same (same), adds something the material lacks (new), "
    "or contradicts the material (conflict). Then propose up to {count} more research ideas for "
    "important gaps: subjects of the material that an employee would benefit from seeing backed by "
    "current official guidance or data, and that the findings do not cover yet. Queries in Hebrew, "
    "aimed at Israeli official sources. Reply with only a JSON object: "
    '{{"findings": [{{"id": "W1", "relation": "same" or "new" or "conflict"}}], '
    '"ideas": [{{"idea": "...", "kind": "guidance" or "data" or "incident", "query": "..."}}]}}.'
) + SOURCE_DATA_RULE

SEARCH_PROMPT = (
    "You research a workplace training topic for an Israeli audience. Use the web search results "
    "you were given. Prefer Israeli government, academic and official safety or health sources in "
    "Hebrew; use news sites only for real incidents; never use social networks, forums or shops. "
    "Report up to 3 facts that answer the research idea. For each fact copy a quote: one or two "
    "full sentences exactly as they appear on the page, character for character, in the page's "
    "language (it will be checked against the page). Reply with only a JSON object: "
    '{"facts": [{"fact": "<the fact, one Hebrew sentence>", "quote": "<exact sentence(s) from the '
    'page>", "url": "<the page URL>", "title": "<page title>", "kind": "guidance" or "data" or '
    '"incident"}]}. If no result supports a fact, reply {"facts": []}.'
)


def _json(text: str, key: str) -> dict:
    from agent.checkers import _json_object

    return _json_object(text, key)


@dataclass
class ResearchRun:
    """Everything one research run did, for the state, the log and report.md."""
    facts: list[dict] = field(default_factory=list)
    searches: int = 0
    ideas: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


SmallCall = Callable[[list], object]  # (langchain messages) -> LLMResult
SearchCall = Callable[[list[dict]], SearchResult]  # (chat messages) -> SearchResult


def _ideas(subject: str, small: SmallCall, count: int, prompt: str, extra: str = "") -> tuple[list[dict], dict]:
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [SystemMessage(prompt.format(count=count)), HumanMessage(f"{subject}{extra}")]
    reply = small(messages)
    key = "ideas" if prompt is IDEAS_PROMPT else "findings"
    data = _json(reply.text, key)
    ideas = [i for i in data.get("ideas") or [] if isinstance(i, dict) and str(i.get("query") or "").strip()]
    return ideas[:count], data


def _search_idea(idea: dict, search: SearchCall, budget: Callable[[], bool], tries: int, run: ResearchRun) -> list[dict]:
    """Up to `tries` searches for one idea; stops at the first that yields facts."""
    found: list[dict] = []
    query = str(idea["query"]).strip()
    for attempt in range(tries):
        if not budget():
            run.notes.append(f"search cap reached before: {idea.get('idea', query)}")
            break
        wording = query if attempt == 0 else f"{query} (other wording, official Israeli sources)"
        messages = [{"role": "system", "content": SEARCH_PROMPT},
                    {"role": "user", "content": f"Research idea: {idea.get('idea', '')}\nSearch: {wording}"}]
        run.searches += 1
        try:
            result = search(messages)
        except Exception as e:  # noqa: BLE001 - a failed search is noted, research goes on
            run.notes.append(f"search failed ({type(e).__name__}): {wording}")
            continue
        try:
            facts = [f for f in _json(result.text, "facts").get("facts") or [] if isinstance(f, dict)]
        except ValueError:
            facts = []
        cited = {c["url"]: c for c in result.citations}
        for f in facts:
            url = str(f.get("url") or "").strip()
            if url and str(f.get("fact") or "").strip():
                found.append({"fact": str(f["fact"]).strip(), "quote": str(f.get("quote") or "").strip(),
                              "url": url, "title": str(f.get("title") or cited.get(url, {}).get("title") or ""),
                              "kind": str(f.get("kind") or idea.get("kind") or ""),
                              "idea": idea.get("idea", ""), "cited": url in cited})
        if found:
            break
    return found


def verify_fact(fact: dict, fetch: Callable[[str], str], small: SmallCall | None) -> dict:
    """Status verified / unverified / rejected, with the reason."""
    tier, reason = classify_source(fact["url"], fact.get("kind", ""))
    fact = {**fact, "site": domain_of(fact["url"]), "tier": tier}
    if tier is None:
        return {**fact, "status": REJECTED, "reason": reason}
    if len(normalize_text(fact.get("quote", ""))) < MIN_QUOTE_CHARS:
        return {**fact, "status": UNVERIFIED, "reason": "no quote to check"}
    try:
        raw = fetch(fact["url"])
    except FetchError as e:
        return {**fact, "status": UNVERIFIED, "reason": f"page not loaded: {e}"}
    except Exception as e:  # noqa: BLE001 - any failure to load is "not verified", not an error
        return {**fact, "status": UNVERIFIED, "reason": f"page not loaded: {type(e).__name__}"}
    if not quote_in_page(fact["quote"], page_text(raw)):
        return {**fact, "status": UNVERIFIED, "reason": "the quote is not on the page"}
    if small is not None:
        chunks = [{"text": p, "kind": "web", "locator": {"file": fact["site"], "paragraph": i}}
                  for i, p in enumerate(page_paragraphs(raw), 1)]
        flagged = scan_chunks(chunks, small)
        quote = normalize_text(fact["quote"])[:60]
        if any(quote[:40] in normalize_text(c["text"]) for c in flagged):
            return {**fact, "status": REJECTED, "reason": "the quoted passage looks like instructions to an AI model"}
        if flagged:
            fact["page_flagged"] = len(flagged)
    return {**fact, "status": VERIFIED, "reason": ""}


def run_research(subject: str, material: str, small: SmallCall, search: SearchCall,
                 fetch: Callable[[str], str] | None = None, config: dict | None = None) -> ResearchRun:
    """The whole research step. `subject` = the lesson's subject (titles, request); `material` =
    a sample of the owner's material."""
    config = config or research_config()
    fetch = fetch or (lambda url: fetch_page(url))  # looked up at call time (tests replace it)
    run = ResearchRun()

    def budget() -> bool:
        return run.searches < config["max_searches"]

    facts: list[dict] = []
    try:
        open_ideas, _ = _ideas(subject, small, config["open_ideas"], IDEAS_PROMPT)
    except Exception as e:  # noqa: BLE001
        open_ideas = []
        run.notes.append(f"no research ideas ({type(e).__name__})")
    run.ideas += [{**i, "round": "open"} for i in open_ideas]
    for idea in open_ideas:
        facts += _search_idea(idea, search, budget, config["searches_per_idea"], run)

    # The owner's material: what do the findings add, and where are the gaps?
    for i, f in enumerate(facts, 1):
        f["id"] = f"W{i}"
    listing = "\n".join(f"{f['id']}: {f['fact']}" for f in facts) or "(no findings yet)"
    try:
        gap_ideas, data = _ideas(subject, small, config["gap_ideas"], GAP_PROMPT,
                                 f"\n\nOwner's material:\n{wrap_sources(material[:SOURCE_EXCERPT_CHARS])}"
                                 f"\n\nWeb findings:\n{listing}")
        relation = {str(r.get("id")): str(r.get("relation", "")) for r in data.get("findings") or []
                    if isinstance(r, dict)}
    except Exception as e:  # noqa: BLE001
        gap_ideas, relation = [], {}
        run.notes.append(f"gap comparison failed ({type(e).__name__})")
    for f in facts:
        f["relation"] = relation.get(f["id"], "new")
    run.ideas += [{**i, "round": "gaps"} for i in gap_ideas]
    for idea in gap_ideas:
        more = _search_idea(idea, search, budget, config["searches_per_idea"], run)
        for f in more:
            f["relation"] = "new"
        facts += more

    seen: set[tuple[str, str]] = set()
    for i, f in enumerate(facts, 1):
        key = (f["url"], normalize_text(f.get("quote", ""))[:80])
        if key in seen:
            continue
        seen.add(key)
        if f.get("relation") == "same":
            run.notes.append(f"dropped (the material already says it): {f['fact'][:80]}")
            continue
        run.facts.append({**verify_fact(f, fetch, small), "id": f"W{len(run.facts) + 1}"})
    return run


# ---- facts as RAG chunks -----------------------------------------------------------------------
def fact_chunk(fact: dict) -> dict:
    """A usable fact as a source chunk of kind web (or web_unverified for an accepted one)."""
    kind = "web" if fact["status"] == VERIFIED else "web_unverified"
    return {"text": f"{fact['fact']}\n\"{fact.get('quote', '')}\"", "kind": kind,
            "locator": {"file": fact["site"], "url": fact["url"], "title": fact.get("title", ""),
                        "paragraph": int(fact["id"][1:])}}


def usable_facts(facts: list[dict] | None, accepted: list[str] | None = None) -> list[dict]:
    """Verified facts, plus unverified ones the owner accepted at gate 2."""
    accepted = set(accepted or [])
    return [f for f in facts or [] if f["status"] == VERIFIED or (f["status"] == UNVERIFIED and f["id"] in accepted)]
