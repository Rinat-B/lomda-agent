"""Local retrieval over the session's source chunks: Hebrew-aware tokenizer + BM25 index.

The index is rebuilt after every /add and cleared by /new (see agent.session.Session).
retrieve(query, k) is also exposed as a LangChain tool (make_retrieve_tool).
"""

import re
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool

STOPWORDS_PATH = Path(__file__).parent / "assets" / "stopwords_he.txt"

# Hebrew points and cantillation marks (niqqud/teamim), U+0591-U+05C7, except the maqaf
# (U+05BE), which joins words and is turned into a space instead.
_MAQAF = chr(0x05BE)
_NIQQUD = re.compile(f"[{chr(0x0591)}-{chr(0x05BD)}{chr(0x05BF)}-{chr(0x05C7)}]")
# Geresh/gershayim (and ASCII quotes) inside abbreviations are dropped, not split on.
_IN_WORD_MARKS = re.compile(f"[{chr(0x05F3)}{chr(0x05F4)}'\"`]")
_TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+")
_HEBREW_LETTER = re.compile(f"[{chr(0x05D0)}-{chr(0x05EA)}]")
# Common one-letter prefixes: vav, he, bet, lamed, mem, shin, kaf.
HEBREW_PREFIXES = frozenset(chr(c) for c in (0x05D5, 0x05D4, 0x05D1, 0x05DC, 0x05DE, 0x05E9, 0x05DB))
MAX_PREFIX_LETTERS = 2  # e.g. vav+he ("and the"), shin+bet ("that in")
MIN_STEM_LETTERS = 3


@lru_cache(maxsize=1)
def stopwords() -> frozenset[str]:
    lines = STOPWORDS_PATH.read_text(encoding="utf-8").splitlines()
    return frozenset(l.strip() for l in lines if l.strip() and not l.startswith("#"))


def normalize(text: str) -> str:
    """Lowercase, remove niqqud and in-word quote marks, turn maqaf into a space."""
    text = text.replace(_MAQAF, " ")
    text = _NIQQUD.sub("", text)
    text = _IN_WORD_MARKS.sub("", text)
    return text.lower()


def prefix_variants(word: str) -> list[str]:
    """The word plus forms with up to two leading prefix letters removed.

    Both forms are indexed, so a word that only looks prefixed still matches itself.
    """
    variants = [word]
    if not _HEBREW_LETTER.match(word):
        return variants
    stem = word
    for _ in range(MAX_PREFIX_LETTERS):
        if stem[0] in HEBREW_PREFIXES and len(stem) - 1 >= MIN_STEM_LETTERS:
            stem = stem[1:]
            variants.append(stem)
        else:
            break
    return variants


def tokenize(text: str) -> list[str]:
    """Tokens for BM25: words and numbers, stopwords removed, Hebrew prefix variants added."""
    tokens: list[str] = []
    stop = stopwords()
    for word in _TOKEN.findall(normalize(text)):
        if word in stop or (len(word) < 2 and not word.isdigit()):
            continue
        tokens.extend(prefix_variants(word))
    return tokens


class ChunkIndex:
    """BM25 over all chunks of the session's source files."""

    def __init__(self) -> None:
        self.chunks: list[dict] = []
        self._bm25 = None

    def __len__(self) -> int:
        return len(self.chunks)

    def reset(self) -> None:
        self.chunks = []
        self._bm25 = None

    def rebuild(self, chunks: list[dict]) -> None:
        from rank_bm25 import BM25Okapi

        self.chunks = list(chunks)
        corpus = [tokenize(c["text"]) for c in self.chunks]
        # BM25Okapi cannot take an empty document list; keep empty docs as a placeholder token.
        self._bm25 = BM25Okapi([doc or ["\0"] for doc in corpus]) if corpus else None

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """The k most relevant chunks (text, kind, locator, score), best first; only scores > 0."""
        if not self._bm25 or k <= 0:
            return []
        terms = tokenize(query)
        if not terms:
            return []
        scores = self._bm25.get_scores(terms)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits = []
        for i in ranked[:k]:
            if scores[i] <= 0:
                break
            chunk = self.chunks[i]
            hits.append({**chunk, "score": round(float(scores[i]), 4)})
        return hits


def make_retrieve_tool(index: ChunkIndex) -> BaseTool:
    """retrieve(query, k) as a LangChain tool bound to one session's index."""

    def retrieve(query: str, k: int = 5) -> list[dict]:
        """Return the k source-file chunks most relevant to the query, with their locators
        (file, slide or page, paragraph)."""
        return index.retrieve(query, k)

    return StructuredTool.from_function(retrieve, name="retrieve")
