"""Query-time retrieval: embed a query and search the persisted vector store.

Ties src/embedder.py (query embedding) and src/vector_store.py (search)
into one interface -- a Retriever holds the loaded index in memory so the
planner can call .retrieve() once per sub-question without re-reading the
~110MB index off disk each time.

Plain top-k semantic search badly under-covers broad "what are the X"
questions: an RHP's Risk Factors section alone can span 40+ pages, so top_k
neighbors, however similar, cover a small fraction of the section. When a
question's topic can be recognized (src.section_topics.infer_topic), this
scopes retrieval to that section and searches a much larger pool -- verified
against the actual failure case (a "what are the risk factors" query where
only 1 of 5 top-k hits was even in the Risk Factors section).

Widening the pool surfaced a second, related gap: a sub-question can name
its company explicitly ("...for Aastha Spintex Limited") and still get
zero chunks from that company, because nothing was using that name to
filter -- open semantic search across the whole corpus can rank a
different company's similarly-worded section above the named company's
own. Verified this happened for real on an "objects of the issue" query.
Same fix shape as section topics: infer the company from the query text
against the companies actually in the index, unless the caller already
said which one.

Matching is punctuation-normalized on both sides, not a raw substring
check. Company names round-trip through a filesystem-safe filename (see
data/scrape_sebi_rhps.py's company_filename / src/loader.py's
company_from_filename) that collapses any punctuation to underscores, then
back to spaces -- so a real name like "Hy-Tech Engineers Limited" is
indexed as "HY TECH ENGINEERS LIMITED" (hyphen -> space). A raw substring
check against a question that writes the name with its actual hyphen would
never match, silently falling through to on-demand ingestion trying (and
failing to recognize) the company that's already indexed, over and over.
Verified this happened for real via src/on_demand_ingest.py.

infer_company only catches an *exact* (post-normalization) name -- a
misspelled or abbreviated company ("Aastha Spintx", "Sanstar Ltd" for
"Sanstar Limited") won't be a substring of anything indexed, and without
fuzzy_match_company that silently falls through to on-demand ingestion,
which re-downloads and re-processes an RHP that's already indexed just to
fail to improve on it. fuzzy_match_company is a separate, deliberately
stricter check (whole-name similarity via difflib, not substring) --
callers should try the exact/substring match first and only fall back to
this, so on-demand ingestion is still reachable for a genuinely new
company.
"""
from __future__ import annotations

import difflib
import re
from pathlib import Path

from src.embedder import embed_query
from src.section_topics import infer_topic
from src.vector_store import VectorStore

DEFAULT_INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "index"
DEFAULT_TOP_K = 5
BROAD_TOP_K = 20  # used when a section topic is active (explicit or inferred)

# Whole-name similarity ratio (difflib.SequenceMatcher) required for
# fuzzy_match_company to accept a match. High enough that two distinct but
# similarly-named companies ("Sanstar Limited" vs "Sanstar Foods Limited")
# don't get confused for a typo of each other, low enough to still catch a
# missing/swapped letter or "Ltd" vs "Limited".
FUZZY_MATCH_CUTOFF = 0.84

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _normalize_for_matching(text: str) -> str:
    return _NON_ALNUM_RE.sub(" ", text.lower()).strip()


class Retriever:
    def __init__(self, store: VectorStore):
        self.store = store
        self.known_companies = self._sorted_companies({m["company"] for m in store.metadata})

    @staticmethod
    def _sorted_companies(companies) -> list[str]:
        # Longest name first, so a more specific company match wins over a
        # shorter name that happens to be a substring of another.
        return sorted(companies, key=len, reverse=True)

    @classmethod
    def load(cls, index_dir: Path = DEFAULT_INDEX_DIR) -> "Retriever":
        return cls(VectorStore.load(index_dir))

    def infer_company(self, query: str) -> str | None:
        """A known indexed company name mentioned in `query`, if any (punctuation-tolerant)."""
        normalized_query = _normalize_for_matching(query)
        for company in self.known_companies:
            if _normalize_for_matching(company) in normalized_query:
                return company
        return None

    def fuzzy_match_company(self, name: str) -> str | None:
        """The known indexed company whose name is the closest typo-level match to `name`, if any.

        Unlike infer_company, this compares `name` as a whole candidate
        company name (e.g. already extracted from a longer question by
        src.on_demand_ingest.extract_company_name) against each indexed
        company name as a whole, via difflib's ratio -- not substring
        containment. That's what lets it catch a misspelling ("Aastha
        Spintx Limited") or abbreviation ("Sanstar Ltd") that infer_company
        can't, while FUZZY_MATCH_CUTOFF keeps it from conflating two
        distinct, similarly-named companies.
        """
        normalized_name = _normalize_for_matching(name)
        normalized_to_company = {_normalize_for_matching(c): c for c in self.known_companies}
        matches = difflib.get_close_matches(
            normalized_name, normalized_to_company.keys(), n=1, cutoff=FUZZY_MATCH_CUTOFF
        )
        return normalized_to_company[matches[0]] if matches else None

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        company: str | None = None,
        section: str | None = None,
        section_topic: str | None = None,
    ) -> list[dict]:
        """Top-k chunks for `query`, each as a metadata dict plus a `score` key.

        `company` and `section_topic`, if not given explicitly, are both
        inferred from the query text. When a topic is active (explicit or
        inferred) and `top_k` wasn't given explicitly either, the search
        pool widens from DEFAULT_TOP_K to BROAD_TOP_K -- a recognized-topic
        question is exactly the case that needs section-wide coverage, not
        five similarity-ranked snippets.
        """
        if company is None:
            company = self.infer_company(query)
        if section_topic is None:
            section_topic = infer_topic(query)
        if top_k is None:
            top_k = BROAD_TOP_K if section_topic else DEFAULT_TOP_K

        query_vec = embed_query(query)
        results = self.store.search(
            query_vec, top_k=top_k, company=company, section=section, section_topic=section_topic
        )
        return [{**meta, "score": score} for meta, score in results]
