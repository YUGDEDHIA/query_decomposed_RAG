"""Canonical RHP section topics, derived empirically from the real corpus.

Two problems retrieval hits with raw `section` strings:
1. Punctuation varies by merchant-banker template ("SECTION II – RISK
   FACTORS" vs "SECTION II: RISK FACTORS" vs "SECTION II - RISK FACTORS").
2. Several sections use "Issue" and "Offer" interchangeably depending on
   template ("OBJECTS OF THE ISSUE" vs "OBJECTS OF THE OFFER"), which are
   the same topic under different names -- not two different topics.

CANONICAL_SECTIONS maps a topic key to the set of normalized section
strings (see normalize_section) that mean the same thing. Built by
aggregating section names across all 37 processed RHPs and grouping
synonyms by hand, not guessed.

infer_topic() is a second, separate heuristic: given a sub-question's text
(not a document section string), guess which canonical topic it's asking
about, so retrieval can scope broad "what are the X" questions to the
right section instead of an open top-k search across the whole document.
It's deliberately conservative -- returns None rather than a wrong guess
when the question doesn't clearly name a known topic.
"""
from __future__ import annotations

import re

_SECTION_PREFIX_RE = re.compile(r"^SECTION\s+[IVXLC]+\s*[-–:]\s*")
_NON_ALNUM_RE = re.compile(r"[^A-Z0-9 ]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_section(section: str | None) -> str | None:
    """"SECTION II – RISK FACTORS" -> "RISK FACTORS" (uppercase, no punctuation)."""
    if not section:
        return None
    s = _SECTION_PREFIX_RE.sub("", section.upper())
    s = _NON_ALNUM_RE.sub("", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s or None


CANONICAL_SECTIONS: dict[str, set[str]] = {
    "RISK FACTORS": {"RISK FACTORS"},
    "OBJECTS OF THE ISSUE": {"OBJECTS OF THE ISSUE", "OBJECTS OF THE OFFER"},
    "CAPITAL STRUCTURE": {"CAPITAL STRUCTURE"},
    "INDUSTRY OVERVIEW": {"INDUSTRY OVERVIEW"},
    "OUR BUSINESS": {"OUR BUSINESS"},
    "OUR MANAGEMENT": {"OUR MANAGEMENT"},
    "OUR PROMOTERS AND PROMOTER GROUP": {
        "OUR PROMOTERS AND PROMOTER GROUP", "OUR PROMOTER AND PROMOTER GROUP",
    },
    "OUR GROUP COMPANIES": {"OUR GROUP COMPANIES", "GROUP COMPANIES", "OUR GROUP COMPANY"},
    "DIVIDEND POLICY": {"DIVIDEND POLICY"},
    "FINANCIAL INDEBTEDNESS": {"FINANCIAL INDEBTEDNESS"},
    "RESTATED FINANCIAL INFORMATION": {
        "RESTATED FINANCIAL INFORMATION", "RESTATED CONSOLIDATED FINANCIAL INFORMATION",
        "RESTATED FINANCIAL STATEMENTS",
    },
    "SUMMARY OF FINANCIAL INFORMATION": {
        "SUMMARY OF FINANCIAL INFORMATION", "SUMMARY FINANCIAL INFORMATION",
        "SUMMARY OF RESTATED FINANCIAL INFORMATION", "SUMMARY OF RESTATED CONSOLIDATED FINANCIAL INFORMATION",
    },
    "OTHER FINANCIAL INFORMATION": {"OTHER FINANCIAL INFORMATION"},
    "CAPITALISATION STATEMENT": {"CAPITALISATION STATEMENT"},
    "SUMMARY OF RELATED PARTY TRANSACTIONS": {"SUMMARY OF RELATED PARTY TRANSACTIONS"},
    "MANAGEMENTS DISCUSSION AND ANALYSIS": {
        "MANAGEMENTS DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
        "MANAGEMENTS DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATION",
    },
    "OUTSTANDING LITIGATION AND MATERIAL DEVELOPMENTS": {
        "OUTSTANDING LITIGATION AND MATERIAL DEVELOPMENTS", "OUTSTANDING LITIGATIONS AND MATERIAL DEVELOPMENTS",
    },
    "GOVERNMENT AND OTHER APPROVALS": {"GOVERNMENT AND OTHER APPROVALS"},
    "OTHER REGULATORY AND STATUTORY DISCLOSURES": {"OTHER REGULATORY AND STATUTORY DISCLOSURES"},
    "KEY REGULATIONS AND POLICIES": {"KEY REGULATIONS AND POLICIES", "KEY REGULATIONS AND POLICIES IN INDIA"},
    "HISTORY AND CERTAIN CORPORATE MATTERS": {"HISTORY AND CERTAIN CORPORATE MATTERS"},
    "THE ISSUE": {"THE ISSUE", "THE OFFER"},
    "TERMS OF THE ISSUE": {"TERMS OF THE ISSUE", "TERMS OF THE OFFER"},
    "ISSUE STRUCTURE": {"ISSUE STRUCTURE", "OFFER STRUCTURE"},
    "ISSUE PROCEDURE": {"ISSUE PROCEDURE", "OFFER PROCEDURE"},
    "BASIS FOR ISSUE PRICE": {"BASIS FOR ISSUE PRICE", "BASIS FOR OFFER PRICE", "BASIS OF ISSUE PRICE"},
    "STATEMENT OF TAX BENEFITS": {"STATEMENT OF SPECIAL TAX BENEFITS", "STATEMENT OF POSSIBLE SPECIAL TAX BENEFITS"},
    "GENERAL INFORMATION": {"GENERAL INFORMATION"},
    "DEFINITIONS AND ABBREVIATIONS": {"DEFINITIONS AND ABBREVIATIONS", "DEFINITIONS AND ABBREVATIONS"},
    "MATERIAL CONTRACTS AND DOCUMENTS FOR INSPECTION": {"MATERIAL CONTRACTS AND DOCUMENTS FOR INSPECTION"},
    "DESCRIPTION OF EQUITY SHARES": {
        "DESCRIPTION OF EQUITY SHARES AND TERMS OF THE ARTICLES OF ASSOCIATION",
        "DESCRIPTION OF EQUITY SHARES AND TERMS OF ARTICLES OF ASSOCIATION",
    },
    "FORWARD LOOKING STATEMENTS": {"FORWARDLOOKING STATEMENTS"},
}


def matches_topic(section: str | None, topic: str) -> bool:
    """Does `section` (a raw chunk section string) belong to canonical `topic`?"""
    normalized = normalize_section(section)
    return normalized is not None and normalized in CANONICAL_SECTIONS.get(topic, set())


# Conservative trigger phrases per topic -- checked as substrings of a
# lowercased sub-question. Order matters: first match wins, so more
# specific triggers should come before more general ones.
_TOPIC_TRIGGERS: dict[str, list[str]] = {
    "RISK FACTORS": ["risk factor"],
    "OBJECTS OF THE ISSUE": [
        "objects of the issue", "objects of the offer", "use of proceeds", "issue proceeds", "offer proceeds",
    ],
    "THE ISSUE": [
        "price band", "lot size", "issue size", "fresh issue", "offer for sale", "minimum bid lot",
    ],
    "ISSUE STRUCTURE": ["qib quota", "retail quota", "nii quota", "reservation", "investor category"],
    "GENERAL INFORMATION": [
        "registrar to the issue", "lead manager", "book running lead manager", "brlm",
    ],
    "CAPITAL STRUCTURE": ["capital structure"],
    "INDUSTRY OVERVIEW": ["industry overview", "industry trends"],
    "OUR BUSINESS": ["business overview", "business operations"],
    "OUR MANAGEMENT": ["board of directors", "management team", "key managerial personnel"],
    "OUR PROMOTERS AND PROMOTER GROUP": ["promoter"],
    "OUR GROUP COMPANIES": ["group compan"],
    "DIVIDEND POLICY": ["dividend"],
    "FINANCIAL INDEBTEDNESS": ["indebtedness", "outstanding borrowings", "outstanding loans"],
    "SUMMARY OF RELATED PARTY TRANSACTIONS": ["related party transaction"],
    "MANAGEMENTS DISCUSSION AND ANALYSIS": [
        "management's discussion", "managements discussion", "md&a", "results of operations",
    ],
    "OUTSTANDING LITIGATION AND MATERIAL DEVELOPMENTS": ["litigation", "legal proceeding"],
    "GOVERNMENT AND OTHER APPROVALS": ["government approval", "regulatory approval", "statutory approval"],
    "HISTORY AND CERTAIN CORPORATE MATTERS": ["corporate history", "incorporation history"],
    "BASIS FOR ISSUE PRICE": ["basis for issue price", "basis for offer price", "basis of issue price"],
    "STATEMENT OF TAX BENEFITS": ["tax benefit"],
}


def infer_topic(question: str) -> str | None:
    """Guess which canonical section topic `question` is about, or None."""
    lowered = question.lower()
    for topic, triggers in _TOPIC_TRIGGERS.items():
        if any(trigger in lowered for trigger in triggers):
            return topic
    return None
