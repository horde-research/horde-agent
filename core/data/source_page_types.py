"""Deterministic source page-type detection for text data quality gates."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, urlparse

SEARCH_PATH_SEGMENTS = {
    "search",
    "find",
    "results",
    "browse",
    "tag",
    "tags",
}

KNOWN_STOCK_DOMAINS = {
    "123rf.com",
    "alamy.com",
    "depositphotos.com",
    "dreamstime.com",
    "freepik.com",
    "gettyimages.com",
    "istockphoto.com",
    "shutterstock.com",
    "stock.adobe.com",
}

KNOWN_DOCUMENT_WRAPPER_DOMAINS = {
    "academia.edu",
    "coursehero.com",
    "scribd.com",
    "slideshare.net",
    "studocu.com",
}

STOCK_TEXT_HINTS = {
    "browse",
    "free trial",
    "license",
    "premium",
    "royalty-free",
    "similar images",
    "stock photo",
    "stock photos",
    "vectors",
}

DOCUMENT_WRAPPER_TEXT_HINTS = {
    "0% found this document useful",
    "download",
    "pages",
    "presentation",
    "read free",
    "scribd",
    "slides",
    "views",
}

HARD_DROP_PAGE_FLAGS = {
    "known_stock_domain",
    "search_query_url",
    "search_suggestion_text",
    "stock_page_text",
}

DOCUMENT_WRAPPER_PAGE_FLAGS = {
    "known_document_wrapper_domain",
    "document_wrapper_text",
}


def detect_page_type_flags(url: Any, text: Any = "") -> list[str]:
    """Return deterministic low-value page-type flags for a source URL/text pair."""

    parsed = urlparse(str(url or ""))
    domain = (parsed.netloc or "").lower()
    if domain.startswith("www."):
        domain = domain[4:]
    path_segments = {segment.lower() for segment in parsed.path.split("/") if segment}
    query = parse_qs(parsed.query)
    lowered_text = str(text or "").lower()

    flags: list[str] = []
    if domain in KNOWN_STOCK_DOMAINS:
        flags.append("known_stock_domain")
    if domain in KNOWN_DOCUMENT_WRAPPER_DOMAINS:
        flags.append("known_document_wrapper_domain")
    if path_segments & SEARCH_PATH_SEGMENTS:
        flags.append("search_or_listing_path")
    if query and {"q", "query", "k", "search"} & set(query) and (
        domain in KNOWN_STOCK_DOMAINS or bool(path_segments & SEARCH_PATH_SEGMENTS)
    ):
        flags.append("search_query_url")
    if _hint_count(lowered_text, STOCK_TEXT_HINTS) >= 2:
        flags.append("stock_page_text")
    if _hint_count(lowered_text, DOCUMENT_WRAPPER_TEXT_HINTS) >= 2:
        flags.append("document_wrapper_text")
    if re.search(r"\bdid\s+you\s+mean\s*:", lowered_text):
        flags.append("search_suggestion_text")
    return flags


def hard_drop_page_type_flags(flags: list[str], *, drop_document_wrappers: bool = True) -> list[str]:
    hard = [flag for flag in flags if flag in HARD_DROP_PAGE_FLAGS]
    if drop_document_wrappers:
        hard.extend(flag for flag in flags if flag in DOCUMENT_WRAPPER_PAGE_FLAGS)
    return sorted(set(hard))


def _hint_count(text: str, hints: set[str]) -> int:
    return sum(1 for hint in hints if hint in text)
