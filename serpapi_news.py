"""SerpApi helpers for football team news (Google News results).

Uses SerpApi's Google Search News tab via httpx:
https://serpapi.com/news-results
"""

from __future__ import annotations

import html
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

SERPAPI_SEARCH_URL = "https://serpapi.com/search"
DEFAULT_GL = "mx"
DEFAULT_HL = "es"
DEFAULT_SEARCH_TIMEOUT = 60.0
ARTICLE_FETCH_TIMEOUT = 20.0
ARTICLE_MAX_CHARS = 1_200

# Maps friendly --when values to Google ``tbs`` recency filters.
WHEN_TO_TBS = {
    "1d": "qdr:d",
    "d": "qdr:d",
    "7d": "qdr:w",
    "w": "qdr:w",
    "1w": "qdr:w",
    "30d": "qdr:m",
    "m": "qdr:m",
    "1m": "qdr:m",
    "y": "qdr:y",
    "1y": "qdr:y",
}


@dataclass(frozen=True)
class NewsArticle:
    """Normalized news item from SerpApi Google News results."""

    title: str
    link: str
    source: str
    date: str | None = None
    snippet: str | None = None
    thumbnail: str | None = None
    team: str | None = None
    body: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SerpApiError(RuntimeError):
    """Raised when SerpApi returns an error response."""


def get_serpapi_key() -> str:
    """Resolve the SerpApi key from common environment variable names."""
    for name in ("SERP_API_KEY", "SERPAPI_API_KEY", "SERPAPI_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    raise KeyError("Missing SerpApi key. Set SERP_API_KEY (or SERPAPI_API_KEY / SERPAPI_KEY) in the environment.")


# Exclude Liga MX Femenil coverage from team news searches.
_EXCLUDE_TERMS = ("femenil", "femenino")
_EXCLUDE_TEXT_RE = re.compile(r"femenil|femenino", re.IGNORECASE)


def build_team_query(team: str) -> str:
    """Build a Google News query focused on a Liga MX (men's) club."""
    team = team.strip()
    exclusions = " ".join(f"-{term}" for term in _EXCLUDE_TERMS)
    return f'"{team}" (Liga MX OR futbol OR fútbol OR soccer) {exclusions}'


def _is_femenil_article(title: str, snippet: str | None) -> bool:
    """Return True when title/snippet look like women's-team coverage."""
    haystack = title if not snippet else f"{title}\n{snippet}"
    return bool(_EXCLUDE_TEXT_RE.search(haystack))


def when_to_tbs(when: str | None) -> str | None:
    """Convert a short recency token (e.g. ``7d``) into a Google ``tbs`` value."""
    if not when:
        return None
    key = when.strip().lower()
    if key.startswith("qdr:"):
        return key
    return WHEN_TO_TBS.get(key)


def _source_name(raw: Any) -> str:
    if isinstance(raw, dict):
        name = raw.get("name")
        return str(name) if name else ""
    if raw is None:
        return ""
    return str(raw)


def normalize_news_results(
    results: dict[str, Any],
    *,
    team: str | None = None,
    limit: int | None = None,
) -> list[NewsArticle]:
    """Convert a SerpApi Google News response into ``NewsArticle`` objects."""
    articles: list[NewsArticle] = []
    for item in results.get("news_results") or []:
        if limit is not None and len(articles) >= limit:
            break
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        link = item.get("link")
        if not title or not link:
            continue
        title_text = str(title)
        snippet = str(item["snippet"]) if item.get("snippet") else None
        if _is_femenil_article(title_text, snippet):
            continue
        articles.append(
            NewsArticle(
                title=title_text,
                link=str(link),
                source=_source_name(item.get("source")),
                date=(str(item["date"]) if item.get("date") else None)
                or (str(item["iso_date"]) if item.get("iso_date") else None),
                snippet=snippet,
                thumbnail=str(item["thumbnail"]) if item.get("thumbnail") else None,
                team=team,
            )
        )
    return articles


def serpapi_search(
    params: dict[str, Any],
    *,
    api_key: str | None = None,
    timeout: float = DEFAULT_SEARCH_TIMEOUT,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Run a SerpApi search and return the JSON payload."""
    key = api_key or get_serpapi_key()
    query = {**params, "api_key": key, "output": "json"}
    owns_client = client is None
    http = client or httpx.Client(timeout=timeout)
    try:
        response = http.get(SERPAPI_SEARCH_URL, params=query)
        response.raise_for_status()
        payload = response.json()
    finally:
        if owns_client:
            http.close()

    if not isinstance(payload, dict):
        raise SerpApiError(f"Unexpected SerpApi response type: {type(payload).__name__}")
    if payload.get("error"):
        raise SerpApiError(str(payload["error"]))
    status = (payload.get("search_metadata") or {}).get("status")
    if status and status != "Success":
        raise SerpApiError(f"SerpApi search status: {status}")
    return payload


def search_team_news(
    team: str,
    *,
    api_key: str | None = None,
    when: str | None = "7d",
    gl: str = DEFAULT_GL,
    hl: str = DEFAULT_HL,
    limit: int | None = 20,
    timeout: float = DEFAULT_SEARCH_TIMEOUT,
    client: httpx.Client | None = None,
) -> tuple[list[NewsArticle], dict[str, Any]]:
    """Search Google News via SerpApi for recent coverage of a football team.

    Uses the Google Search News tab (``engine=google`` + ``tbm=nws``), which is
    faster and returns snippets suitable for reading.

    Returns ``(articles, raw_response)``.
    """
    params: dict[str, Any] = {
        "engine": "google",
        "tbm": "nws",
        "q": build_team_query(team),
        "gl": gl,
        "hl": hl,
    }
    if limit is not None:
        params["num"] = max(1, min(int(limit), 100))
    tbs = when_to_tbs(when)
    if tbs:
        params["tbs"] = tbs

    raw = serpapi_search(
        params,
        api_key=api_key,
        timeout=timeout,
        client=client,
    )
    return normalize_news_results(raw, team=team, limit=limit), raw


def search_teams_news(
    teams: list[str],
    *,
    api_key: str | None = None,
    when: str | None = "7d",
    gl: str = DEFAULT_GL,
    hl: str = DEFAULT_HL,
    limit_per_team: int | None = 20,
    timeout: float = DEFAULT_SEARCH_TIMEOUT,
) -> list[NewsArticle]:
    """Search news for multiple teams (one SerpApi request per team)."""
    key = api_key or get_serpapi_key()
    articles: list[NewsArticle] = []
    with httpx.Client(timeout=timeout) as client:
        for team in teams:
            team_articles, _ = search_team_news(
                team,
                api_key=key,
                when=when,
                gl=gl,
                hl=hl,
                limit=limit_per_team,
                timeout=timeout,
                client=client,
            )
            articles.extend(team_articles)
    return articles


_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript)\b[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_WHITESPACE_RE = re.compile(r"\s+")


def html_to_text(document: str) -> str:
    """Rough HTML-to-text extraction without extra parsing dependencies."""
    cleaned = _SCRIPT_STYLE_RE.sub(" ", document)
    cleaned = _TAG_RE.sub(" ", cleaned)
    cleaned = html.unescape(cleaned)
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def fetch_article_body(
    url: str,
    *,
    timeout: float = ARTICLE_FETCH_TIMEOUT,
    max_chars: int = ARTICLE_MAX_CHARS,
    client: httpx.Client | None = None,
) -> str | None:
    """Download an article URL and return extracted plain text, if possible."""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; quiniela-agentica/0.1; +https://github.com/)",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "es-MX,es;q=0.9,en;q=0.8",
    }
    try:
        if client is None:
            response = httpx.get(url, headers=headers, follow_redirects=True, timeout=timeout)
        else:
            response = client.get(url, headers=headers, follow_redirects=True, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError:
        return None

    content_type = response.headers.get("content-type", "")
    if "html" not in content_type.lower() and not response.text.lstrip().startswith("<"):
        return None

    text = html_to_text(response.text)
    if len(text) < 80:
        return None
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "…"
    return text


def enrich_articles_with_bodies(
    articles: list[NewsArticle],
    *,
    limit: int | None = None,
) -> list[NewsArticle]:
    """Fetch article bodies for the first ``limit`` items (all if ``limit`` is None)."""
    enriched: list[NewsArticle] = []
    with httpx.Client(timeout=ARTICLE_FETCH_TIMEOUT) as client:
        for index, article in enumerate(articles):
            if limit is not None and index >= limit:
                enriched.append(article)
                continue
            body = fetch_article_body(article.link, client=client)
            enriched.append(
                NewsArticle(
                    title=article.title,
                    link=article.link,
                    source=article.source,
                    date=article.date,
                    snippet=article.snippet,
                    thumbnail=article.thumbnail,
                    team=article.team,
                    body=body,
                )
            )
    return enriched
