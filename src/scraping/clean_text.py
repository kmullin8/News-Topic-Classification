"""
Centralized cleaning, filtering, and normalization utilities for RSS ingestion.
Used by rss_poll.py and rss_test.py.
"""

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from datetime import datetime
from dateutil import parser as dateparser
from zoneinfo import ZoneInfo
import warnings

# ======================================================
# Timezone configuration
# ======================================================

# Default timezone for normalized article timestamps
TZ = ZoneInfo("America/Denver")

# Map US timezone abbreviations → offsets (seconds)
TZINFOS = {
    "EST": -5 * 3600,
    "EDT": -4 * 3600,
    "CST": -6 * 3600,
    "CDT": -5 * 3600,
    "MST": -7 * 3600,
    "MDT": -6 * 3600,
    "PST": -8 * 3600,
    "PDT": -7 * 3600,
}

# ======================================================
# Unified unwanted-content patterns
# ======================================================

UNWANTED_PATTERNS_TITLE = [
    "sports",
    "lifestyle",
    "opinion",
    "analysis:",
    "elliott wave",
    "powered by embedpress",
    "palm pulse",
]

UNWANTED_PATTERNS_DESCRIPTION = [
    "powered by embedpress",
    "newsletter",
    "subscribe",
    "sign up",
    "weekly",
    "weekly roundup",
    "career opportunities",
    "check out",
]

BOILERPLATE_PATTERNS = [
    "welcome to",
    "subscribe",
    "weekly",
    "newsletter",
    "sign up",
    "career opportunities",
]


# ======================================================
# Extract text cleanly from HTML with warnings suppressed
# ======================================================
def extract_clean_text(html: str) -> str:
    """
    Convert RSS HTML descriptions into plain text.
    Suppresses warnings for malformed input.
    """
    if not isinstance(html, str):
        return ""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MarkupResemblesLocatorWarning)
        try:
            soup = BeautifulSoup(html, "html.parser")
            return soup.get_text(" ", strip=True)
        except Exception:
            return html.strip()


# ======================================================
# Date parsing logic 
# ======================================================
def parse_published_date(entry) -> datetime:
    """
    Safely parse published/updated date from an RSS entry.
    Falls back to local timezone 'now' if parsing fails.
    """
    raw = getattr(entry, "published", None) or getattr(entry, "updated", None)
    if not raw:
        return datetime.now(tz=TZ)

    try:
        dt = dateparser.parse(raw, tzinfos=TZINFOS)
        return dt.astimezone(TZ)
    except Exception:
        return datetime.now(tz=TZ)


# ======================================================
# Unified filter logic
# ======================================================
def should_filter_entry(title: str, description: str) -> bool:
    """
    Returns True if the RSS entry should be filtered out.
    """
    title_l = (title or "").lower()
    desc_l = (description or "").lower()

    # Hard title filters
    for pat in UNWANTED_PATTERNS_TITLE:
        if pat in title_l:
            return True

    # Hard description filters
    for pat in UNWANTED_PATTERNS_DESCRIPTION:
        if pat in desc_l:
            return True

    # Boilerplate test
    boiler_count = sum(1 for p in BOILERPLATE_PATTERNS if p in desc_l)
    if boiler_count >= 2 and len(desc_l) < 250:
        return True

    return False
