"""
Production RSS polling module.
Loads RSS feeds from config.yaml, cleans entries,
filters unwanted items, and returns standardized article dicts.

Returned item format:
{
    "title": str,
    "body": str,
    "url": str,
    "published": datetime,
    "source": str
}
"""

import feedparser
from datetime import datetime
from dateutil import parser as dateparser
from zoneinfo import ZoneInfo
from pathlib import Path
import yaml

from src.scraping.clean_text import extract_clean_text, should_filter_entry


CONFIG_PATH = Path("config.yaml")
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


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ======================================================
# Parse published date safely
# ======================================================
def _parse_date(entry):
    raw = getattr(entry, "published", None) or getattr(entry, "updated", None)
    if not raw:
        return datetime.now(tz=TZ)

    try:
        # Use timezone mapping to eliminate warnings
        dt = dateparser.parse(raw, tzinfos=TZINFOS)
        return dt.astimezone(TZ)
    except Exception:
        return datetime.now(tz=TZ)


# ======================================================
# Poll a single RSS feed
# ======================================================
def poll_feed(feed_info: dict):
    name = feed_info.get("name", "Unnamed Feed")
    url = feed_info.get("url")

    feed = feedparser.parse(url)

    if not feed.entries:
        print(f"[RSS] ERROR: No entries for feed {name}: {url}")
        return []

    articles = []

    for entry in feed.entries:
        title = getattr(entry, "title", "").strip()
        desc_raw = getattr(entry, "description", "")
        body = extract_clean_text(desc_raw)
        link = getattr(entry, "link", "").strip()
        pub_date = _parse_date(entry)

        # Apply unified filtering
        if should_filter_entry(title, body):
            continue

        articles.append({
            "title": title,
            "body": body,
            "url": link,
            "published": pub_date,
            "source": name,
        })

    return articles


# ======================================================
# Poll ALL feeds in config.yaml
# ======================================================
def poll_all_feeds():
    cfg = load_config()
    feeds = cfg.get("rss_sources", [])

    all_articles = []
    print(f"\n[RSS] Polling {len(feeds)} feeds...\n")

    for feed in feeds:
        name = feed.get("name")
        print(f"[RSS] Polling: {name}")
        feed_articles = poll_feed(feed)
        print(f"[RSS] → {len(feed_articles)} valid items.\n")
        all_articles.extend(feed_articles)

    print(f"[RSS] TOTAL collected articles: {len(all_articles)}\n")
    return all_articles
