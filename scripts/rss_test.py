import yaml
import feedparser
from pathlib import Path
from bs4 import BeautifulSoup

CONFIG_PATH = Path("config.yaml")

# ======================================================
# Unified unwanted-content patterns
# (titles, descriptions, and boilerplate all handled here)
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


def extract_clean_text(html: str) -> str:
    """Extract readable text from RSS HTML description."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(" ", strip=True)
    except Exception:
        return html  # fallback


# ======================================================
# Unified filter: ALL filtering happens here
# ======================================================
def should_filter_entry(title: str, description: str) -> bool:
    """
    Unified filtering logic for titles + descriptions.

    Filters out entries if:
    - Title contains any unwanted pattern
    - Description contains any unwanted pattern
    - Description contains 2+ boilerplate markers AND is short (<250 chars)
    """

    title_l = (title or "").lower()
    desc_l = (description or "").lower()

    # ---------- Hard title filters ----------
    for pat in UNWANTED_PATTERNS_TITLE:
        if pat in title_l:
            return True

    # ---------- Hard description filters ----------
    for pat in UNWANTED_PATTERNS_DESCRIPTION:
        if pat in desc_l:
            return True

    # ---------- Boilerplate detection ----------
    boiler_count = sum(1 for p in BOILERPLATE_PATTERNS if p in desc_l)
    if boiler_count >= 2 and len(desc_l) < 250:
        return True

    return False


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def test_rss_sources():
    cfg = load_config()

    rss_sources = cfg.get("rss_sources", [])
    test_cfg = cfg.get("rss_test", {})

    print_output = test_cfg.get("print_output", True)
    max_entries = test_cfg.get("max_entries", 3)

    print(f"\nLoaded {len(rss_sources)} RSS feeds from config\n")

    for feed_info in rss_sources:
        name = feed_info.get("name", "Unnamed feed")
        url = feed_info.get("url")

        print(f"Testing {name}: {url}")

        feed = feedparser.parse(url)

        if not feed.entries:
            print("   ❌ ERROR: No entries returned!")
            print("--------------------------------------------------\n")
            continue

        entries = feed.entries

        print(f"   Feed working: {len(entries)} total articles found.")

        shown = 0

        if print_output:
            print(f"   Showing first {min(max_entries, len(entries))} valid entries:\n")
            for entry in entries:
                if shown >= max_entries:
                    break

                title = getattr(entry, "title", "")
                description_raw = getattr(entry, "description", "")
                description = extract_clean_text(description_raw)

                # Apply unified filtering
                if should_filter_entry(title, description):
                    continue

                print(f"     Link  {shown}: {getattr(entry, 'link', '')}")
                print(f"     Title {shown}: {title}")
                print(f"     Description {shown}: {description}\n")

                shown += 1

        print("--------------------------------------------------\n")


if __name__ == "__main__":
    test_rss_sources()
