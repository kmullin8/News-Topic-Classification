import yaml
import feedparser
from pathlib import Path
from bs4 import BeautifulSoup

CONFIG_PATH = Path("config.yaml")

# Titles RSS.app uses for fake or blocked feeds
PLACEHOLDER_TITLES = [
    "not found",
    "sign up to rss.app",
    "rss.app"
]


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def extract_clean_text(html: str) -> str:
    """Extract readable text from Reuters RSS HTML description."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(" ", strip=True)
    except Exception:
        return html  # fallback if something unexpected happens


def is_placeholder_entry(entry):
    """Detect RSS.app placeholder/fake entries."""
    title = getattr(entry, "title", "").lower()
    return any(term in title for term in PLACEHOLDER_TITLES)


def is_valid_reuters_article(link: str) -> bool:
    """
    A REAL Reuters article link:
    - starts with https://www.reuters.com/
    - ends with /
    - contains a date in the form -YYYY-MM-DD
    """
    if not link.startswith("https://www.reuters.com/"):
        return False

    if not link.endswith("/"):
        return False

    # Must contain a date pattern
    parts = link.split("-")
    if len(parts) < 4:
        return False  # not enough sections for a date

    try:
        int(parts[-3])  # year
        int(parts[-2])  # month
        int(parts[-1].replace("/", ""))  # day (strip trailing slash)
    except Exception:
        return False

    return True


def has_useful_description(entry) -> bool:
    """Ensure the description contains actual text, not only an image."""
    desc = getattr(entry, "description", "")
    return len(desc.strip()) > 50  # must include some real text


def has_valid_pubdate(entry) -> bool:
    return hasattr(entry, "published") and len(entry.published.strip()) > 10


def is_good_entry(entry) -> bool:
    """Combined filtering logic — only return true Reuters articles."""
    link = getattr(entry, "link", "")

    if is_placeholder_entry(entry):
        return False

    if not is_valid_reuters_article(link):
        return False

    if not has_useful_description(entry):
        return False

    if not has_valid_pubdate(entry):
        return False

    return True


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
            print("   ❌ ERROR: No entries returned! Feed is invalid.")
            print("--------------------------------------------------\n")
            continue

        # Filter valid entries only
        valid_entries = [e for e in feed.entries if is_good_entry(e)]

        if not valid_entries:
            print("   ❌ ERROR: Feed has entries, but none are valid Reuters articles.")
            print("--------------------------------------------------\n")
            continue

        print(f"   Feed working: {len(valid_entries)} valid Reuters articles found.")
        print(f"   Showing first {min(max_entries, len(valid_entries))} entries:")

        if print_output:
            for i, entry in enumerate(valid_entries[:max_entries]):
                clean_description = extract_clean_text(entry.description)

                #print(f"     Link {i}: {entry.link}")
                print(f"     Title {i}: {entry.title}")
                print(f"     Description {i}: {clean_description}\n")

        print("--------------------------------------------------\n")


if __name__ == "__main__":
    test_rss_sources()
