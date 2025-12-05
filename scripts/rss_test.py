import yaml
import feedparser
from pathlib import Path

CONFIG_PATH = Path("config.yaml")

PLACEHOLDER_TITLES = [
    "not found",
    "sign up to rss.app",
    "rss.app"
]


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def is_placeholder_entry(entry):
    """Detect RSS.app 'fake' entries that represent invalid feeds."""
    title = getattr(entry, "title", "").lower()
    return any(marker in title for marker in PLACEHOLDER_TITLES)


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

        # Case 1 — No entries
        if not feed.entries:
            print("     ❌ ERROR: No entries returned! Feed is invalid.")
            print("--------------------------------------------------\n")
            continue

        # Case 2 — All placeholder content
        if all(is_placeholder_entry(e) for e in feed.entries):
            print("     ❌ ERROR: RSS.app placeholder feed detected — invalid or locked feed.")
            print("--------------------------------------------------\n")
            continue

        # Case 3 — Valid feed
        print(f"     Feed working: {len(feed.entries)} entries returned, Showing first {max_entries} entries:")

        if print_output:
            valid_count = 0

            for entry in feed.entries[:max_entries]:
                if is_placeholder_entry(entry):
                    continue

                link = getattr(entry, "link", "(no link)")
                print(f"     Link {valid_count} :      {link}")
                valid_count += 1

            if valid_count == 0:
                print("  ⚠ WARNING: All entries were placeholders — feed unusable.\n")

        print("--------------------------------------------------\n")


if __name__ == "__main__":
    test_rss_sources()
