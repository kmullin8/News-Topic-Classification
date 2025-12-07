"""
Pipeline Manager
----------------
1. Poll RSS feeds
2. Skip articles already in MongoDB
3. Run classification on new articles
4. Save results to MongoDB
5. Print formatted output
"""

from src.scraping.rss_poll import poll_all_feeds
from src.inference.inference import classify_text
from src.db.mongo_client import article_exists, save_article
from datetime import datetime


def run_pipeline(print_results: bool = True):
    print("\n========== PIPELINE START ==========\n")

    articles = poll_all_feeds()

    saved_count = 0
    skipped_count = 0

    for art in articles:
        url = art["url"]
        title = art["title"]
        body = art["body"]
        published = art["published"]

        # ---------------------------------------
        # STEP 1 — MongoDB deduplication check
        # ---------------------------------------
        if article_exists(url):
            skipped_count += 1
            print(f"[SKIP] Already ingested: {url}")
            continue

        # ---------------------------------------
        # STEP 2 — Classification
        # ---------------------------------------
        pred = classify_text(title, body)

        article_doc = {
            "url": url,
            "title": title,
            "body": body,
            "published": published,
            "main_topic": pred["main_topic"],
            "topic_scores": pred["topic_scores"],
            "full_topics": pred["topics"],
        }

        # ---------------------------------------
        # STEP 3 — Save to MongoDB
        # ---------------------------------------
        save_article(article_doc)
        saved_count += 1

        # ---------------------------------------
        # STEP 4 — Print to console (optional)
        # ---------------------------------------
        if print_results:
            print("--------------------------------------------")
            print(f"URL:       {url}")
            print(f"Title:     {title}")
            print(f"Published: {published}")
            print(f"Category:  {pred['main_topic']}")
            print(f"Body:      {body[:300]}...")
            print("[MongoDB] Saved.")
            print("--------------------------------------------\n")

    # ---------------------------------------
    # FINAL SUMMARY
    # ---------------------------------------
    print("\n========== PIPELINE COMPLETE ==========")
    print(f"Saved new articles:     {saved_count}")
    print(f"Skipped duplicates:     {skipped_count}")
    print("========================================\n")

    return {
        "saved": saved_count,
        "skipped": skipped_count,
    }


if __name__ == "__main__":
    run_pipeline()
