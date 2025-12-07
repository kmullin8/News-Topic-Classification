"""
Pipeline Manager
----------------
1. Poll RSS feeds (rss_poll)
2. Run classification on each article (inference.classify_text)
3. Print formatted results (MongoDB saving added later)
"""

from src.scraping.rss_poll import poll_all_feeds
from src.inference.inference import classify_text
from datetime import datetime


def run_pipeline(print_results: bool = True):
    print("\n========== PIPELINE START ==========\n")

    articles = poll_all_feeds()
    results = []

    for art in articles:
        title = art["title"]
        body = art["body"]
        url = art["url"]
        published = art["published"]

        pred = classify_text(title, body)

        row = {
            "title": title,
            "body": body,
            "url": url,
            "published": published,
            "main_topic": pred["main_topic"],
            "scores": pred["topic_scores"],
        }

        results.append(row)

        if print_results:
            print("--------------------------------------------")
            print(f"URL:       {url}")
            print(f"Title:     {title}")
            print(f"Published: {published}")
            print(f"Category:  {pred['main_topic']}")
            print(f"Body:      {body[:300]}...")
            print("--------------------------------------------\n")

    print("\n========== PIPELINE COMPLETE ==========\n")

    return results

if __name__ == "__main__":
    run_pipeline()
