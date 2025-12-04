#!/usr/bin/env python3
"""
CLI tool for classifying a news article using src.inference.inference.classify_text

Supports:
    --title "..." --body "..."
    --file article.txt [--title "..."]
    --json (machine-readable output)
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure src/ is importable when running from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.inference import classify_text  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify a news article using the Reuters-trained DistilBERT model."
    )

    # Exactly one of: --body OR --file
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--body", type=str, help="Article body text")
    group.add_argument("--file", type=str, help="Path to text file containing article body")

    parser.add_argument("--title", type=str, default="", help="Optional article title/headline")
    parser.add_argument("--max_length", type=int, default=None, help="Override model max_length")
    parser.add_argument("--json", action="store_true", help="Output full result as JSON")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load body text
    if args.file:
        fp = Path(args.file)
        if not fp.exists():
            raise FileNotFoundError(f"File not found: {fp}")
        body_text = fp.read_text(encoding="utf-8")
    else:
        body_text = args.body

    result = classify_text(
        title=args.title,
        body=body_text,
        max_length=args.max_length,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    # Human-friendly summary
    print(f"\nMain topic: {result['main_topic']}\n")
    print("Top 5 topics:")
    for i, (label, score) in enumerate(zip(result["topics"], result["topic_scores"])):
        if i >= 5:
            break
        print(f"  {i+1}. {label:20s}  prob={score:.4f}")


if __name__ == "__main__":
    main()
