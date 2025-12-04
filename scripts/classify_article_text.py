"""
CLI tool to classify a news article using the Reuters-trained DistilBERT model.
Provides simple top-topic output or full JSON when requested.
"""

import argparse
import json
import sys
import warnings
import os
from pathlib import Path

# ------------------------------------------------------------
# Suppress all warnings from PyTorch & Transformers
# ------------------------------------------------------------
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["PYTORCH_JIT_WARN"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Ensure src/ is importable when running from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.inference import classify_text  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify a news article using the Reuters-trained DistilBERT model."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--body", type=str, help="Article body text")
    group.add_argument("--file", type=str, help="Path to text file containing the article body")

    parser.add_argument("--title", type=str, default="", help="Optional article title/headline")
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--json", action="store_true",
                        help="Print full JSON output instead of simplified output.")

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

    # Run model inference
    result = classify_text(
        title=args.title,
        body=body_text,
        max_length=args.max_length,
    )

    # JSON mode (if needed for debugging/automation)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    # ------------------------------------------------------------
    # Simplified clean output: Only print top label + score
    # ------------------------------------------------------------
    main_topic = result["main_topic"]
    score = result["topic_scores"][0]

    #print(f"{main_topic} ({score:.5f})")

    print(f"\nMain topic: {result['main_topic']}\n")
    print("Top 5 topics:")
    for i, (label, score) in enumerate(zip(result["topics"], result["topic_scores"])):
        if i >= 5:
            break
        print(f"  {i+1}. {label:20s}  prob={score:.4f}")
    print("\n")


if __name__ == "__main__":
    main()
