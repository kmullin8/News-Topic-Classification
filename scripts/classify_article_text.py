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
# Hide noisy framework warnings (PyTorch / Transformers)
# ------------------------------------------------------------
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["PYTORCH_JIT_WARN"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ------------------------------------------------------------
# Make the src/ directory importable (so classify_text can be found)
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Core inference function (loads model + predicts)
from src.inference.inference import classify_text


def parse_args():
    """
    Define command-line arguments for the classifier.
    Allows input text via --body or from a file via --file.
    """
    print("Parsing title and body arguments...")

    parser = argparse.ArgumentParser(
        description="Classify a news article using the Reuters-trained DistilBERT model."
    )

    # Input source: either raw text OR a text file
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--body", type=str, help="Article body text")
    group.add_argument("--file", type=str, help="Path to text file containing the article body")

    # Optional metadata and settings
    parser.add_argument("--title", type=str, default="", help="Optional article title/headline")
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Return the full prediction dictionary in JSON format."
    )

    return parser.parse_args()


def main():
    print("\n=== News Article Topic Classifier ===")
    args = parse_args()

    print("Starting classifier...")

    # ------------------------------------------------------------
    # Load article body (either from file or --body argument)
    # ------------------------------------------------------------
    if args.file:
        fp = Path(args.file)
        if not fp.exists():
            raise FileNotFoundError(f"File not found: {fp}")
        body_text = fp.read_text(encoding="utf-8")
    else:
        body_text = args.body

    print("Input loaded. Calling inference engine...")

    # ------------------------------------------------------------
    # Run inference through src.inference.inference.classify_text
    # ------------------------------------------------------------
    result = classify_text(
        title=args.title,
        body=body_text,
        max_length=args.max_length,
    )

    print("Inference complete. Preparing output...")

    # ------------------------------------------------------------
    # JSON output mode for debugging or automated pipelines
    # ------------------------------------------------------------
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    # ------------------------------------------------------------
    # Human-friendly output: top topic + top 5 ranked topics
    # ------------------------------------------------------------
    print(f"\nMain topic: {result['main_topic']}")

    print("Top 5 topics:")
    for i, (label, score) in enumerate(zip(result["topics"], result["topic_scores"])):
        if i >= 5:
            break
        print(f"  {i+1}. {label:20s}  prob={score:.4f}")
    print("\n")

if __name__ == "__main__":
    # Entry point for CLI execution
    main()
