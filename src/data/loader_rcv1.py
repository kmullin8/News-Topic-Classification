# Loads the RCV1-v2 dataset from HuggingFace (raw text version) and converts
# its multi-label targets into a single dominant label per document so that
# it works with our single-label DistilBERT classifier.

from dataclasses import dataclass
from typing import Dict
import numpy as np

from datasets import load_dataset, DatasetDict


@dataclass
class DatasetInfo:
    """
    Holds:
      - dataset: HuggingFace DatasetDict with 'train' and 'test'
      - label2id: mapping from string label → integer ID
      - id2label: mapping from integer ID → string label
    Matching the structure used in loader_20ng.py so training code stays consistent.
    """
    dataset: DatasetDict
    label2id: Dict[str, int]
    id2label: Dict[int, str]


def load_rcv1_dataset() -> DatasetInfo:
    """
    Loads the RCV1-v2 dataset from the HuggingFace Hub.

    The HuggingFace version contains:
      - 'text' field: raw text for each news article
      - 'labels': multi-hot encoded list of topic IDs

    Since DistilBERT in this project is a *single-label classifier*, we reduce
    the multi-label target to a single "dominant" label per document using:
         dominant_label = argmax(multi_label_vector)

    Returns:
        DatasetInfo containing:
            - DatasetDict(train=..., test=...)
            - label2id / id2label mappings
    """

    print("Loading RCV1-v2 dataset from HuggingFace (raw text version)...")
    ds = load_dataset("rcv1")   # Automatically loads the 'train' and 'test' splits

    # Extract class label names
    label_names = ds["train"].features["labels"].names
    num_labels = len(label_names)

    # Build mappings
    id2label = {i: label_names[i] for i in range(num_labels)}
    label2id = {v: k for k, v in id2label.items()}

    print(f"➡ Found {num_labels} unique topic labels.")

    # --- Convert multi-label → dominant single label -------------------------
    # Each example["labels"] is a *multi-hot list*, e.g. [0,1,0,0,1,...]
    # Taking argmax gives us the "primary" topic ID.
    def convert_to_single_label(example):
        labels = example["labels"]  # list of ints (0 or 1)
        dominant_id = int(np.argmax(labels))
        return {"label": dominant_id}

    print("Converting multi-label vectors → single dominant label...")
    ds = ds.map(convert_to_single_label)

    # Remove the original multi-label field now that we have a single label
    ds = ds.remove_columns(["labels"])

    # Ensure dataset only has text + label fields
    print("🧹 Fields after cleaning:", ds["train"].column_names)

    # Wrap into DatasetDict so it matches 20NG structure
    dataset = DatasetDict(
        train=ds["train"],
        test=ds["test"],
    )

    print("RCV1-v2 dataset successfully loaded and processed.")
    print(f"   Train samples: {len(dataset['train'])}")
    print(f"   Test samples:  {len(dataset['test'])}")

    return DatasetInfo(
        dataset=dataset,
        label2id=label2id,
        id2label=id2label,
    )
