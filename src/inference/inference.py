"""
Unified article classifier for News-Topic-Classification.
Loads a fine-tuned DistilBERT model (Reuters-21578 by default) and exposes:

- load_model()
- classify_text(title, body, max_length=None)

Returns:
{
    "main_topic": str,
    "topics": [str],
    "topic_scores": [float]
}
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import json

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
)

from src.utils.config_loader import load_config

# Cached components so we only load once
_TOKENIZER = None
_MODEL = None
_DEVICE = None
_ID2LABEL: Dict[int, str] = {}
_CONFIG = None


# ---------------------------------------------------------
# Device resolution
# ---------------------------------------------------------
def _resolve_device(pref: str):
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # auto


# ---------------------------------------------------------
# Load model + tokenizer + labels
# ---------------------------------------------------------
def load_model(force_reload: bool = False):
    global _TOKENIZER, _MODEL, _DEVICE, _ID2LABEL, _CONFIG

    if _MODEL is not None and not force_reload:
        return _TOKENIZER, _MODEL, _DEVICE, _ID2LABEL

    cfg = load_config()
    _CONFIG = cfg

    inference_cfg = cfg.get("inference", {})
    model_dir = Path(inference_cfg.get("model_dir", "models/bert_reuters21578"))
    device_pref = inference_cfg.get("device", "auto")

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir.resolve()}")

    # Load tokenizer + config + weights
    _TOKENIZER = AutoTokenizer.from_pretrained(model_dir)
    config = AutoConfig.from_pretrained(model_dir)
    _MODEL = AutoModelForSequenceClassification.from_pretrained(model_dir, config=config)
    _MODEL.eval()

    _DEVICE = _resolve_device(device_pref)
    _MODEL.to(_DEVICE)

    # ---------------------------------------------------------
    # LABEL LOADING PRIORITY:
    #   1. labels.json
    #   2. config.id2label
    #   3. fallback LABEL_#
    # ---------------------------------------------------------
    labels_json_path = model_dir / "labels.json"
    if labels_json_path.exists():
        with open(labels_json_path, "r") as f:
            data = json.load(f)
        id2label_raw = data.get("id_to_label")
        if isinstance(id2label_raw, dict):
            _ID2LABEL = {int(k): v for k, v in id2label_raw.items()}
            return _TOKENIZER, _MODEL, _DEVICE, _ID2LABEL

    # fallback: use config.json mappings
    cfg_id2label = getattr(config, "id2label", None)
    if cfg_id2label:
        _ID2LABEL = {int(k): v for k, v in cfg_id2label.items()}
        return _TOKENIZER, _MODEL, _DEVICE, _ID2LABEL

    # final fallback: generic labels
    num_labels = getattr(config, "num_labels", None)
    if num_labels is None:
        raise ValueError("Model does not contain id2label or num_labels.")
    _ID2LABEL = {i: f"LABEL_{i}" for i in range(num_labels)}

    return _TOKENIZER, _MODEL, _DEVICE, _ID2LABEL


# ---------------------------------------------------------
# Build text input: "[TITLE] [SEP] body"
# ---------------------------------------------------------
def _build_text(title: Optional[str], body: str) -> str:
    title = (title or "").strip()
    body = body.strip()

    if title and body:
        return f"{title} [SEP] {body}"
    return title or body


# ---------------------------------------------------------
# Main classification function
# ---------------------------------------------------------
def classify_text(title: Optional[str], body: str, max_length: Optional[int] = None):
    if not body or not body.strip():
        raise ValueError("Body text is required for classification.")

    tokenizer, model, device, id2label = load_model()

    # Load default max_length if not provided
    cfg = _CONFIG or {}
    if max_length is None:
        max_length = cfg.get("inference", {}).get("max_length", 512)

    text = _build_text(title, body)

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    # Pair (label, probability)
    label_scores = [(id2label[i], float(probs[i])) for i in range(len(probs))]
    label_scores.sort(key=lambda x: x[1], reverse=True)

    topics = [lbl for lbl, _ in label_scores]
    topic_scores = [p for _, p in label_scores]

    return {
        "main_topic": topics[0],
        "topics": topics,
        "topic_scores": topic_scores,
    }
