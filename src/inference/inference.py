"""
Unified inference pipeline for the Reuters-trained DistilBERT model.
Loads the model, tokenizer, device, and label mapping (labels.json),
then exposes `classify_text()` for easy text → topic prediction.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from src.utils.config_loader import load_config

# ------------------------------------------------------------------
# Cached global objects (loaded once, reused for all classifications)
# ------------------------------------------------------------------
_TOKENIZER = None
_MODEL = None
_DEVICE = None
_ID2LABEL: Dict[int, str] = {}
_CONFIG = None


# ------------------------------------------------------------------
# Determine which device to use (cpu / cuda / auto)
# ------------------------------------------------------------------
def _resolve_device(pref: str):
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # "auto"


# ------------------------------------------------------------------
# Load model, tokenizer, and label mappings from the model directory.
# Uses caching so repeated calls don't reload weights.
# ------------------------------------------------------------------
def load_model(force_reload: bool = False):
    global _TOKENIZER, _MODEL, _DEVICE, _ID2LABEL, _CONFIG

    # Return cached model unless we explicitly reload
    if _MODEL is not None and not force_reload:
        print("Model already loaded. Using cached instance.")
        return _TOKENIZER, _MODEL, _DEVICE, _ID2LABEL

    # Load global configuration (paths, inference settings)
    cfg = load_config()
    _CONFIG = cfg
    inference_cfg = cfg.get("inference", {})

    # Model directory + device preference from config.yaml
    model_dir = Path(inference_cfg.get("model_dir", "models/bert_reuters21578"))
    device_pref = inference_cfg.get("device", "auto")

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir.resolve()}")

    # Load tokenizer, model config, and model weights
    _TOKENIZER = AutoTokenizer.from_pretrained(model_dir)

    config = AutoConfig.from_pretrained(model_dir)

    print("Loading model weights...")
    _MODEL = AutoModelForSequenceClassification.from_pretrained(model_dir, config=config)
    _MODEL.eval()

    # Move model to CPU/GPU based on preference
    _DEVICE = _resolve_device(device_pref)
    _MODEL.to(_DEVICE)

    # ---------------------------------------------
    # 1. PRIMARY LABEL SOURCE → labels.json
    # ---------------------------------------------
    labels_file = model_dir / "labels.json"
    if labels_file.exists():
        with labels_file.open("r") as f:
            data = json.load(f)

        raw = data.get("id_to_label")
        if isinstance(raw, dict):
            # Convert JSON keys from strings → ints
            _ID2LABEL = {int(k): v for k, v in raw.items()}
            return _TOKENIZER, _MODEL, _DEVICE, _ID2LABEL

    # ---------------------------------------------
    # 2. FALLBACK → Use HF config.json id2label field
    # ---------------------------------------------
    print("labels.json missing. Checking config.json id2label...")
    config_map = getattr(config, "id2label", None)
    if config_map:
        _ID2LABEL = {int(k): v for k, v in config_map.items()}
        print(f"Loaded {len(_ID2LABEL)} labels from config.json.")
        return _TOKENIZER, _MODEL, _DEVICE, _ID2LABEL

    # ---------------------------------------------
    # 3. FINAL FALLBACK → Generate generic label names
    # ---------------------------------------------
    print("No label mapping found. Generating placeholder labels...")
    num_labels = getattr(config, "num_labels", None)
    if num_labels is None:
        raise ValueError("Model missing id2label and num_labels fields.")

    _ID2LABEL = {i: f"LABEL_{i}" for i in range(num_labels)}
    print(f"Generated {len(_ID2LABEL)} fallback labels.")

    return _TOKENIZER, _MODEL, _DEVICE, _ID2LABEL


# ------------------------------------------------------------------
# Combine title + body into one input sequence for BERT
# ------------------------------------------------------------------
def _build_text(title: Optional[str], body: str) -> str:
    title = (title or "").strip()
    body = body.strip()
    return f"{title} [SEP] {body}" if title else body


# ------------------------------------------------------------------
# Main public API: classify text → return sorted topic predictions
# ------------------------------------------------------------------
def classify_text(title: Optional[str], body: str, max_length: Optional[int] = None):
    print("Beginning classification...")
    tokenizer, model, device, id2label = load_model()

    # Use config default max_length if not manually provided
    cfg = _CONFIG or {}
    if max_length is None:
        max_length = cfg.get("inference", {}).get("max_length", 512)

    # Prepare final input text
    text = _build_text(title, body)

    # Tokenize and move tensors to target device
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}

    # Forward pass (no gradients needed during inference)
    with torch.no_grad():
        logits = model(**tokens).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    # Pair each label with its probability
    scores = [(id2label[i], float(probs[i])) for i in range(len(probs))]
    scores.sort(key=lambda x: x[1], reverse=True)  # highest first

    # Unpack sorted labels and scores
    topics = [s[0] for s in scores]
    topic_scores = [s[1] for s in scores]

    print("Classification complete")
    return {
        "main_topic": topics[0],       # Best predicted topic
        "topics": topics,             # Ordered list of labels
        "topic_scores": topic_scores, # Corresponding probabilities
    }
