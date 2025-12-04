from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from src.utils.config_loader import load_config

_TOKENIZER = None
_MODEL = None
_DEVICE = None
_ID2LABEL: Dict[int, str] = {}
_CONFIG = None

def _resolve_device(pref: str):
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    _TOKENIZER = AutoTokenizer.from_pretrained(model_dir)
    config = AutoConfig.from_pretrained(model_dir)
    _MODEL = AutoModelForSequenceClassification.from_pretrained(model_dir, config=config)
    _MODEL.eval()
    _DEVICE = _resolve_device(device_pref)
    _MODEL.to(_DEVICE)

    # ---------------------------------------------
    # ALWAYS LOAD labels.json FIRST
    # ---------------------------------------------
    labels_file = model_dir / "labels.json"
    if labels_file.exists():
        with labels_file.open("r") as f:
            data = json.load(f)
        raw = data.get("id_to_label")
        if isinstance(raw, dict):
            _ID2LABEL = {int(k): v for k, v in raw.items()}
            return _TOKENIZER, _MODEL, _DEVICE, _ID2LABEL

    # fallback: config.json
    config_map = getattr(config, "id2label", None)
    if config_map:
        _ID2LABEL = {int(k): v for k, v in config_map.items()}
        return _TOKENIZER, _MODEL, _DEVICE, _ID2LABEL

    # final fallback
    num_labels = getattr(config, "num_labels", None)
    if num_labels is None:
        raise ValueError("Model missing id2label and num_labels")
    _ID2LABEL = {i: f"LABEL_{i}" for i in range(num_labels)}

    return _TOKENIZER, _MODEL, _DEVICE, _ID2LABEL

def _build_text(title: Optional[str], body: str) -> str:
    title = (title or "").strip()
    body = body.strip()
    return f"{title} [SEP] {body}" if title else body

def classify_text(title: Optional[str], body: str, max_length: Optional[int] = None):
    tokenizer, model, device, id2label = load_model()

    cfg = _CONFIG or {}
    if max_length is None:
        max_length = cfg.get("inference", {}).get("max_length", 512)

    text = _build_text(title, body)
    tokens = tokenizer(text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        logits = model(**tokens).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    scores = [(id2label[i], float(probs[i])) for i in range(len(probs))]
    scores.sort(key=lambda x: x[1], reverse=True)

    topics = [s[0] for s in scores]
    topic_scores = [s[1] for s in scores]

    return {
        "main_topic": topics[0],
        "topics": topics,
        "topic_scores": topic_scores,
    }
