"""
MongoDB client utility.
Handles connection, deduplication checks, and document inserts.
"""

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from datetime import datetime

from src.utils.config_loader import load_config
import os

_client = None
_db = None


def get_db():
    """
    Returns a cached MongoDB database instance using credentials from .env + config.yaml.
    """
    global _client, _db

    if _db is not None:
        return _db

    cfg = load_config()
    db_cfg = cfg.get("database", {})

    user = os.environ.get("MONGODB_USER")
    password = os.environ.get("MONGODB_PASSWORD")
    host = os.environ.get("MONGODB_HOST")
    db_name = os.environ.get("MONGODB_DB")

    uri_template = db_cfg.get("uri_template")
    uri = uri_template.format(user=user, password=password, host=host)

    try:
        _client = MongoClient(uri)
        _db = _client[db_name]
        print(f"[MongoDB] Connected to database: {db_name}")
    except ConnectionFailure as e:
        raise RuntimeError(f"[MongoDB] Connection failed: {e}")

    return _db


# -------------------------------------------------------------
# Deduplication: check whether article was already digested
# -------------------------------------------------------------
def article_exists(url: str) -> bool:
    db = get_db()
    col_name = load_config()["database"]["collection"]
    col = db[col_name]

    return col.find_one({"url": url}) is not None


# -------------------------------------------------------------
# Save processed article to MongoDB
# -------------------------------------------------------------
def save_article(doc: dict):
    """
    Inserts article document into MongoDB.
    Automatically adds ingestion timestamp.
    """
    db = get_db()
    col_name = load_config()["database"]["collection"]
    col = db[col_name]

    doc = dict(doc)
    doc["ingested_at"] = datetime.utcnow()

    col.insert_one(doc)
    print(f"[MongoDB] Saved article: {doc.get('url')}")
