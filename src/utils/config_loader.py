import yaml
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env


def load_config(path: str = "config.yaml"):
    """
    Loads YAML config and merges environment variables.
    """

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    # Load YAML
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Merge environment variables (MongoDB, API keys, etc.)
    cfg["mongo_uri"] = os.getenv("MONGO_URI")
    cfg["mongo_db"] = os.getenv("MONGO_DB")

    return cfg
