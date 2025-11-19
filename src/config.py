#handles loading configuration from yaml and environment variables

import yaml
import os
from dotenv import load_dotenv

load_dotenv() # take environment variables from .env.

def load_config():
    with open("config.yaml", "r") as f: #load configuration from yaml file convets to dictionary
        cfg = yaml.safe_load(f)

    cfg["mongo_uri"] = os.getenv("MONGO_URI") #load MongoDB URI from environment variable
    cfg["mongo_db"] = os.getenv("MONGO_DB") #load MongoDB database name from environment variable

    return cfg
