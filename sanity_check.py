# Sanity check script to verify that all required libraries are installed correctly.

def main():
    print("Running sanity check...")

    try:
        # --- Core ML stack ---
        import torch               # PyTorch backend
        import transformers        # HuggingFace model utilities
        import datasets            # HuggingFace datasets loader

        # --- Scraping + Parsing ---
        import feedparser          # RSS ingestion
        import newspaper           # Article extraction
        import bs4                 # BeautifulSoup (HTML parsing)

        # --- Database + Utilities ---
        import pymongo             # MongoDB client

        print("All imports succeeded.")
    except Exception as e:
        print("Import failed:", e)

if __name__ == "__main__":
    main()
