from src.db.mongo_client import get_db

db = get_db()
db.articles.create_index("url", unique=True)
print("Index created!")
