import sys
sys.path.insert(0, '.')
from app.generation.generator import AnswerGenerator
from app.ingestion.pipeline import IngestionPipeline
from app.database.chroma import ChromaClient
from app.config import COLLECTIONS
print("All imports OK")
print(f"Collections: {list(COLLECTIONS.keys())}")
