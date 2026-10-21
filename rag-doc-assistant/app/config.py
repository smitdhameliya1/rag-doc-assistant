import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = "text-embedding-3-small"   # 1536 dimensions, cheap
EMBED_DIM = 1536
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 150     # overlap so ideas aren't cut in half
TOP_K = 4               # chunks retrieved per question
INDEX_DIR = "data/index"
