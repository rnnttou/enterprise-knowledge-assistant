"""Central settings for the app.

Every value is read from environment variables (loaded from a .env file),
so secrets like the API key never live in the code.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root and fills os.environ

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")

# How we cut documents into pieces (used in Step 2)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))        # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))   # characters shared between neighbours


if __name__ == "__main__":
    # Quick sanity check: python -m app.config
    print(f"Data folder:     {DATA_DIR}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Chat model:      {CHAT_MODEL}")
    print(f"API key set:     {'yes' if GEMINI_API_KEY else 'NO - add it to .env'}")
