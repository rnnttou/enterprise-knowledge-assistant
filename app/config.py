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
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-flash-latest")

# Embeddings (used in Step 3). 768 dimensions keeps the index small with
# little quality loss; changing it means re-embedding into a new collection.
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "50"))   # texts per API call

# How we cut documents into pieces (used in Step 2)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))        # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))   # characters shared between neighbours

# Question answering (used in Step 4). Search returns a cosine distance per
# chunk: 0 = same meaning, higher = less related. If even the best chunk is
# further away than this, we refuse instead of asking the model. 0.45 sits
# between on-topic questions (~0.25-0.35) and off-topic ones (~0.5) on our PDFs.
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.45"))

# Scanned documents and images (Step 11, needs requirements-ocr.txt).
#   auto:   OCR only PDF pages whose text layer has fewer than OCR_MIN_CHARS
#           characters (scans), plus every image file
#   always: OCR every PDF page too, ignoring the text layer
#   off:    never OCR; scanned pages are reported as empty
OCR_MODE = os.getenv("OCR_MODE", "auto").strip().lower()
OCR_MIN_CHARS = int(os.getenv("OCR_MIN_CHARS", "20"))
OCR_LANGUAGES = [l.strip() for l in os.getenv("OCR_LANGUAGES", "en,fr").split(",") if l.strip()]
OCR_DPI = int(os.getenv("OCR_DPI", "200"))                         # PDF page render resolution
OCR_MIN_CONFIDENCE = float(os.getenv("OCR_MIN_CONFIDENCE", "0.3"))  # drop OCR words below this
LAYOUT_MODEL_REPO = os.getenv("LAYOUT_MODEL_REPO", "juliozhao/DocLayout-YOLO-DocStructBench")
LAYOUT_MODEL_FILE = os.getenv("LAYOUT_MODEL_FILE", "doclayout_yolo_docstructbench_imgsz1024.pt")
LAYOUT_CONFIDENCE = float(os.getenv("LAYOUT_CONFIDENCE", "0.25"))


if __name__ == "__main__":
    # Quick sanity check: python -m app.config
    print(f"Data folder:     {DATA_DIR}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Embedding dim:   {EMBEDDING_DIM}")
    print(f"Chat model:      {CHAT_MODEL}")
    print(f"API key set:     {'yes' if GEMINI_API_KEY else 'NO - add it to .env'}")
    from app.ocr import ocr_available
    print(f"OCR mode:        {OCR_MODE} ({'installed' if ocr_available() else 'not installed'})")
