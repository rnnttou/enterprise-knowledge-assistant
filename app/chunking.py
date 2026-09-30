"""Step 2: load PDFs and split them into chunks.

Each chunk comes from exactly one page, so every answer can cite a single
page. Chunk ids are stable: re-reading the same file gives the same ids,
which lets the vector store replace a document's chunks instead of
duplicating them.
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR

PUBLIC_GROUP = "public"

# Where we prefer to cut, best first: paragraph, line, sentence, word.
SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Document:
    """A PDF found in the data folder, before it is read."""

    path: Path
    doc_id: str           # stable id derived from the path inside data/
    source: str           # path relative to data/, e.g. "hr/handbook.pdf"
    content_hash: str     # changes whenever the file's bytes change
    modified_at: str      # ISO timestamp of the last modification
    allowed_groups: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str             # the text shown to users and stored in Chroma
    page: int             # 1-based page number
    chunk_index: int      # position of the chunk within its page
    title: str

    @property
    def embed_text(self) -> str:
        """Text sent to the embedding model: the title gives short chunks context."""
        return f"{self.title}\n\n{self.text}"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_chunk_id(doc_id: str, page: int, chunk_index: int) -> str:
    return _sha256(f"{doc_id}:{page}:{chunk_index}".encode())[:32]


def groups_for(source: str) -> list[str]:
    """Access groups come from the first folder: data/hr/x.pdf -> ["hr"].

    Files directly in data/ are readable by everyone.
    """
    parts = Path(source).parts
    if len(parts) < 2:
        return [PUBLIC_GROUP]
    group = re.sub(r"[^a-z0-9_]", "_", parts[0].lower())
    return [group]


def describe(path: Path, data_dir: Path = DATA_DIR) -> Document:
    source = path.resolve().relative_to(data_dir.resolve()).as_posix()
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return Document(
        path=path,
        doc_id=_sha256(source.encode())[:16],
        source=source,
        content_hash=_sha256(path.read_bytes()),
        modified_at=modified.isoformat(timespec="seconds"),
        allowed_groups=groups_for(source),
    )


def discover(data_dir: Path = DATA_DIR) -> list[Document]:
    """Every PDF under data_dir, including sub-folders."""
    paths = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() == ".pdf")
    return [describe(p, data_dir) for p in paths]


def clean_text(text: str) -> str:
    """Undo common PDF extraction artefacts while keeping paragraph breaks."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)       # re-join hyphenated words
    text = re.sub(r"[ \t ]+", " ", text)            # collapse spaces
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Cut text into pieces of at most `size` characters.

    Each cut is moved back to the nearest paragraph, line, sentence or word
    break, and neighbouring pieces share about `overlap` characters.
    """
    if overlap >= size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
    text = text.strip()
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            for sep in SEPARATORS:
                cut = text.rfind(sep, start + size // 2, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        next_start = max(end - overlap, start + 1)
        space = text.find(" ", next_start, end)   # don't start mid-word
        start = space + 1 if space != -1 else next_start
    return chunks


def pdf_title(reader: PdfReader, path: Path) -> str:
    title = (reader.metadata.title if reader.metadata else None) or ""
    # Word exports often set the title to "Microsoft Word - file.docx"
    title = re.sub(r"^Microsoft Word - ", "", title.strip())
    title = re.sub(r"\.docx?$", "", title).strip()
    return title or path.stem.replace("-", " ").replace("_", " ")


def chunk_document(doc: Document) -> list[Chunk]:
    reader = PdfReader(doc.path)
    title = pdf_title(reader, doc.path)
    chunks = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        for index, piece in enumerate(split_text(text)):
            chunks.append(Chunk(
                chunk_id=make_chunk_id(doc.doc_id, page_number, index),
                doc_id=doc.doc_id,
                text=piece,
                page=page_number,
                chunk_index=index,
                title=title,
            ))
    return chunks


if __name__ == "__main__":
    # Quick look at what Step 2 produces: python -m app.chunking
    for doc in discover():
        chunks = chunk_document(doc)
        pages = len({c.page for c in chunks})
        print(f"{doc.source}: {len(chunks)} chunks from {pages} pages, groups={doc.allowed_groups}")
