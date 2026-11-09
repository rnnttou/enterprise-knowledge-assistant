"""Step 2: load PDFs (and, since Step 11, images) and split them into chunks.

Each chunk comes from exactly one page, so every answer can cite a single
page. Chunk ids are stable: re-reading the same file gives the same ids,
which lets the vector store replace a document's chunks instead of
duplicating them.

Pages come from the PDF's text layer when it has one. Scanned pages and
image files are read by `app.ocr` (layout detection + OCR) instead, and
their chunks are marked `extraction="ocr"`.
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, OCR_MIN_CHARS, OCR_MODE
from app.ocr import IMAGE_SUFFIXES, get_default_ocr, image_pages, render_pdf_pages

PUBLIC_GROUP = "public"

# Where we prefer to cut, best first: paragraph, line, sentence, word.
SEPARATORS = ["\n\n", "\n", ". ", " "]

SUPPORTED_SUFFIXES = {".pdf"} | IMAGE_SUFFIXES

# Default for chunk_document's `ocr`: use the shared pipeline if installed.
AUTO = "auto"


@dataclass
class Document:
    """A PDF or image found in the data folder, before it is read."""

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
    extraction: str = "text"   # "text" (PDF text layer) or "ocr"

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
    """Every PDF and image under data_dir, including sub-folders."""
    paths = sorted(p for p in data_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
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


def page_has_images(page) -> bool:
    """Does the page draw any raster image (a scan is one big image)?"""
    def walk(resources, depth=0) -> bool:
        if resources is None or depth > 3:
            return False
        xobjects = resources.get_object().get("/XObject")
        if xobjects is None:
            return False
        for ref in xobjects.get_object().values():
            obj = ref.get_object()
            if obj.get("/Subtype") == "/Image":
                return True
            if obj.get("/Subtype") == "/Form" and walk(obj.get("/Resources"), depth + 1):
                return True
        return False

    try:
        return walk(page.get("/Resources"))
    except Exception:   # malformed resources: let OCR decide
        return True


def needs_ocr(text: str, has_images: bool, mode: str = OCR_MODE,
              min_chars: int = OCR_MIN_CHARS) -> bool:
    """Should this PDF page be OCR'd instead of trusting its text layer?

    In auto mode: only pages with (almost) no text that do contain an image.
    A short page without images (a divider, "Notes") has nothing to read.
    """
    if mode == "off":
        return False
    if mode == "always":
        return True
    return len(text.strip()) < min_chars and has_images


def read_pages(doc: Document, ocr) -> tuple[str, list[tuple[int, str, str]]]:
    """(title, [(page number, cleaned text, extraction)]) for one document."""
    if doc.path.suffix.lower() in IMAGE_SUFFIXES:
        title = doc.path.stem.replace("-", " ").replace("_", " ")
        if ocr is None:
            return title, []
        pages = []
        for number, image in enumerate(image_pages(doc.path), start=1):
            print(f"OCR {doc.source} page {number}...")
            pages.append((number, clean_text(ocr.page_text(image)), "ocr"))
        return title, pages

    reader = PdfReader(doc.path)
    title = pdf_title(reader, doc.path)
    pages, to_ocr = [], []
    for n, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        pages.append((n, text, "text"))
        if ocr is not None and needs_ocr(text, page_has_images(page)):
            to_ocr.append(n - 1)

    if to_ocr:
        for index, image in render_pdf_pages(doc.path, to_ocr):
            print(f"OCR {doc.source} page {index + 1} ({to_ocr.index(index) + 1}/{len(to_ocr)})...")
            text = clean_text(ocr.page_text(image))
            if text:   # keep the text layer if OCR finds nothing ("always" mode)
                pages[index] = (index + 1, text, "ocr")
    return title, pages


def chunk_document(doc: Document, ocr=AUTO) -> list[Chunk]:
    """Split a document into chunks.

    `ocr` reads scanned pages and images: a `PageOcr`, None to skip them,
    or AUTO for the shared pipeline (None when OCR isn't installed).
    """
    if ocr is AUTO:
        ocr = get_default_ocr()
    title, pages = read_pages(doc, ocr)
    chunks = []
    for page_number, text, extraction in pages:
        for index, piece in enumerate(split_text(text)):
            chunks.append(Chunk(
                chunk_id=make_chunk_id(doc.doc_id, page_number, index),
                doc_id=doc.doc_id,
                text=piece,
                page=page_number,
                chunk_index=index,
                title=title,
                extraction=extraction,
            ))
    return chunks


if __name__ == "__main__":
    # Quick look at what Step 2 produces: python -m app.chunking
    for doc in discover():
        chunks = chunk_document(doc)
        pages = len({c.page for c in chunks})
        ocr_pages = len({c.page for c in chunks if c.extraction == "ocr"})
        print(f"{doc.source}: {len(chunks)} chunks from {pages} pages "
              f"({ocr_pages} via OCR), groups={doc.allowed_groups}")
