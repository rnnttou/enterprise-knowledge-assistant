"""Step 11: read scanned PDFs and images with layout detection + OCR.

Pages without a usable text layer (scans, photos, screenshots) go through
two models:

1. **Layout detection** with DocLayout-YOLO, a YOLO model pre-trained on
   document pages. It finds the regions of a page -- titles, paragraphs,
   tables, figures, captions -- and flags the ones to ignore (running
   headers, footers, page numbers).
2. **OCR** with EasyOCR on each region, cropped from the page.

The regions are then put back in reading order (two-column layouts
included) and joined into plain text, which `app.chunking` splits and the
vector store indexes like any other page.

Both models are heavy optional dependencies (PyTorch). If they aren't
installed, `get_default_ocr()` returns None and scanned pages are reported
as empty, exactly as before this step. Models are downloaded and loaded on
first use only, so indexing text PDFs never pays for them.

Usage:
    python -m app.ocr path/to/scan.png        # print the text of one file
    python -m app.ocr path/to/scan.pdf --regions
"""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.config import (
    LAYOUT_CONFIDENCE,
    LAYOUT_MODEL_FILE,
    LAYOUT_MODEL_REPO,
    OCR_DPI,
    OCR_LANGUAGES,
    OCR_MIN_CONFIDENCE,
    OCR_MODE,
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# DocLayout-YOLO (DocStructBench) class names -> the few kinds we act on.
LABELS = {
    "title": "title",
    "plain text": "text",
    "abandon": "abandon",            # headers, footers, page numbers
    "figure": "figure",
    "figure_caption": "caption",
    "table": "table",
    "table_caption": "caption",
    "table_footnote": "caption",
    "isolate_formula": "formula",
    "formula_caption": "caption",
}

# Regions we don't OCR: formulas (OCR turns math into noise).
SKIPPED_KINDS = {"formula"}

# "abandon" regions are what the model thinks is page furniture (running
# headers, footers, page numbers). It also flags key-figure strips at the
# bottom of report pages ("2,7 MILLIONS D'ABONNES"), so instead of dropping
# them unread we keep their text when it has at least this many words.
MIN_MARGIN_WORDS = 3

# A region wider than this share of the page spans both columns.
FULL_WIDTH_RATIO = 0.55

Box = tuple[float, float, float, float]   # x0, y0, x1, y1 in pixels


@dataclass
class Region:
    kind: str        # title, text, table, figure, caption, formula, abandon
    box: Box
    score: float = 1.0


@dataclass
class OcrLine:
    text: str
    box: Box
    confidence: float = 1.0


class LayoutDetector(Protocol):
    def detect(self, image) -> list[Region]: ...


class TextRecognizer(Protocol):
    def read(self, image) -> list[OcrLine]: ...


# --- geometry helpers ------------------------------------------------------

def _area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(a: Box, b: Box) -> float:
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def dedupe(regions: list[Region], threshold: float = 0.8) -> list[Region]:
    """Drop a region mostly contained in a higher-scoring one.

    The detector sometimes reports the same paragraph twice (e.g. as one
    block and as two halves); OCR-ing both would index the text twice.
    """
    kept: list[Region] = []
    for region in sorted(regions, key=lambda r: r.score, reverse=True):
        area = _area(region.box) or 1.0
        if all(_intersection(region.box, k.box) / area < threshold for k in kept):
            kept.append(region)
    return kept


def reading_order(regions: list[Region], page_width: float) -> list[Region]:
    """Sort regions the way a person reads the page.

    Regions spanning the page (titles, wide tables) cut it into horizontal
    bands. Inside a band, the left column is read top to bottom before the
    right one, so two-column text isn't interleaved line by line.
    """
    ordered: list[Region] = []
    band: list[Region] = []

    def flush():
        middle = page_width / 2
        left = [r for r in band if (r.box[0] + r.box[2]) / 2 < middle]
        right = [r for r in band if (r.box[0] + r.box[2]) / 2 >= middle]
        for column in (left, right):
            ordered.extend(sorted(column, key=lambda r: (r.box[1], r.box[0])))
        band.clear()

    for region in sorted(regions, key=lambda r: (r.box[1], r.box[0])):
        if region.box[2] - region.box[0] >= FULL_WIDTH_RATIO * page_width:
            flush()
            ordered.append(region)
        else:
            band.append(region)
    flush()
    return ordered


def group_rows(lines: list[OcrLine]) -> list[list[OcrLine]]:
    """Group OCR lines sitting on the same visual row, each row left to right.

    Two lines share a row when their vertical spans overlap by more than
    half of the shorter one.
    """
    rows: list[list[OcrLine]] = []
    for line in sorted(lines, key=lambda l: (l.box[1] + l.box[3]) / 2):
        top, bottom = line.box[1], line.box[3]
        if rows:
            row = rows[-1]
            row_top = min(l.box[1] for l in row)
            row_bottom = max(l.box[3] for l in row)
            overlap = min(bottom, row_bottom) - max(top, row_top)
            shorter = min(bottom - top, row_bottom - row_top) or 1.0
            if overlap / shorter > 0.5:
                row.append(line)
                continue
        rows.append([line])
    return [sorted(row, key=lambda l: l.box[0]) for row in rows]


def split_columns(lines: list[OcrLine]) -> list[list[OcrLine]]:
    """Split lines into side-by-side columns separated by wide empty gaps.

    A paragraph's lines cover its whole width, so it stays one column. A
    strip of separate items (key figures next to icons) has wide gaps
    between them; without this, its rows would interleave the items.
    """
    if not lines:
        return []
    heights = sorted(l.box[3] - l.box[1] for l in lines)
    min_gap = 1.5 * heights[len(heights) // 2]
    columns: list[list[OcrLine]] = []
    right_edge = None
    for line in sorted(lines, key=lambda l: l.box[0]):
        if right_edge is None or line.box[0] - right_edge > min_gap:
            columns.append([])
            right_edge = line.box[2]
        columns[-1].append(line)
        right_edge = max(right_edge, line.box[2])
    return columns


def lines_to_text(lines: list[OcrLine]) -> str:
    """Paragraph text: columns left to right, one output line per visual row."""
    return "\n".join(
        " ".join(l.text for l in row)
        for column in split_columns(lines)
        for row in group_rows(column)
    )


def table_to_text(lines: list[OcrLine]) -> str:
    """Table text: one line per row, cells separated by " | ".

    Keeping each row on one line keeps a value next to its row label, which
    is what lets a question like "subscribers in Q2" match the right number.
    """
    return "\n".join(" | ".join(l.text for l in row) for row in group_rows(lines))


def region_text(kind: str, lines: list[OcrLine]) -> str:
    if not lines:
        return ""
    if kind == "table":
        return table_to_text(lines)
    text = lines_to_text(lines)
    if kind == "abandon":
        words = [w for w in text.split() if any(c.isalpha() for c in w)]
        return text if len(words) >= MIN_MARGIN_WORDS else ""   # drop "Page 3"
    if kind == "figure":
        return f"[Figure] {text}"
    return text


# --- the pipeline -------------------------------------------------------------

@dataclass
class PageResult:
    text: str
    regions: list[tuple[Region, str]]   # (region, its OCR text), in reading order


class PageOcr:
    """Layout detection, then OCR of each detected region, in reading order."""

    def __init__(self, detector: LayoutDetector, recognizer: TextRecognizer,
                 min_confidence: float = OCR_MIN_CONFIDENCE, padding: int = 4):
        self.detector = detector
        self.recognizer = recognizer
        self.min_confidence = min_confidence
        self.padding = padding

    def _read(self, image) -> list[OcrLine]:
        return [l for l in self.recognizer.read(image)
                if l.text.strip() and l.confidence >= self.min_confidence]

    def read_page(self, image) -> PageResult:
        width, height = image.size
        regions = [r for r in self.detector.detect(image) if r.kind not in SKIPPED_KINDS]
        regions = reading_order(dedupe(regions), width)

        if not regions:
            # No layout found (e.g. a photo of a single note): OCR the whole image.
            text = lines_to_text(self._read(image))
            return PageResult(text=text, regions=[])

        results = []
        for region in regions:
            x0, y0, x1, y1 = region.box
            p = self.padding
            crop = image.crop((max(0, int(x0) - p), max(0, int(y0) - p),
                               min(width, int(x1) + p), min(height, int(y1) + p)))
            results.append((region, region_text(region.kind, self._read(crop))))

        text = "\n\n".join(t for _, t in results if t)
        return PageResult(text=text, regions=results)

    def page_text(self, image) -> str:
        return self.read_page(image).text


# --- real models (loaded lazily) --------------------------------------------

def _device() -> str:
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


class DocLayoutYoloDetector:
    """DocLayout-YOLO, pre-trained on DocStructBench, from the Hugging Face Hub."""

    def __init__(self, repo: str = LAYOUT_MODEL_REPO, filename: str = LAYOUT_MODEL_FILE,
                 confidence: float = LAYOUT_CONFIDENCE, image_size: int = 1024):
        self.repo, self.filename = repo, filename
        self.confidence, self.image_size = confidence, image_size
        self._model = None

    def _load(self):
        if self._model is None:
            from doclayout_yolo import YOLOv10
            from huggingface_hub import hf_hub_download
            weights = hf_hub_download(repo_id=self.repo, filename=self.filename)
            self._model = YOLOv10(weights)
            self._device = _device()
        return self._model

    def detect(self, image) -> list[Region]:
        model = self._load()
        result = model.predict(image.convert("RGB"), imgsz=self.image_size,
                               conf=self.confidence, device=self._device, verbose=False)[0]
        regions = []
        for box, cls, score in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                   result.boxes.conf.tolist()):
            name = result.names[int(cls)]
            regions.append(Region(kind=LABELS.get(name, "text"), box=tuple(box), score=float(score)))
        return regions


class EasyOcrRecognizer:
    """EasyOCR: pure pip install (no system binary), supports French and English."""

    def __init__(self, languages: list[str] = OCR_LANGUAGES):
        self.languages = languages
        self._reader = None

    def _load(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(self.languages, gpu=_device() != "cpu", verbose=False)
        return self._reader

    def read(self, image) -> list[OcrLine]:
        import numpy as np
        if min(image.size) < 8:
            return []
        results = self._load().readtext(np.array(image.convert("RGB")), detail=1, paragraph=False)
        lines = []
        for points, text, confidence in results:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            lines.append(OcrLine(text=text, box=(min(xs), min(ys), max(xs), max(ys)),
                                 confidence=float(confidence)))
        return lines


def ocr_available() -> bool:
    """True if the optional OCR dependencies are installed."""
    return all(importlib.util.find_spec(m) is not None
               for m in ("doclayout_yolo", "easyocr", "pypdfium2", "torch"))


_DEFAULT_OCR: PageOcr | None = None


def get_default_ocr() -> PageOcr | None:
    """The shared OCR pipeline, or None if OCR is turned off or not installed.

    Cheap to call: models load on the first page that actually needs OCR.
    """
    global _DEFAULT_OCR
    if OCR_MODE == "off" or not ocr_available():
        return None
    if _DEFAULT_OCR is None:
        _DEFAULT_OCR = PageOcr(DocLayoutYoloDetector(), EasyOcrRecognizer())
    return _DEFAULT_OCR


# --- turning files into page images -----------------------------------------

def render_pdf_pages(path: Path, page_indexes: list[int] | None = None, dpi: int = OCR_DPI):
    """Yield (0-based index, PIL image) for PDF pages (pypdfium2: no Poppler needed)."""
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(str(path))
    try:
        for index in (range(len(pdf)) if page_indexes is None else page_indexes):
            yield index, pdf[index].render(scale=dpi / 72).to_pil()
    finally:
        pdf.close()


def image_pages(path: Path) -> list:
    """Every frame of an image file (multi-page TIFFs have several)."""
    from PIL import Image, ImageOps
    pages = []
    with Image.open(path) as image:
        for frame in range(getattr(image, "n_frames", 1)):
            image.seek(frame)
            # Honour the EXIF rotation phones write, or OCR reads sideways text.
            pages.append(ImageOps.exif_transpose(image).convert("RGB"))
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Layout detection + OCR on one file.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--regions", action="store_true", help="show each detected region")
    args = parser.parse_args()

    if not ocr_available():
        raise SystemExit("OCR dependencies missing: pip install -r requirements-ocr.txt")
    ocr = PageOcr(DocLayoutYoloDetector(), EasyOcrRecognizer())

    if args.path.suffix.lower() == ".pdf":
        images = (image for _, image in render_pdf_pages(args.path))
    else:
        images = iter(image_pages(args.path))

    for number, image in enumerate(images, start=1):
        result = ocr.read_page(image)
        print(f"===== page {number} =====")
        if args.regions:
            for region, text in result.regions:
                box = ", ".join(f"{v:.0f}" for v in region.box)
                print(f"--- {region.kind} ({region.score:.2f}) [{box}]")
                print(text or "(no text)")
        else:
            print(result.text)


if __name__ == "__main__":
    main()
