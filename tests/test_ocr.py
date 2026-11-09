"""OCR pipeline tests with fake layout/OCR models, so they run offline without PyTorch."""

from pathlib import Path

from PIL import Image
from pypdf import PdfWriter

from app.chunking import chunk_document, describe, discover, needs_ocr
from app.ocr import OcrLine, PageOcr, Region, dedupe, reading_order, region_text, table_to_text

WIDTH = 1000


class FakeDetector:
    def __init__(self, regions):
        self.regions = regions

    def detect(self, image):
        return list(self.regions)


class FakeRecognizer:
    """Returns the words whose centre falls inside the image it's given.

    Crops lose their page coordinates, so the fake tracks them by size:
    each call is matched against the region boxes it was built from.
    """

    def __init__(self, words: list[OcrLine], regions: list[Region] = ()):
        self.words = words
        self.regions = list(regions)
        self.calls = 0

    def read(self, image):
        self.calls += 1
        w, h = image.size
        # Find the region this crop came from (padding is 4px on each side).
        for region in self.regions:
            x0, y0, x1, y1 = region.box
            if abs((x1 - x0 + 8) - w) <= 8 and abs((y1 - y0 + 8) - h) <= 8:
                return [OcrLine(l.text, (l.box[0] - x0, l.box[1] - y0, l.box[2] - x0, l.box[3] - y0), l.confidence)
                        for l in self.words if x0 <= (l.box[0] + l.box[2]) / 2 <= x1
                        and y0 <= (l.box[1] + l.box[3]) / 2 <= y1]
        return list(self.words)   # whole page


def line(text, x0, y0, x1, y1, confidence=0.9):
    return OcrLine(text, (x0, y0, x1, y1), confidence)


def test_two_columns_are_read_left_then_right():
    regions = [
        Region("text", (520, 200, 950, 400)),   # right column, top
        Region("title", (50, 50, 950, 120)),    # spans the page
        Region("text", (50, 450, 480, 600)),    # left column, bottom
        Region("text", (50, 200, 480, 400)),    # left column, top
        Region("table", (50, 700, 950, 900)),   # spans the page
    ]
    order = [r.box for r in reading_order(regions, WIDTH)]
    assert order == [
        (50, 50, 950, 120),
        (50, 200, 480, 400), (50, 450, 480, 600),
        (520, 200, 950, 400),
        (50, 700, 950, 900),
    ]


def test_dedupe_drops_region_inside_a_better_one():
    big = Region("text", (0, 0, 500, 500), score=0.9)
    inner = Region("text", (10, 10, 490, 240), score=0.5)
    other = Region("text", (0, 600, 500, 700), score=0.4)
    assert dedupe([inner, big, other]) == [big, other]


def test_table_rows_keep_cells_on_one_line():
    lines = [
        line("Q2", 300, 52, 340, 70),
        line("Operator", 10, 10, 100, 30),
        line("Subscribers", 150, 12, 260, 28),
        line("Maroc Telecom", 10, 50, 140, 70),
        line("18.9M", 150, 50, 210, 72),
    ]
    assert table_to_text(lines) == "Operator | Subscribers\nMaroc Telecom | 18.9M | Q2"


def test_figure_text_is_labelled():
    assert region_text("figure", [line("Revenue 2019", 0, 0, 100, 20)]) == "[Figure] Revenue 2019"
    assert region_text("figure", []) == ""


def test_page_ocr_skips_page_furniture_and_orders_regions():
    regions = [
        Region("abandon", (50, 960, 950, 990)),   # footer "Page 3"
        Region("text", (50, 200, 950, 300)),
        Region("title", (50, 50, 950, 120)),
    ]
    words = [
        line("Page 3", 400, 965, 500, 985),
        line("Annual", 60, 60, 200, 110), line("Report", 220, 60, 400, 110),
        line("Leave is 25 days.", 60, 210, 600, 240),
        line("low confidence junk", 60, 260, 600, 290, confidence=0.05),
    ]
    ocr = PageOcr(FakeDetector(regions), FakeRecognizer(words, regions))
    result = ocr.read_page(Image.new("RGB", (WIDTH, 1000), "white"))
    assert result.text == "Annual Report\n\nLeave is 25 days."
    assert [r.kind for r, text in result.regions if text] == ["title", "text"]


def test_page_without_layout_falls_back_to_whole_image():
    ocr = PageOcr(FakeDetector([]), FakeRecognizer([line("Sticky note: call IT", 0, 0, 300, 30)]))
    assert ocr.page_text(Image.new("RGB", (400, 100))) == "Sticky note: call IT"


def test_needs_ocr_modes():
    text = "A real text layer with plenty of words."
    assert needs_ocr("", has_images=True, mode="auto")             # a scan
    assert not needs_ocr("", has_images=False, mode="auto")        # blank divider page
    assert not needs_ocr(text, has_images=True, mode="auto")       # text PDF with a logo
    assert needs_ocr(text, has_images=False, mode="always")
    assert not needs_ocr("", has_images=True, mode="off")


def test_key_figure_strip_is_read_item_by_item():
    # Four items side by side, each on two lines, separated by icons (no text).
    lines = [
        line("2,7 MILLIONS", 130, 0, 260, 25), line("D'ABONNES", 130, 30, 250, 55),
        line("TEMPS MOYEN", 360, 0, 540, 25), line("37 MIN/MOIS", 360, 30, 520, 55),
        line("PARC FTTH", 990, 0, 1110, 25), line("730 MILLE", 990, 30, 1100, 55),
    ]
    assert region_text("text", lines) == (
        "2,7 MILLIONS\nD'ABONNES\nTEMPS MOYEN\n37 MIN/MOIS\nPARC FTTH\n730 MILLE"
    )


def test_margin_regions_keep_real_text_and_drop_page_numbers():
    assert region_text("abandon", [line("Page 3", 0, 0, 80, 20)]) == ""
    assert region_text("abandon", [line("- 12 -", 0, 0, 80, 20)]) == ""
    kpi = [line("730 MILLE ABONNES FTTH", 0, 0, 300, 20)]
    assert region_text("abandon", kpi) == "730 MILLE ABONNES FTTH"


class StubOcr:
    def __init__(self, text):
        self.text, self.calls = text, 0

    def page_text(self, image):
        self.calls += 1
        return self.text


def test_images_are_discovered_and_chunked_via_ocr(tmp_path: Path):
    Image.new("RGB", (200, 100), "white").save(tmp_path / "scanned-memo.png")
    (tmp_path / "notes.txt").write_text("ignored")
    docs = discover(tmp_path)
    assert [d.source for d in docs] == ["scanned-memo.png"]

    chunks = chunk_document(docs[0], StubOcr("The VPN must be used off-site."))
    assert [(c.text, c.page, c.extraction, c.title) for c in chunks] == [
        ("The VPN must be used off-site.", 1, "ocr", "scanned memo")
    ]
    assert chunk_document(docs[0], None) == []   # OCR unavailable -> reported empty


def test_scanned_pdf_pages_are_ocrd_but_blank_pages_are_not(tmp_path: Path):
    scan = tmp_path / "scan.pdf"
    Image.new("RGB", (200, 200), "white").save(scan)   # image-only PDF, like a scanner makes
    blank = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)       # no text and no image
    writer.write(blank)

    stub = StubOcr("Scanned contract, signed 2024.")
    chunks = chunk_document(describe(scan, tmp_path), stub)
    assert stub.calls == 1
    assert [(c.page, c.extraction, c.text) for c in chunks] == [(1, "ocr", "Scanned contract, signed 2024.")]

    assert chunk_document(describe(blank, tmp_path), stub) == []
    assert stub.calls == 1   # nothing to read on a blank page
