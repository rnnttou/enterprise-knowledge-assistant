import pytest


@pytest.fixture(autouse=True)
def no_real_ocr(monkeypatch):
    """Keep tests offline and fast: never load the real layout/OCR models.

    Tests that exercise OCR pass a fake pipeline to chunk_document explicitly.
    """
    monkeypatch.setattr("app.chunking.get_default_ocr", lambda: None)
