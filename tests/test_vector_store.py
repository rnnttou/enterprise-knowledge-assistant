"""Vector store tests with a fake embedder, so they run offline and cost nothing."""

import hashlib
import shutil
from pathlib import Path

import chromadb
import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.vector_store import get_collection, index_documents, search


class FakeEmbedder:
    model = "fake-embedder"

    def __init__(self):
        self.calls = 0

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.lower().encode()).digest()
        return [b / 255 for b in digest[:8]]

    def embed_documents(self, texts):
        self.calls += 1
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


def write_pdf(path: Path, pages: list[str]) -> None:
    """Build a small text PDF with one string per page."""
    writer = PdfWriter()
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})
        })
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        writer.write(f)


@pytest.fixture
def collection():
    client = chromadb.EphemeralClient()
    col = get_collection(client, model="fake-embedder", dim=8)
    yield col
    client.delete_collection(col.name)


@pytest.fixture
def data_dir(tmp_path):
    write_pdf(tmp_path / "handbook.pdf", ["Holiday policy page one", "Remote work page two"])
    write_pdf(tmp_path / "hr" / "salaries.pdf", ["Salary bands are confidential"])
    yield tmp_path
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_indexes_every_page_with_metadata(collection, data_dir):
    stats = index_documents(collection, FakeEmbedder(), data_dir)

    assert sorted(stats.indexed) == ["handbook.pdf", "hr/salaries.pdf"]
    assert collection.count() == 3
    meta = collection.get(where={"source": "hr/salaries.pdf"})["metadatas"][0]
    assert meta["page"] == 1
    assert meta["allowed_groups"] == "hr"
    assert meta["grp_hr"] is True
    assert meta["embed_model"] == "fake-embedder"


def test_reindexing_unchanged_files_does_nothing(collection, data_dir):
    index_documents(collection, FakeEmbedder(), data_dir)
    ids_before = sorted(collection.get()["ids"])

    embedder = FakeEmbedder()
    stats = index_documents(collection, embedder, data_dir)

    assert embedder.calls == 0
    assert len(stats.skipped) == 2
    assert sorted(collection.get()["ids"]) == ids_before


def test_changed_file_replaces_its_chunks(collection, data_dir):
    index_documents(collection, FakeEmbedder(), data_dir)
    write_pdf(data_dir / "handbook.pdf", ["Only one page now"])

    stats = index_documents(collection, FakeEmbedder(), data_dir)

    assert stats.indexed == ["handbook.pdf"]
    handbook = collection.get(where={"source": "handbook.pdf"})
    assert len(handbook["ids"]) == 1
    assert "Only one page now" in handbook["documents"][0]


def test_deleted_file_is_removed(collection, data_dir):
    index_documents(collection, FakeEmbedder(), data_dir)
    (data_dir / "hr" / "salaries.pdf").unlink()

    stats = index_documents(collection, FakeEmbedder(), data_dir)

    assert len(stats.removed) == 1
    assert collection.get(where={"source": "hr/salaries.pdf"})["ids"] == []


def test_search_hides_documents_outside_the_users_groups(collection, data_dir):
    index_documents(collection, FakeEmbedder(), data_dir)

    public_hits = search(collection, FakeEmbedder(), "salary", groups=[], k=10)
    hr_hits = search(collection, FakeEmbedder(), "salary", groups=["hr"], k=10)

    assert {h["source"] for h in public_hits} == {"handbook.pdf"}
    assert {h["source"] for h in hr_hits} == {"handbook.pdf", "hr/salaries.pdf"}
