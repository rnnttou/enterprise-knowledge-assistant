import pytest

from app.chunking import clean_text, groups_for, make_chunk_id, split_text


def test_short_text_is_one_chunk():
    assert split_text("Hello world.", size=100, overlap=10) == ["Hello world."]


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = split_text(text, size=300, overlap=60)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    # neighbours share some text
    assert chunks[1][:20] in chunks[0]


def test_cuts_prefer_paragraph_breaks():
    para = "word " * 50
    chunks = split_text(f"{para}\n\n{para}", size=300, overlap=0)
    assert chunks[0] == para.strip()


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        split_text("abc", size=10, overlap=10)


def test_chunk_ids_are_stable_and_distinct():
    assert make_chunk_id("doc", 1, 0) == make_chunk_id("doc", 1, 0)
    assert make_chunk_id("doc", 1, 0) != make_chunk_id("doc", 2, 0)
    assert make_chunk_id("doc", 1, 0) != make_chunk_id("doc", 1, 1)


def test_groups_come_from_first_folder():
    assert groups_for("handbook.pdf") == ["public"]
    assert groups_for("HR/salaries.pdf") == ["hr"]
    assert groups_for("finance-team/q3/report.pdf") == ["finance_team"]


def test_clean_text_rejoins_hyphenated_words():
    assert clean_text("infor-\nmation   security\n\n\n\nnext") == "information security\n\nnext"
