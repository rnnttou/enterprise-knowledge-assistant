"""Step 3b: store chunk embeddings in a persistent ChromaDB collection.

Indexing is safe to re-run:
- unchanged PDFs (same content hash) are skipped,
- changed PDFs have their old chunks replaced,
- PDFs removed from data/ have their chunks deleted.

Every chunk carries its document id, title, page and access groups as
metadata, so later steps can cite sources and filter by permission.

Usage:
    python -m app.vector_store                  # index everything in data/
    python -m app.vector_store --query "..."    # try a search
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import chromadb

from app.chunking import PUBLIC_GROUP, Document, chunk_document, discover
from app.config import CHROMA_DIR, DATA_DIR, EMBEDDING_DIM, EMBEDDING_MODEL


def collection_name(model: str = EMBEDDING_MODEL, dim: int = EMBEDDING_DIM) -> str:
    # One collection per model and size: vectors from different models can't be mixed.
    return f"chunks-{model}-{dim}".replace("/", "-")


def get_collection(client=None, model: str = EMBEDDING_MODEL, dim: int = EMBEDDING_DIM):
    client = client or chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=collection_name(model, dim),
        metadata={"embed_model": model, "embed_dim": dim},
        configuration={"hnsw": {"space": "cosine"}},
    )


def group_key(group: str) -> str:
    return f"grp_{group}"


def chunk_metadata(doc: Document, chunk, embed_model: str) -> dict:
    metadata = {
        "doc_id": doc.doc_id,
        "title": chunk.title,
        "source": doc.source,
        "page": chunk.page,
        "chunk_index": chunk.chunk_index,
        "content_hash": doc.content_hash,
        "modified_at": doc.modified_at,
        "embed_model": embed_model,
        "allowed_groups": ",".join(doc.allowed_groups),   # readable copy
    }
    # One boolean per group, because Chroma filters on scalar values.
    for group in doc.allowed_groups:
        metadata[group_key(group)] = True
    return metadata


@dataclass
class IndexStats:
    indexed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    empty: list[str] = field(default_factory=list)
    chunks_added: int = 0


def stored_hashes(collection) -> dict[str, str]:
    """doc_id -> content hash of every document currently in the collection."""
    result = collection.get(include=["metadatas"])
    return {m["doc_id"]: m["content_hash"] for m in result["metadatas"]}


def index_documents(collection, embedder, data_dir: Path = DATA_DIR) -> IndexStats:
    stats = IndexStats()
    docs = discover(data_dir)
    known = stored_hashes(collection)

    for doc in docs:
        if known.get(doc.doc_id) == doc.content_hash:
            stats.skipped.append(doc.source)
            continue

        chunks = chunk_document(doc)
        if not chunks:
            # Likely a scanned PDF with no text layer (OCR comes later).
            stats.empty.append(doc.source)
            collection.delete(where={"doc_id": doc.doc_id})
            continue

        print(f"Embedding {doc.source} ({len(chunks)} chunks)...")
        # Embed before deleting, so a failed API call leaves the old chunks in place.
        embeddings = embedder.embed_documents([c.embed_text for c in chunks])
        collection.delete(where={"doc_id": doc.doc_id})
        collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[chunk_metadata(doc, c, embedder.model) for c in chunks],
        )
        stats.indexed.append(doc.source)
        stats.chunks_added += len(chunks)

    current_ids = {d.doc_id for d in docs}
    for doc_id in known.keys() - current_ids:
        collection.delete(where={"doc_id": doc_id})
        stats.removed.append(doc_id)

    return stats


def group_filter(groups: list[str]) -> dict:
    """Chroma `where` filter: chunks readable by any of the user's groups (or by everyone)."""
    keys = sorted({group_key(g) for g in [*groups, PUBLIC_GROUP]})
    clauses = [{key: True} for key in keys]
    return clauses[0] if len(clauses) == 1 else {"$or": clauses}


def search(collection, embedder, question: str, groups: list[str] | None = None, k: int = 5) -> list[dict]:
    """Top-k chunks for a question, restricted to what the user's groups may read."""
    result = collection.query(
        query_embeddings=[embedder.embed_query(question)],
        n_results=k,
        where=group_filter(groups or []),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"chunk_id": cid, "text": text, "distance": dist, **meta}
        for cid, text, meta, dist in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


def main() -> None:
    from app.embeddings import GeminiEmbedder

    parser = argparse.ArgumentParser(description="Index PDFs into ChromaDB, or search them.")
    parser.add_argument("--query", help="search instead of indexing")
    parser.add_argument("--groups", default="", help="comma-separated groups for --query, e.g. hr,finance")
    parser.add_argument("-k", type=int, default=5, help="number of results for --query")
    args = parser.parse_args()

    embedder = GeminiEmbedder()
    collection = get_collection()

    if args.query:
        groups = [g.strip() for g in args.groups.split(",") if g.strip()]
        for rank, hit in enumerate(search(collection, embedder, args.query, groups, args.k), start=1):
            snippet = hit["text"][:200].replace("\n", " ")
            print(f"[{rank}] {hit['title']}, p. {hit['page']} (distance {hit['distance']:.3f})\n    {snippet}...")
        return

    stats = index_documents(collection, embedder)
    print(f"\nIndexed {len(stats.indexed)} document(s), {stats.chunks_added} chunks added.")
    print(f"Skipped {len(stats.skipped)} unchanged, removed {len(stats.removed)} deleted.")
    if stats.empty:
        print(f"No text found in: {', '.join(stats.empty)} (scanned PDFs need OCR)")
    print(f"Collection '{collection.name}' now holds {collection.count()} chunks in {CHROMA_DIR}")


if __name__ == "__main__":
    main()
