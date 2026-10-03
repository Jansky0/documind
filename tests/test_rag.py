import tempfile
from pathlib import Path
import pytest
from src.document_loader import (
    recursive_split_text,
    load_and_chunk_document,
    DocumentChunk,
)
from src.vector_store import VectorStoreManager


def test_recursive_split_text():
    sample_text = (
        "Paragraf pertama menjelaskan tentang arsitektur RAG modern.\n\n"
        "Paragraf kedua membahas pentingnya chunking dan embedding vektor.\n\n"
        "Paragraf ketiga menguraikan integrasi dengan Google Gemini LLM."
    )
    # Split with small chunk size to force chunking
    chunks = recursive_split_text(sample_text, chunk_size=80, chunk_overlap=20)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) <= 120  # Respects boundary within reasonable separator margin
        assert len(chunk.strip()) > 0


def test_load_and_chunk_document(tmp_path: Path):
    doc_path = tmp_path / "test_doc.md"
    doc_path.write_text(
        "# Panduan Sistem\n\n"
        "DocuMind adalah platform knowledge base dokumen berbasis AI.\n\n"
        "Sistem ini menggunakan ChromaDB dan Gemini Flash.",
        encoding="utf-8",
    )

    chunks = load_and_chunk_document(doc_path, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 1
    first_chunk = chunks[0]
    assert isinstance(first_chunk, DocumentChunk)
    assert first_chunk.source == "test_doc.md"
    assert first_chunk.page == 1
    assert "DocuMind" in first_chunk.text


def test_vector_store_crud(tmp_path: Path):
    # Use temporary directory for ChromaDB
    chroma_dir = tmp_path / "chroma_test"
    store = VectorStoreManager(persist_dir=chroma_dir, use_gemini_embeddings=False)

    chunks = [
        DocumentChunk(
            chunk_id="doc1_p1_c1",
            text="DocuMind adalah sistem AI untuk tanya jawab dokumen lokal.",
            source="doc1.txt",
            page=1,
            metadata={"source": "doc1.txt", "page": 1},
        ),
        DocumentChunk(
            chunk_id="doc2_p1_c1",
            text="Python 3.12 dan uv digunakan untuk performa maksimal.",
            source="doc2.txt",
            page=1,
            metadata={"source": "doc2.txt", "page": 1},
        ),
    ]

    # Test Add
    added = store.add_chunks(chunks)
    assert added == 2

    # Test List Sources
    sources = store.list_sources()
    assert len(sources) == 2
    sources_names = [s["source"] for s in sources]
    assert "doc1.txt" in sources_names
    assert "doc2.txt" in sources_names

    # Test Search
    results = store.search("Apa itu DocuMind?", top_k=2)
    assert len(results) > 0
    assert results[0]["source"] == "doc1.txt"
    assert "DocuMind" in results[0]["text"]

    # Test Delete Source
    store.delete_source("doc1.txt")
    remaining_sources = store.list_sources()
    assert len(remaining_sources) == 1
    assert remaining_sources[0]["source"] == "doc2.txt"
