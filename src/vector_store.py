from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from google import genai
from src.config import settings
from src.document_loader import DocumentChunk


class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    """Custom Chroma embedding function using Google GenAI text-embedding-004."""

    def __init__(self, api_key: str, model_name: str = "text-embedding-004"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        embeddings: List[List[float]] = []
        # Process in batches of 32 to stay well within API limits
        batch_size = 32
        for i in range(0, len(input), batch_size):
            batch = input[i : i + batch_size]
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=batch,
            )
            for item in response.embeddings:
                embeddings.append(item.values)
        return embeddings


class VectorStoreManager:
    """Manages persistent document embeddings and semantic search with ChromaDB."""

    COLLECTION_NAME = "documind_docs"

    def __init__(
        self,
        persist_dir: Path | None = None,
        api_key: Optional[str] = None,
        use_gemini_embeddings: bool = False,
    ):
        self.persist_dir = persist_dir or settings.chroma_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

        self.api_key = api_key or settings.gemini_api_key
        self.use_gemini_embeddings = use_gemini_embeddings and bool(self.api_key)

        # Select embedding function
        if self.use_gemini_embeddings:
            self.embedding_fn = GeminiEmbeddingFunction(
                api_key=self.api_key, model_name=settings.embedding_model
            )
        else:
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Adds a list of DocumentChunk to the vector database."""
        if not chunks:
            return 0

        # Remove existing chunks for the same source to avoid duplication on re-upload
        sources_to_replace = list({c.source for c in chunks})
        for src in sources_to_replace:
            self.delete_source(src)

        batch_size = 100
        total_added = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c.chunk_id for c in batch]
            documents = [c.text for c in batch]
            metadatas = [
                {
                    "source": c.source,
                    "page": c.page,
                    "chunk_id": c.chunk_id,
                    **c.metadata,
                }
                for c in batch
            ]

            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            total_added += len(batch)

        return total_added

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Searches for most relevant chunks matching the query.
        Returns list of dicts with text, source, page, chunk_id, and similarity score.
        """
        count = self.collection.count()
        if count == 0:
            return []

        actual_k = min(top_k, count)
        results = self.collection.query(
            query_texts=[query],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"],
        )

        matched_chunks: List[Dict[str, Any]] = []

        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else []
            dists = results["distances"][0] if results.get("distances") else []

            for doc, meta, dist in zip(docs, metas, dists):
                # Cosine distance: 0 is identical, 2 is opposite.
                # Similarity score roughly = 1 - (dist / 2) or 1 - dist
                score = max(0.0, 1.0 - float(dist)) if dist is not None else 1.0

                matched_chunks.append(
                    {
                        "text": doc,
                        "source": meta.get("source", "Unknown"),
                        "page": int(meta.get("page", 1)),
                        "chunk_id": meta.get("chunk_id", ""),
                        "score": round(score, 4),
                        "metadata": meta,
                    }
                )

        return matched_chunks

    def list_sources(self) -> List[Dict[str, Any]]:
        """Returns statistics of all unique sources indexed in the vector store."""
        count = self.collection.count()
        if count == 0:
            return []

        data = self.collection.get(include=["metadatas"])
        metadatas = data.get("metadatas") or []

        sources_map: Dict[str, Dict[str, Any]] = {}
        for m in metadatas:
            if not m:
                continue
            src = m.get("source", "Unknown")
            page = int(m.get("page", 1))

            if src not in sources_map:
                sources_map[src] = {
                    "source": src,
                    "chunk_count": 0,
                    "pages": set(),
                }
            sources_map[src]["chunk_count"] += 1
            sources_map[src]["pages"].add(page)

        summary = []
        for src, info in sources_map.items():
            summary.append(
                {
                    "source": src,
                    "chunks": info["chunk_count"],
                    "page_count": len(info["pages"]),
                }
            )

        return sorted(summary, key=lambda x: x["source"])

    def delete_source(self, source_name: str) -> None:
        """Deletes all chunks belonging to a specific file."""
        self.collection.delete(where={"source": source_name})

    def clear_all(self) -> None:
        """Deletes all records from the collection."""
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
