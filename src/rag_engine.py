from typing import List, Dict, Any, Generator, Optional
from google import genai
from google.genai import types
from src.config import settings
from src.vector_store import VectorStoreManager


RAG_SYSTEM_PROMPT = """Kamu adalah DocuMind, asisten AI cerdas untuk analisis dokumen dan knowledge base.
Tugasmu adalah menjawab pertanyaan pengguna secara akurat, terstruktur, dan berbasis fakta HANYA dari dokumen konteks yang diberikan.

PANDUAN MENJAWAB:
1. Dasarkan seluruh jawabanmu HANYA pada [DOKUMEN KONTEKS]. Jangan membuat asumsi atau mengarang fakta (hindari halusinasi).
2. Jika dokumen konteks TIDAK memuat informasi yang cukup untuk menjawab pertanyaan, nyatakan dengan jujur dan sopan: "Maaf, informasi mengenai hal tersebut tidak ditemukan dalam dokumen yang diunggah."
3. Cantumkan sitasi atau referensi sumber di dalam teks atau di akhir poin dengan format: `[Sumber: <nama_file> | Hal. <nomor_halaman>]`.
4. Format jawaban dengan markdown yang rapi (gunakan bullet points, bold, atau numbering jika diperlukan).
5. Gunakan bahasa yang sama dengan pertanyaan pengguna (Bahasa Indonesia atau Bahasa Inggris).
"""


class RAGEngine:
    """Retrieval-Augmented Generation Engine powered by Google Gemini and ChromaDB."""

    def __init__(
        self,
        vector_store: VectorStoreManager,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.vector_store = vector_store
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = model_name or settings.gemini_model
        self.last_retrieved_sources: List[Dict[str, Any]] = []

        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    def build_prompt(
        self, question: str, retrieved_chunks: List[Dict[str, Any]]
    ) -> str:
        """Constructs the prompt combining system instructions, retrieved context, and the question."""
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source = chunk.get("source", "Unknown")
            page = chunk.get("page", 1)
            text = chunk.get("text", "")
            context_blocks.append(
                f"[Konteks #{i} | Sumber: {source} | Hal. {page}]\n{text}"
            )

        context_str = "\n\n".join(context_blocks)

        prompt = f"""[DOKUMEN KONTEKS]:
{context_str if context_str else "(Tidak ada dokumen yang relevan ditemukan)"}

[PERTANYAAN PENGGUNA]:
{question}

Silakan berikan jawaban yang komprehensif dengan sitasi sumber sesuai panduan."""
        return prompt

    def query_stream(
        self, question: str, top_k: int = 4
    ) -> Generator[str, None, None]:
        """
        Retrieves relevant document chunks and streams the Gemini response chunk-by-chunk.
        Retrieved sources are stored in `self.last_retrieved_sources`.
        """
        # Step 1: Semantic Retrieval
        self.last_retrieved_sources = self.vector_store.search(
            query=question, top_k=top_k
        )

        if not self.client:
            yield "⚠️ **Peringatan**: API Key Gemini belum disetel. Silakan masukkan Google Gemini API Key Anda di panel sidebar atau buat file `.env`."
            return

        if not self.last_retrieved_sources:
            yield "Belum ada dokumen yang diindeks. Silakan unggah dokumen PDF/Markdown/TXT terlebih dahulu di sidebar!"
            return

        # Step 2: Build Grounded Prompt
        prompt = self.build_prompt(question, self.last_retrieved_sources)

        # Step 3: Stream from Gemini LLM
        try:
            config = types.GenerateContentConfig(
                system_instruction=RAG_SYSTEM_PROMPT,
                temperature=0.2,  # Low temperature for factual grounding
            )
            response_stream = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=prompt,
                config=config,
            )

            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            yield f"\n\n❌ **Terjadi kesalahan saat memanggil Gemini API**: {str(e)}"

    def query(self, question: str, top_k: int = 4) -> Dict[str, Any]:
        """Non-streaming query returning full answer text and sources."""
        full_text = ""
        for token in self.query_stream(question, top_k=top_k):
            full_text += token

        return {
            "answer": full_text,
            "sources": self.last_retrieved_sources,
        }
