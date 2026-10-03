from typing import List, Dict, Any, Generator, Optional
from google import genai
from google.genai import types
from src.config import settings
from src.vector_store import VectorStoreManager


RAG_SYSTEM_PROMPT = """You are DocuMind, an intelligent AI research assistant designed for document analysis and knowledge base retrieval.
Your task is to answer user queries accurately, concisely, and factually based ONLY on the provided context documents.

GUIDELINES:
1. Base your answer STRICTLY on the [CONTEXT DOCUMENTS]. Do NOT invent facts or hallucinate.
2. If the context documents DO NOT contain sufficient information to answer the question, politely state: "I'm sorry, but that information is not available in the uploaded documents."
3. Include inline citations or bullet citations where relevant using the format: `[Source: <filename> | Page <page_number>]`.
4. Format your answer using clear Markdown (bullet points, bold text, code blocks when suitable).
5. Always respond in the same language as the user's query (English by default, or Indonesian/other languages if asked in that language).
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
                f"[Context #{i} | Source: {source} | Page {page}]\n{text}"
            )

        context_str = "\n\n".join(context_blocks)

        prompt = f"""[CONTEXT DOCUMENTS]:
{context_str if context_str else "(No matching documents found)"}

[USER QUESTION]:
{question}

Please provide a comprehensive answer with accurate source citations according to the guidelines."""
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
            yield "⚠️ **Warning**: Gemini API Key is not configured. Please enter your Google Gemini API Key in the sidebar or define it in `.env`."
            return

        if not self.last_retrieved_sources:
            yield "No indexed documents found yet. Please upload and index PDF, Markdown, or TXT documents in the sidebar first!"
            return

        # Step 2: Build Grounded Prompt
        prompt = self.build_prompt(question, self.last_retrieved_sources)

        # Step 3: Stream from Gemini LLM
        try:
            config = types.GenerateContentConfig(
                system_instruction=RAG_SYSTEM_PROMPT,
                temperature=0.2,  # Low temperature for grounded facts
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
            yield f"\n\n❌ **Error calling Gemini API**: {str(e)}"

    def query(self, question: str, top_k: int = 4) -> Dict[str, Any]:
        """Non-streaming query returning full answer text and sources."""
        full_text = ""
        for token in self.query_stream(question, top_k=top_k):
            full_text += token

        return {
            "answer": full_text,
            "sources": self.last_retrieved_sources,
        }
