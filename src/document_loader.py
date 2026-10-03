from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
import re
from pypdf import PdfReader


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source: str
    page: int
    metadata: dict = field(default_factory=dict)


def recursive_split_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    separators: List[str] | None = None
) -> List[str]:
    """
    Splits text recursively by trying separators in priority order:
    paragraphs ("\n\n"), lines ("\n"), sentence ends (". "), spaces (" "), and characters.
    Ensures chunks do not exceed chunk_size while maintaining chunk_overlap.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    if not text.strip():
        return []

    if len(text) <= chunk_size:
        return [text.strip()]

    # Choose the highest priority separator present in the text
    separator = separators[-1]
    new_separators = []
    for i, sep in enumerate(separators):
        if sep == "":
            separator = ""
            new_separators = []
            break
        if sep in text:
            separator = sep
            new_separators = separators[i + 1:]
            break

    splits = text.split(separator) if separator != "" else list(text)

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for split in splits:
        split_len = len(split) + (len(separator) if current_chunk else 0)

        if current_len + split_len > chunk_size and current_chunk:
            combined = separator.join(current_chunk).strip()
            if combined:
                chunks.append(combined)

            # Rebuild with overlap
            overlap_acc: List[str] = []
            overlap_len = 0
            for item in reversed(current_chunk):
                item_len = len(item) + (len(separator) if overlap_acc else 0)
                if overlap_len + item_len <= chunk_overlap:
                    overlap_acc.insert(0, item)
                    overlap_len += item_len
                else:
                    break

            current_chunk = overlap_acc
            current_len = overlap_len

        # If a single split item is larger than chunk_size, recurse with finer separators
        if len(split) > chunk_size and new_separators:
            sub_chunks = recursive_split_text(split, chunk_size, chunk_overlap, new_separators)
            for sub in sub_chunks:
                if sub.strip():
                    chunks.append(sub.strip())
            current_chunk = []
            current_len = 0
        else:
            current_chunk.append(split)
            current_len += len(split) + (len(separator) if len(current_chunk) > 1 else 0)

    if current_chunk:
        combined = separator.join(current_chunk).strip()
        if combined:
            chunks.append(combined)

    return [c for c in chunks if c.strip()]


def extract_text_from_pdf(pdf_path: Path) -> List[Tuple[int, str]]:
    """
    Extracts text page by page from a PDF file.
    Returns list of (page_number, text).
    """
    reader = PdfReader(str(pdf_path))
    pages_text: List[Tuple[int, str]] = []

    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        cleaned = re.sub(r"[ \t]+", " ", text).strip()
        if cleaned:
            pages_text.append((idx + 1, cleaned))

    return pages_text


def extract_text_from_txt_or_md(file_path: Path) -> List[Tuple[int, str]]:
    """
    Extracts text from markdown or text file.
    Returns (1, text) since text/md typically doesn't have page numbers.
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    cleaned = re.sub(r"[ \t]+", " ", text).strip()
    return [(1, cleaned)] if cleaned else []


def load_and_chunk_document(
    file_path: Path,
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[DocumentChunk]:
    """
    Loads a document (PDF, Markdown, TXT) and splits it into structured chunks
    with citation metadata.
    """
    suffix = file_path.suffix.lower()
    source_name = file_path.name

    if suffix == ".pdf":
        pages = extract_text_from_pdf(file_path)
    elif suffix in [".md", ".markdown", ".txt"]:
        pages = extract_text_from_txt_or_md(file_path)
    else:
        raise ValueError(f"Format file tidak didukung: {suffix}. Gunakan .pdf, .md, atau .txt")

    chunks: List[DocumentChunk] = []
    chunk_counter = 0

    for page_num, page_text in pages:
        sub_chunks = recursive_split_text(page_text, chunk_size, chunk_overlap)
        for sub_chunk in sub_chunks:
            chunk_counter += 1
            chunk_id = f"{source_name}_p{page_num}_c{chunk_counter}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=sub_chunk,
                    source=source_name,
                    page=page_num,
                    metadata={
                        "source": source_name,
                        "page": page_num,
                        "chunk_index": chunk_counter,
                        "char_count": len(sub_chunk)
                    }
                )
            )

    return chunks
