import json
import os
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from loguru import logger
from typing import Any, List

# ==============================
# Configuration
# ==============================
COMPANY_DATA_FILE = "src/avaada_presentation.json"
INDEX_FILE_PATH = "src/rag_integration/company_rag_index_2.bin"
CORPUS_FILE_PATH = "src/rag_integration/company_corpus_chunks_2.npy"

device = "cpu"
torch.set_default_device(device)


# ==============================
# Utilities
# ==============================
def load_company_data() -> dict:
    """Load structured company JSON data."""
    if not os.path.exists(COMPANY_DATA_FILE):
        raise FileNotFoundError(f"{COMPANY_DATA_FILE} not found")

    with open(COMPANY_DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_text_chunks(
    data: Any,
    path: str = "",
    min_length: int = 30
) -> List[str]:
    """
    Recursively walk JSON and extract meaningful text chunks.

    Rules:
    - Strings become chunks
    - Lists of strings are joined into readable sentences
    - Dict structure is preserved via path prefix
    """
    chunks = []

    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{path}.{key}" if path else key
            chunks.extend(extract_text_chunks(value, new_path, min_length))

    elif isinstance(data, list):
        if all(isinstance(item, str) for item in data):
            joined = ", ".join(data)
            text = f"{path.replace('.', ' ')}: {joined}"
            if len(text) >= min_length:
                chunks.append(text)
        else:
            for idx, item in enumerate(data):
                new_path = f"{path}[{idx}]"
                chunks.extend(extract_text_chunks(item, new_path, min_length))

    elif isinstance(data, str):
        text = f"{path.replace('.', ' ')}: {data}"
        if len(text) >= min_length:
            chunks.append(text)

    return chunks


def json_to_corpus(data: dict) -> List[str]:
    """Convert JSON directly into embedding-ready text chunks."""
    logger.info("🔍 Extracting text chunks directly from JSON...")
    chunks = extract_text_chunks(data)
    logger.info(f"✅ Extracted {len(chunks)} text chunks from JSON")
    return chunks


# ==============================
# FAISS Index Creation
# ==============================
def create_faiss_index(corpus: List[str]):
    logger.info("🧠 Loading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    logger.info("📐 Creating embeddings...")
    embeddings = embedder.encode(
        corpus,
        convert_to_tensor=False,
        show_progress_bar=True
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)
    dimension = embeddings.shape[1]

    logger.info("📦 Building FAISS index...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE_PATH)
    np.save(CORPUS_FILE_PATH, np.array(corpus, dtype=object))

    logger.success("🚀 FAISS index successfully created")
    logger.info(f"Index file: {INDEX_FILE_PATH}")
    logger.info(f"Corpus file: {CORPUS_FILE_PATH}")
    logger.info(f"Total chunks indexed: {len(corpus)}")


# ==============================
# Main
# ==============================
if __name__ == "__main__":
    logger.remove()
    logger.add(
        lambda msg: print(msg),
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
    )

    try:
        logger.info("⏳ Loading company JSON...")
        company_data = load_company_data()

        corpus = json_to_corpus(company_data)
        create_faiss_index(corpus)

    except Exception as e:
        logger.exception(f"❌ Failed to create RAG index: {e}")
