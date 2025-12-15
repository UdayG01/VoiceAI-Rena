"""
VERSION: 15-12-2025 22:21pm

In this edit I am trying to attach metadata to each chunk extracted from the JSON file.
and then create a FAISS index that includes these metadata for better retrieval later on.

"""

import json
import os
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from loguru import logger
from typing import Any, List, Dict

# ==============================
# Configuration
# ==============================
COMPANY_DATA_FILE = "src/avaada_presentation.json"
INDEX_FILE_PATH = "src/rag_integration/company_rag_index_2.bin"
CORPUS_FILE_PATH = "src/rag_integration/company_corpus_chunks_2.npy"
METADATA_FILE_PATH = "src/rag_integration/company_corpus_metadata_2.npy"

device = "cpu"
torch.set_default_device(device)


# ==============================
# Load JSON
# ==============================
def load_company_data() -> dict:
    if not os.path.exists(COMPANY_DATA_FILE):
        raise FileNotFoundError(f"{COMPANY_DATA_FILE} not found")

    with open(COMPANY_DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ==============================
# JSON → Chunks with Metadata
# ==============================
def extract_chunks_with_metadata(
    data: Any,
    path: str = "",
    source: str = "",
    min_length: int = 30
) -> List[Dict]:
    """
    Recursively extract text chunks and attach metadata.
    """
    chunks = []

    def top_section(p: str) -> str:
        return p.split(".")[0] if "." in p else p

    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{path}.{key}" if path else key
            chunks.extend(
                extract_chunks_with_metadata(value, new_path, source, min_length)
            )

    elif isinstance(data, list):
        if all(isinstance(item, str) for item in data):
            joined = ", ".join(data)
            text = f"{path.replace('.', ' ')}: {joined}"
            if len(text) >= min_length:
                chunks.append({
                    "text": text,
                    "metadata": {
                        "json_path": path,
                        "top_section": top_section(path),
                        "category": path.split(".")[-1],
                        "chunk_type": "string_list",
                        "source": source
                    }
                })
        else:
            for idx, item in enumerate(data):
                new_path = f"{path}[{idx}]"
                chunks.extend(
                    extract_chunks_with_metadata(item, new_path, source, min_length)
                )

    elif isinstance(data, (str, int, float, bool)):
        text = f"{path.replace('.', ' ')}: {data}"
        if len(text) >= min_length:
            chunks.append({
                "text": text,
                "metadata": {
                    "json_path": path,
                    "top_section": top_section(path),
                    "category": path.split(".")[-1],
                    "chunk_type": type(data).__name__,
                    "source": source
                }
            })


    return chunks


def json_to_chunks(data: dict) -> List[Dict]:
    logger.info("🔍 Extracting chunks with metadata from JSON...")
    chunks = extract_chunks_with_metadata(
        data,
        source=os.path.basename(COMPANY_DATA_FILE)
    )
    logger.info(f"✅ Extracted {len(chunks)} chunks")
    return chunks


# ==============================
# FAISS Index Creation
# ==============================
def create_faiss_index(chunks: List[Dict]):
    texts = [c["text"] for c in chunks]
    metadata = [c["metadata"] for c in chunks]

    logger.info("🧠 Loading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    logger.info("📐 Creating embeddings...")
    embeddings = embedder.encode(
        texts,
        convert_to_tensor=False,
        show_progress_bar=True
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)
    dim = embeddings.shape[1]

    logger.info("📦 Building FAISS index...")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Persist everything
    faiss.write_index(index, INDEX_FILE_PATH)
    np.save(CORPUS_FILE_PATH, np.array(texts, dtype=object))
    np.save(METADATA_FILE_PATH, np.array(metadata, dtype=object))

    logger.success("🚀 RAG index created successfully")
    logger.info(f"Index: {INDEX_FILE_PATH}")
    logger.info(f"Corpus: {CORPUS_FILE_PATH}")
    logger.info(f"Metadata: {METADATA_FILE_PATH}")
    logger.info(f"Total chunks indexed: {len(texts)}")


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
        logger.info("⏳ Loading JSON data...")
        company_data = load_company_data()

        chunks = json_to_chunks(company_data)
        create_faiss_index(chunks)

    except Exception as e:
        logger.exception(f"❌ Failed to create RAG index: {e}")
