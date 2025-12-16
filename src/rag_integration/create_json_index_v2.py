"""
Enhanced JSON Chunking Strategy for RAG - Version 3.0
DATE: 16-12-2025

This version implements contextual chunking with:
1. Hierarchical context preservation
2. Explicit context type markers (COMPANY INFO, CUSTOMER INFO, etc.)
3. Section summaries + detailed chunks
4. Enhanced metadata with context_type
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
INDEX_FILE_PATH = "src/rag_integration/company_rag_index_3.bin"
CORPUS_FILE_PATH = "src/rag_integration/company_corpus_chunks_3.npy"
METADATA_FILE_PATH = "src/rag_integration/company_corpus_metadata_3.npy"

device = "cpu"
torch.set_default_device(device)


# ==============================
# Context Classification
# ==============================
def classify_context_type(json_path: str) -> str:
    """Classify the context type based on JSON path."""
    path_lower = json_path.lower()
    
    if any(x in path_lower for x in ["organization_overview", "founder", "company_name", "location", "team_size"]):
        return "company_info"
    elif any(x in path_lower for x in ["customer", "industry_presence", "international", "reference_deployment"]):
        return "customer_info"
    elif any(x in path_lower for x in ["solution_domains", "capabilities", "production", "quality", "automation"]):
        return "solution_info"
    elif any(x in path_lower for x in ["organizational_structure", "roles", "support"]):
        return "organizational_info"
    else:
        return "general_info"


def get_context_prefix(context_type: str) -> str:
    """Get human-readable prefix for context type."""
    prefixes = {
        "company_info": "COMPANY INFO",
        "customer_info": "CUSTOMER PRESENCE",
        "solution_info": "SOLUTIONS",
        "organizational_info": "ORGANIZATION",
        "general_info": "INFO"
    }
    return prefixes.get(context_type, "INFO")


# ==============================
# Load JSON
# ==============================
def load_company_data() -> dict:
    if not os.path.exists(COMPANY_DATA_FILE):
        raise FileNotFoundError(f"{COMPANY_DATA_FILE} not found")
    
    with open(COMPANY_DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ==============================
# Enhanced Chunking with Context
# ==============================
def create_section_summary(section_name: str, section_data: Any, context_type: str) -> Dict:
    """Create a summary chunk for a major section."""
    prefix = get_context_prefix(context_type)
    
    # Extract key information
    summary_parts = [f"{prefix} - "]
    
    if isinstance(section_data, dict):
        # Get first few key-value pairs for summary
        items = []
        for k, v in list(section_data.items())[:5]:  # First 5 items
            if isinstance(v, (str, int, float, bool)):
                items.append(f"{k.replace('_', ' ')}: {v}")
            elif isinstance(v, list) and all(isinstance(x, str) for x in v):
                items.append(f"{k.replace('_', ' ')}: {', '.join(v[:3])}")  # First 3 items
        
        summary_parts.append(". ".join(items))
    
    text = "".join(summary_parts)
    
    return {
        "text": text,
        "metadata": {
            "json_path": section_name,
            "top_section": section_name,
            "context_type": context_type,
            "chunk_type": "section_summary",
            "source": os.path.basename(COMPANY_DATA_FILE)
        }
    }


def extract_contextual_chunks(
    data: Any,
    path: str = "",
    parent_context: str = "",
    source: str = "",
    min_length: int = 40
) -> List[Dict]:
    """
    Recursively extract chunks with full context preservation.
    Each chunk includes parent context to avoid confusion.
    """
    chunks = []
    
    def get_top_section(p: str) -> str:
        return p.split(".")[0] if "." in p else p
    
    if isinstance(data, dict):
        # For major sections, create a summary chunk first
        if not parent_context and path:  # Top-level section
            context_type = classify_context_type(path)
            summary_chunk = create_section_summary(path, data, context_type)
            if len(summary_chunk["text"]) >= min_length:
                chunks.append(summary_chunk)
        
        # Process child elements
        for key, value in data.items():
            new_path = f"{path}.{key}" if path else key
            new_parent = f"{parent_context} > {key.replace('_', ' ')}" if parent_context else key.replace('_', ' ')
            
            chunks.extend(
                extract_contextual_chunks(value, new_path, new_parent, source, min_length)
            )
    
    elif isinstance(data, list):
        if all(isinstance(item, str) for item in data):
            # String list - create contextual chunk
            context_type = classify_context_type(path)
            prefix = get_context_prefix(context_type)
            
            joined = ", ".join(data)
            text = f"{prefix} - {parent_context}: {joined}"
            
            if len(text) >= min_length:
                chunks.append({
                    "text": text,
                    "metadata": {
                        "json_path": path,
                        "top_section": get_top_section(path),
                        "context_type": context_type,
                        "category": path.split(".")[-1],
                        "chunk_type": "string_list",
                        "parent_context": parent_context,
                        "source": source
                    }
                })
        else:
            # Mixed list - process each item
            for idx, item in enumerate(data):
                new_path = f"{path}[{idx}]"
                chunks.extend(
                    extract_contextual_chunks(item, new_path, parent_context, source, min_length)
                )
    
    elif isinstance(data, (str, int, float, bool)):
        # Leaf value - create contextual chunk
        context_type = classify_context_type(path)
        prefix = get_context_prefix(context_type)
        
        text = f"{prefix} - {parent_context}: {data}"
        
        if len(text) >= min_length:
            chunks.append({
                "text": text,
                "metadata": {
                    "json_path": path,
                    "top_section": get_top_section(path),
                    "context_type": context_type,
                    "category": path.split(".")[-1],
                    "chunk_type": type(data).__name__,
                    "parent_context": parent_context,
                    "source": source
                }
            })
    
    return chunks


def json_to_chunks(data: dict) -> List[Dict]:
    logger.info("🔍 Extracting contextual chunks from JSON...")
    chunks = extract_contextual_chunks(
        data,
        source=os.path.basename(COMPANY_DATA_FILE)
    )
    logger.info(f"✅ Extracted {len(chunks)} contextual chunks")
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
    
    logger.success("🚀 Enhanced RAG index created successfully")
    logger.info(f"Index: {INDEX_FILE_PATH}")
    logger.info(f"Corpus: {CORPUS_FILE_PATH}")
    logger.info(f"Metadata: {METADATA_FILE_PATH}")
    logger.info(f"Total chunks indexed: {len(texts)}")
    
    # Print sample chunks for verification
    logger.info("\n📋 Sample chunks:")
    for i, (text, meta) in enumerate(zip(texts[:5], metadata[:5])):
        logger.info(f"\nChunk {i+1}:")
        logger.info(f"  Type: {meta['context_type']}")
        logger.info(f"  Text: {text[:150]}...")


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
