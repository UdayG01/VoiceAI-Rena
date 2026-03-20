# Changes on 2026-03-20: Path compliance refactoring.
# Purpose: Align code with new directory structure (RAG assets in 'rag_data/', JSON in 'json/').
# Documentation: See 'docs/RAG_UPDATES_2026_03_20.md' for details.

""" "
Enhanced JSON Chunking Strategy for RAG - Version 3.1
DATE: 29-01-2026

Updates from v3.0:
1. Implemented "Smart List Traversal" using semantic keys (id, name, client) instead of indices.
2. Added classifiers for 'knowledge_repository', 'case_studies', and 'cross_cutting'.
3. Refined list-of-strings handling for data evidence arrays.
"""

import json
import os
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from loguru import logger
from typing import Any, List, Dict, Optional

# ==============================
# Configuration
# ==============================
COMPANY_DATA_FILE = "src/json/renata_data2.json"
INDEX_FILE_PATH = "src/rag_integration/rag_data/company_rag_index_3.bin"
CORPUS_FILE_PATH = "src/rag_integration/rag_data/company_corpus_chunks_3.npy"
METADATA_FILE_PATH = "src/rag_integration/rag_data/company_corpus_metadata_3.npy"

device = "cpu"
torch.set_default_device(device)


# ==============================
# Context Classification
# ==============================
def classify_context_type(json_path: str) -> str:
    """Classify the context type based on JSON path keywords."""
    path_lower = json_path.lower()

    if "knowledge_repository" in path_lower or "compliance" in path_lower:
        return "compliance_info"
    elif "case_studies" in path_lower:
        return "case_study"
    elif "cross_cutting" in path_lower or "analytics" in path_lower:
        return "technology_stack"
    elif any(
        x in path_lower
        for x in ["organization", "founder", "company_name", "location", "team"]
    ):
        return "company_info"
    elif any(
        x in path_lower for x in ["market_presence", "customer", "industry", "clients"]
    ):
        return "market_info"
    elif any(
        x in path_lower for x in ["solution", "capabilities", "production", "quality"]
    ):
        return "solution_info"
    else:
        return "general_info"


def get_context_prefix(context_type: str) -> str:
    """Get human-readable prefix for context type."""
    prefixes = {
        "compliance_info": "COMPLIANCE FRAMEWORK",
        "case_study": "CASE STUDY",
        "technology_stack": "CORE TECH",
        "company_info": "COMPANY INFO",
        "market_info": "MARKET PRESENCE",
        "solution_info": "SOLUTION OFFERING",
        "general_info": "INFO",
    }
    return prefixes.get(context_type, "INFO")


def get_semantic_label(data: Dict) -> Optional[str]:
    """
    Look for a meaningful label in a dictionary to use instead of an array index.
    Prioritizes specific keys that likely contain identity info.
    """
    # Priority list of keys to identify an object
    identity_keys = ["name", "id", "client", "type", "component", "domain"]

    for key in identity_keys:
        if key in data and isinstance(data[key], str):
            return data[key]
    return None


# ==============================
# Load JSON
# ==============================
def load_company_data() -> dict:
    if not os.path.exists(COMPANY_DATA_FILE):
        # Create dummy file if not exists for testing logic
        logger.warning(
            f"File {COMPANY_DATA_FILE} not found. Please ensure the file exists."
        )
        return {}

    with open(COMPANY_DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ==============================
# Enhanced Chunking Logic
# ==============================
def create_section_summary(
    section_name: str, section_data: Any, context_type: str
) -> Dict:
    """Create a high-level summary chunk for major sections."""
    prefix = get_context_prefix(context_type)

    summary_parts = [f"{prefix} - Section: {section_name.replace('_', ' ').title()}"]

    if isinstance(section_data, dict):
        items = []
        # Extract first 5 string/numeric values to give a preview
        for k, v in list(section_data.items())[:5]:
            if isinstance(v, (str, int, float, bool)):
                items.append(f"{k.replace('_', ' ')}: {v}")
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], str):
                items.append(f"{k.replace('_', ' ')}: {', '.join(v[:3])}...")

        if items:
            summary_parts.append(" | Contains: " + "; ".join(items))

    text = " ".join(summary_parts)

    return {
        "text": text,
        "metadata": {
            "json_path": section_name,
            "top_section": section_name,
            "context_type": context_type,
            "chunk_type": "section_summary",
            "source": os.path.basename(COMPANY_DATA_FILE),
        },
    }


def extract_contextual_chunks(
    data: Any,
    path: str = "",
    parent_context: str = "",
    source: str = "",
    min_length: int = 20,  # Lowered min_length to catch specific data points
) -> List[Dict]:
    """
    Recursively extract chunks.
    CRITICAL CHANGE: Uses 'get_semantic_label' to name list items dynamically.
    """
    chunks = []

    def get_top_section(p: str) -> str:
        return p.split(".")[0] if "." in p else p

    # 1. Handle Dictionaries
    if isinstance(data, dict):
        # Create summary for top-level keys
        if not parent_context and path:
            context_type = classify_context_type(path)
            chunks.append(create_section_summary(path, data, context_type))

        for key, value in data.items():
            new_path = f"{path}.{key}" if path else key
            # Clean key for context text (e.g., "data_evidence_required" -> "Data Evidence Required")
            clean_key = key.replace("_", " ").title()

            # Append to parent context path
            if parent_context:
                new_parent = f"{parent_context} > {clean_key}"
            else:
                new_parent = clean_key

            chunks.extend(
                extract_contextual_chunks(
                    value, new_path, new_parent, source, min_length
                )
            )

    # 2. Handle Lists
    elif isinstance(data, list):
        if not data:
            return []

        # Case A: List of Strings (e.g., ["Invoice", "Logbook"])
        if all(isinstance(item, str) for item in data):
            context_type = classify_context_type(path)
            prefix = get_context_prefix(context_type)

            # Join all items into one comprehensive block
            joined_items = ", ".join(data)
            text = f"{prefix} - {parent_context}: {joined_items}"

            chunks.append(
                {
                    "text": text,
                    "metadata": {
                        "json_path": path,
                        "top_section": get_top_section(path),
                        "context_type": context_type,
                        "chunk_type": "string_list",
                        "parent_context": parent_context,
                        "source": source,
                    },
                }
            )

        # Case B: List of Objects (e.g., Emission Scopes, Case Studies)
        else:
            for idx, item in enumerate(data):
                # Smart Labeling: Try to find a name for this item
                semantic_label = None
                if isinstance(item, dict):
                    semantic_label = get_semantic_label(item)

                # Determine context string for this item
                if semantic_label:
                    item_context_str = semantic_label
                else:
                    item_context_str = f"Item {idx + 1}"

                new_parent = f"{parent_context} > {item_context_str}"

                # We keep index in path for technical uniqueness, but use label in context for semantics
                new_path = f"{path}[{idx}]"

                chunks.extend(
                    extract_contextual_chunks(
                        item, new_path, new_parent, source, min_length
                    )
                )

    # 3. Handle Leaf Values (Strings, Ints, Booleans)
    elif isinstance(data, (str, int, float, bool)):
        context_type = classify_context_type(path)
        prefix = get_context_prefix(context_type)

        # Format: "COMPLIANCE FRAMEWORK - Direct Emissions > Stationary Combustion > Data Evidence Required: Purchase invoices"
        text = f"{prefix} - {parent_context}: {data}"

        if len(str(text)) >= min_length:
            chunks.append(
                {
                    "text": text,
                    "metadata": {
                        "json_path": path,
                        "top_section": get_top_section(path),
                        "context_type": context_type,
                        "chunk_type": type(data).__name__,
                        "parent_context": parent_context,
                        "source": source,
                    },
                }
            )

    return chunks


def json_to_chunks(data: dict) -> List[Dict]:
    logger.info("🔍 Extracting contextual chunks with semantic labeling...")
    chunks = extract_contextual_chunks(data, source=os.path.basename(COMPANY_DATA_FILE))
    logger.info(f"✅ Extracted {len(chunks)} contextual chunks")
    return chunks


# ==============================
# FAISS Index Creation
# ==============================
def create_faiss_index(chunks: List[Dict]):
    if not chunks:
        logger.error("No chunks to index!")
        return

    texts = [c["text"] for c in chunks]
    metadata = [c["metadata"] for c in chunks]

    logger.info("🧠 Loading embedding model...")
    # Using a slightly stronger model for better semantic matching if available, else standard
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    logger.info("📐 Creating embeddings...")
    embeddings = embedder.encode(texts, convert_to_tensor=False, show_progress_bar=True)

    embeddings = np.asarray(embeddings, dtype=np.float32)
    dim = embeddings.shape[1]

    logger.info("📦 Building FAISS index...")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Ensure directory exists
    os.makedirs(os.path.dirname(INDEX_FILE_PATH), exist_ok=True)

    # Persist everything
    faiss.write_index(index, INDEX_FILE_PATH)
    np.save(CORPUS_FILE_PATH, np.array(texts, dtype=object))
    np.save(METADATA_FILE_PATH, np.array(metadata, dtype=object))

    logger.success("🚀 Enhanced RAG index 3.1 created successfully")

    # Print semantic verification
    logger.info("\n📋 Verification of Semantic Breadcrumbs:")
    for i, text in enumerate(texts):
        if "Scope 1" in text or "FCC Clutch" in text:
            logger.info(f"Chunk {i}: {text[:150]}...")


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

        if company_data:
            chunks = json_to_chunks(company_data)
            create_faiss_index(chunks)
        else:
            logger.error("Failed to load data.")

    except Exception as e:
        logger.exception(f"❌ Failed to create RAG index: {e}")
