# RAG System Updates - December 15, 2025

**Author:** Uday Gupta  
**Commit ID:** `700e125512e02ca64fc69b08c4cf6f5a8cd66f02`  
**Description:** Add BM25 hybrid search and metadata-based search for RAG improvement.

## Summary of Changes
Significantly improved the Retrieval-Augmented Generation (RAG) pipeline by implementing a **Hybrid Search** mechanism that combines semantic vector search (FAISS) with keyword-based search (BM25). Additionally, introduced **Metadata Filtering** to diversify search results across different sections of the knowledge base.

## 1. Hybrid Search Implementation
**File:** `src/company_support_agent.py`

- **BM25 Search Added:**
  - Integrated `rank_bm25.BM25Okapi` to perform keyword-based retrieval.
  - The corpus is tokenized and indexed alongside the FAISS vector index during system initialization.
- **Hybrid Scoring:**
  - The search results are now a weighted combination of Vector Similarity and Keyword Matching.
  - **Formula:** `Final Score = (0.7 * Vector Score) + (0.3 * BM25 Score)`
  - This helps retrieve documents that might have low semantic similarity but exact keyword matches (e.g., specific acronyms or terminology).

## 2. Metadata Extraction & Indexing
**File:** `src/rag_integration/create_json_index.py`

- **Metadata Extraction:**
  - The indexing script now parses the source JSON (`src/avaada_presentation.json`) recursively.
  - For every text chunk, it extracts metadata including:
    - `json_path`: Full path to the data in the JSON structure.
    - `top_section`: The root key of the JSON object (e.g., `organization_overview`, `solution_domains`), providing a high-level context.
    - `category`: The immediate parent key.
    - `chunk_type`: Type of data (e.g., string list, string).
- **Persistence:**
  - Metadata is saved to a separate `.npy` file (`src/rag_integration/company_corpus_metadata_2.npy`) alongside the FAISS index and corpus.

## 3. Search Result Diversification
**File:** `src/company_support_agent.py`

- **Diversification Logic:**
  - The search tool (`rag_search`) now retrieves a larger candidate pool (up to 6 items) initially.
  - It then filters these candidates to ensure diversity based on the `top_section` metadata field.
  - **Goal:** Prevent the RAG system from returning multiple chunks from the exact same section if they are redundant, ensuring a broader context for the answer.

## files Modified
- `src/company_support_agent.py`: Integrated BM25 and hybrid search logic.
- `src/rag_integration/create_json_index.py`: Updated indexing logic to capture and save metadata.
- `src/avaada_presentation.json`: Source data (structure utilized for metadata).

---
*This document logs the changes introduced to improve RAG accuracy and context diversity.*
