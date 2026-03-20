# RAG Path Compliance Refactoring - 2026-03-20

## Overview
This session focused on refactoring hardcoded path variables across the codebase to ensure compatibility with the updated project directory structure. 

## Key Changes
The following architectural standard was enforced:
- **RAG Artifacts**: Vector indices (`.bin`), corpus chunks (`.npy`), and metadata (`.npy`) are now exclusively accessed from and written to `src/rag_integration/rag_data/`.
- **Structured Data**: JSON configuration and company data files are now managed within `src/json/`.
- **Transcripts**: Temporary and source knowledge base text files are now located in `src/Audio_transcripts/`.

## File-by-File Breakdown

### 1. `src/company_support_agent.py`
- Updated `INDEX_FILE_PATH`, `CORPUS_FILE_PATH`, and `METADATA_FILE_PATH` to point to `src/rag_integration/rag_data/`.
- Updated `load_company_data` to point to `src/json/renata_data.json`.

### 2. `src/company_support_agent2.py`
- Updated `RAG_INTEGRATION_DIR` to point to `src/rag_integration/rag_data/`.
- Updated `DATA_FILE_PATH` to point to `src/json/renata_data.json`.

### 3. `src/rag_integration/src/create_index.py`
- Updated `COMPANY_DATA_FILE` to point to `src/json/renata_data.json`.
- Updated `INDEX_FILE_PATH` and `CORPUS_FILE_PATH` to point to `src/rag_integration/rag_data/`.
- Updated `KNOWLEDGE_BASE_TEXT_FILE` to point to `src/Audio_transcripts/knowledge_base.txt`.

### 4. `src/rag_integration/src/create_json_index_v3.py`
- Updated `COMPANY_DATA_FILE` to point to `src/json/renata_data2.json`.
- Updated `INDEX_FILE_PATH`, `CORPUS_FILE_PATH`, and `METADATA_FILE_PATH` to point to `src/rag_integration/rag_data/`.

### 5. `src/rag_integration/src/create_json_index_v2.py`
- Updated `COMPANY_DATA_FILE` to point to `src/json/renata_data2.json`.
- Updated `INDEX_FILE_PATH`, `CORPUS_FILE_PATH`, and `METADATA_FILE_PATH` to point to `src/rag_integration/rag_data/`.

### 6. `src/rag_integration/src/create_json_index.py`
- Updated `COMPANY_DATA_FILE` to point to `src/json/avaada_presentation.json`.
- Updated `INDEX_FILE_PATH`, `CORPUS_FILE_PATH`, and `METADATA_FILE_PATH` to point to `src/rag_integration/rag_data/`.

### 7. `src/rag_integration/src/create_redis_index.py`
- Updated `DATA_FILE` to point to `src/json/renata_data.json`.

---
*Documentation generated automatically on 2026-03-20*
