# Path Compliance Refactoring & NVIDIA Magpie TTS Integration - 2026-03-20

## Overview
This session focused on two major tasks:
1. **Path Compliance Refactoring**: Standardizing hardcoded path variables across the codebase to ensure compatibility with the updated project directory structure.
2. **Magpie TTS Integration**: Adding support for NVIDIA's NeMo-based Magpie TTS (`nvidia/magpie_tts_multilingual_357m`) to generate high-quality Hinglish (Devanagari) voice responses.

## Part 1: Path Compliance Refactoring
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

## Magpie TTS Integration (NVIDIA NeMo)
In addition to the path compliance refactoring, a new integration for NVIDIA's Magpie TTS (`nvidia/magpie_tts_multilingual_357m`) was developed to support robust Hinglish (Devanagari) voice generation.

### 1. `src/nemo_magpie_tts.py` (New File)
- Implemented a complete FastRTC handler that orchestrates Faster-Whisper (STT), the LangGraph `company_support_agent` (LLM), and Magpie TTS.
- Introduced a sentence-level chunking stream to reduce Time-To-First-Audio (TTFA).
- Configured the NeMo `do_tts` pipeline with `language="hi"`, `apply_TN=True` (Text Normalization), and `speaker_index=1` (Sofia).

### 2. `docs/Nemo_setup.md` (New File)
- Created a comprehensive guide outlining setup procedures for NeMo on Windows using WSL2 and Docker.
- Documented a streamlined deployment workflow leveraging the `uv` package manager (`uv venv --system-site-packages` and `uv sync`).

### 3. `pyproject.toml`
- Added new dependencies specifically required by the Magpie TTS integration:
  - `"kaldialign>=0.9.0"`
  - `"nemo_toolkit[tts] @ git+https://github.com/NVIDIA/NeMo.git"`

---
*Documentation generated automatically on 2026-03-20*
