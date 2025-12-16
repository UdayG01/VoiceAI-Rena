# RAG System Updates - December 16, 2025

**Author:** Uday Gupta  
**Commit ID:** `4fcddb6916b2ab1b88d10f99d01aed4960868217`  
**Description:** reducing hallucination by editing system prompt, shifting to a larger model, add query validating functions, changing to 'medium' whisper for transcription

## Summary of Changes
Focused on reducing RAG hallucinations and improving query understanding. This involved a multi-layered approach: switching to a more capable LLM, implementing a query validation safety layer, enhancing the system prompt with strict grounding rules, improved contextual chunking, and upgrading the transcription model.

## 1. Hallucination Reduction & Model Upgrade
**File:** `src/company_support_agent.py`

- **Model Switch:**
  - Upgraded from `qwen3:1.7b` to `openai/gpt-oss-20b:free` (via OpenRouter) to improve reasoning and tool calling capabilities.
  - Reduced `max_tokens` (to 180) to encourage brevity and reduce the likelihood of unauthorized elaboration.
- **Strict Grounding Rules:**
  - Updated System Prompt with a dedicated **"STRICT GROUNDING RULES"** section.
  - Explicit instructions to ONLY state information explicitly found in tool results.
  - Added new few-shot examples demonstrating how to list items without inventing implementation details.

## 2. Query Validation Layer (Safeguard)
**File:** `src/company_support_agent.py`, `src/test_native_fastrtc.py`

- **Mechanism:**
  - Implemented `validate_rag_query` and `extract_keywords` functions.
  - Before searching, the system checks if the LLM's reformulated query has preserved critical keywords from the original user transcript.
- **Fallback Logic:**
  - If <50% of keywords are preserved, the system rejects the LLM's reformulation and falls back to the original user query.
  - **Goal:** Prevent the "intent loss" observed with smaller models (e.g., turning "solution verticals" into generic "About company" queries).

## 3. Contextual Chunking Strategy (v3)
**File:** `src/rag_integration/create_json_index_v3.py`

- **Hierarchical Context:**
  - Moved from atomic field chunking to **Contextual Chunking**.
  - Chunks now carry their parent context path (e.g., `reference_deployment_example > customer_origin`).
- **Context Prefixes:**
  - Added explicit string prefixes like `COMPANY INFO -`, `CUSTOMER PRESENCE -`, `SOLUTIONS -` to chunks.
  - **Goal:** Resolve ambiguity where "Japan" could be interpreted as company origin vs. customer origin.

## 4. Transcription & Logging Improvements
**File:** `src/test_native_fastrtc.py`, `src/company_support_agent.py`

- **Whisper Model Upgrade:**
  - Changed `WHISPER_MODEL_SIZE` from `'small'` to `'medium'` for higher accuracy transcription.
- **Comprehensive Logging:**
  - Implemented a dual-logging system:
    - **Console:** Minimal info to keep UI clean.
    - **Files (`logs/`):** Detailed markdown logs (`rag_session_*.md`) capturing every tool call, retrieved chunk, and raw agent response for debugging.

## Files Modified
- `src/company_support_agent.py`: Model config, system prompt, validation logic, logging.
- `src/rag_integration/create_json_index_v3.py`: New chunking strategy script.
- `src/test_native_fastrtc.py`: Whisper config, log integration, query capture.
- `.gitignore`: Added `logs/` directory.

---
*This document logs changes aimed at solving specific hallucination and context-loss issues observed in user testing.*
