# Logging System Documentation

## Overview
The RAG Voice Agent now has a comprehensive logging system that captures all debugging information to timestamped markdown files in the `logs/` directory, while keeping the console clean.

## Log File Location
- **Directory**: `logs/`
- **Filename Format**: `rag_session_YYYYMMDD_HHMMSS.md`
- **Example**: `logs/rag_session_20251216_100230.md`

## What Gets Logged

### File Logs (Complete Detail)
All of the following are logged to the markdown file:

1. **Session Information**
   - Session start timestamp
   - Log file name

2. **RAG System Loading**
   - Component loading status
   - Index dimensions
   - Number of chunks loaded
   - Any errors during loading

3. **Tool Calls** (🔧 prefix)
   - Tool name and arguments
   - Example: `🔧 TOOL CALL: rag_search(query='Where is RenataAI based?', k=3)`

4. **Tool Results** (🔧 prefix)
   - Retrieved chunks count
   - Metadata sections used
   - Full chunk content (at DEBUG level)
   - Example: `🔧 TOOL RESULT: Retrieved 3 chunks from sections: ['organization_overview', 'customer_and_industry_presence']`

5. **Agent Interactions** (🧠 prefix)
   - User input (transcribed audio)
   - Agent response
   - Number of messages in conversation
   - Tool usage details

6. **Audio Processing**
   - Audio input received
   - Transcription results
   - TTS generation status

7. **Errors and Exceptions**
   - Full exception traces
   - Error context

### Console Output (Minimal)
Only these messages appear in console:
- `📝 Logs saved to: logs/rag_session_YYYYMMDD_HHMMSS.md` (on startup)
- Any messages explicitly marked with `logger.bind(console=True)`

## Log File Format

The log files are formatted as markdown for easy reading:

```markdown
# RAG Voice Agent Session Log
**Session Started:** 2025-12-16 10:00:00  
**Log File:** `rag_session_20251216_100000.md`

---

## Session Activity

**10:00:15** | `INFO` | ⏳ Loading RAG components...
**10:00:18** | `INFO` | ✅ RAG loaded | Chunks: 89 | Index dim: 384
**10:00:20** | `INFO` | 📝 Transcribed: "Where is RenataAI based?"
**10:00:20** | `INFO` | 🔧 TOOL CALL: rag_search(query='Where is RenataAI based?', k=3)
**10:00:20** | `INFO` | 🔧 TOOL RESULT: Retrieved 3 chunks from sections: ['organization_overview']
**10:00:21** | `DEBUG` | 🔧 Retrieved chunks: ['organization_overview company_name: RenataAI', ...]
**10:00:22** | `INFO` | 💬 Response (Cleaned): "RenataAI is based in Gurgaon, India."
```

## Usage

### Running the Voice Agent
```bash
python src/test_native_fastrtc.py
```

### Running Test Queries
```bash
python src/test_rag_logging.py
```

### Reading Logs
1. Navigate to `logs/` directory
2. Open the latest `.md` file
3. View in any markdown viewer or text editor

## Configuration

The logging configuration is in `src/company_support_agent.py`:

```python
# File logger (detailed)
logger.add(
    LOG_FILE,
    format="**{time:HH:mm:ss}** | `{level}` | {message}",
    level="DEBUG",  # Captures everything
    mode="w",
    encoding="utf-8"
)

# Console logger (minimal)
logger.add(
    lambda msg: print(msg),
    level="INFO",
    filter=lambda record: record["extra"].get("console", False)
)
```

## Debugging with Logs

When investigating issues:

1. **Check Tool Calls**: Look for `🔧 TOOL CALL` entries to see what tools were invoked
2. **Check Tool Results**: Look for `🔧 TOOL RESULT` to see what data was retrieved
3. **Check Retrieved Sections**: Verify the metadata sections to ensure correct context
4. **Check Agent Response**: Compare raw response vs cleaned response
5. **Check Errors**: Look for `ERROR` or `EXCEPTION` entries

## Example Log Analysis

If the agent gives incorrect information about company location:

1. Search for the query in the log
2. Find the corresponding `🔧 TOOL CALL: rag_search`
3. Check the `🔧 TOOL RESULT` to see what sections were retrieved
4. Look at the `Retrieved chunks` at DEBUG level to see the actual text
5. Identify if the wrong chunks were retrieved or if the chunking strategy needs improvement

## Benefits

✅ **Clean Console**: No clutter during voice interactions  
✅ **Complete History**: Every session is preserved  
✅ **Easy Debugging**: Markdown format is human-readable  
✅ **Tool Visibility**: See exactly what RAG is retrieving  
✅ **Timestamped**: Each log has precise timing information  
✅ **Version Control**: Logs are gitignored automatically
