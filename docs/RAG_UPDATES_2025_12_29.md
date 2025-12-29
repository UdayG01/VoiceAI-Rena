# Voice Agent Remote Execution Support – December 29, 2025

**Author:** Uday Gupta  
**Commit ID:** `4c40ff65c3a133fbccec24f29227f39442333c89`  
**Description:** Add remote execution support using ngrok to test the voice agent on a GPU-enabled remote PC.

---

## Summary of Changes

Introduced a **Remote Execution Mode** for the FastRTC-based voice agent, enabling real-time voice interaction with a **locally hosted large voice model running on a remote GPU machine**. This allows developers to use their **local microphone and browser UI** while leveraging **remote GPU resources**, without deploying the application to cloud infrastructure.

The solution uses **FastAPI + FastRTC + Gradio**, exposed securely via an **ngrok tunnel**, while preserving the existing voice pipeline and agent logic.

---

## 1. Remote Launch Mode (`--remote`)

**File:** `src/test_native_fastrtc.py`

* Added a new CLI flag `--remote` using `argparse` with `action="store_true"`.
* Enables a dedicated launch path for running the voice agent on a remote machine.
* Designed specifically for testing large voice models that cannot be loaded locally.

---

## 2. FastAPI Integration for Remote Inference

**File:** `src/test_native_fastrtc.py`

* Mounted the existing FastRTC `Stream` onto a FastAPI application using:

  ```python
  stream.mount(app)
  ```
* This exposes FastRTC WebRTC and WebSocket endpoints required for real-time audio streaming.
* The FastAPI server is explicitly bound to:

  ```python
  host="0.0.0.0"
  ```

  which is required for ngrok to forward external traffic correctly.

---

## 3. Gradio UI Mounting on FastAPI

**File:** `src/test_native_fastrtc.py`

* Mounted the Gradio UI onto the same FastAPI application:

  ```python
  gr.mount_gradio_app(app, stream.ui, path="/")
  ```
* This exposes the Gradio interface at the root (`/`) URL while keeping FastRTC endpoints active.

**Important Implementation Detail:**

* The correct object to mount is `stream.ui` (a `gr.Blocks` instance).
* Attempting to mount `stream.ui.app` (the generated ASGI app) results in a runtime error because it lacks Gradio configuration methods such as `get_config_file()`.

---

## 4. ngrok Tunnel Integration

**File:** `src/test_native_fastrtc.py`

* Integrated `pyngrok` to expose the locally running FastAPI server via a secure HTTP tunnel.
* The public ngrok URL is generated dynamically and logged at startup for easy access.
* Enables remote access across different networks and NAT configurations without cloud deployment.

---

## 5. Preserved Voice Pipeline Architecture

* No changes were made to the core voice-processing logic, including:

  * Voice Activity Detection (VAD)
  * Pause-based response triggering
  * Faster-Whisper STT pipeline
  * Agent invocation and tool routing
  * Kokoro TTS with dynamic language selection

* The same pipeline now functions seamlessly in:

  * Local Gradio UI mode
  * Phone mode
  * Fast phone mode
  * **Remote GPU-backed mode**

---

## Files Modified

* `src/test_native_fastrtc.py`

  * Added `--remote` launch mode.
  * Integrated FastAPI and ngrok for remote execution.
  * Mounted Gradio UI and FastRTC stream on a single application.
  * Updated server binding for external accessibility.

---

## Resulting Capabilities

* 🎙️ Local microphone input via browser or phone
* 🧠 Large voice models running on a remote GPU machine
* 🔊 Real-time streamed audio responses
* 🌍 No cloud deployment required
* 🔐 Secure, temporary public access via ngrok

---

*This update enables scalable and realistic testing of GPU-intensive voice models while maintaining a clean local development workflow.*
