import argparse
from typing import Generator, Tuple
import numpy as np
import os
from contextlib import asynccontextmanager

# 1. Import STT and TTS model loaders
from fastrtc import (
    AlgoOptions,
    ReplyOnPause,
    Stream,
    get_tts_model,
    # get_stt_model, # Commented out as we are using faster_whisper
    KokoroTTSOptions,
    SileroVadOptions,
    WebRTC,
    audio_to_bytes
)
from fastapi import FastAPI

import gradio as gr
from loguru import logger

# Keep agent imports for the LLM part
from company_support_agent import agent, agent_config

# from process_groq_tts import process_groq_tts 

# Note: Logger is configured in company_support_agent.py
# We import the configured logger, so no need to reconfigure here

# 2. Initialize the Local Models (Global scope to load weights once)
# Moonshine is the default STT model in FastRTC
#stt_model = get_stt_model() 

"""
START LOADING CTRANSLATE2 (FASTER-WHISPER)
"""

# Use compute_type='int8' for 4x speedup on CPU over vanilla Whisper.
from faster_whisper import WhisperModel
WHISPER_MODEL_SIZE = "medium"
WHISPER_DEVICE = "cpu"
try:
    logger.info(f"⏳ Loading local STT model ({WHISPER_MODEL_SIZE} on {WHISPER_DEVICE})...")
    local_stt_model = WhisperModel(
        WHISPER_MODEL_SIZE, 
        device=WHISPER_DEVICE, 
        compute_type="int8"
    )
    logger.info(" Local STT model loaded successfully.")
except Exception as e:
    logger.error(f" Failed to load Faster-Whisper model: {e}")
    # Consider keeping groq_client call as a fallback here, but for now, we'll proceed with local model.

"""
END LOADING CTRANSLATE2
"""
# Kokoro is the default TTS model
tts_model = get_tts_model() 

# Configure TTS Voice/Speed
options = KokoroTTSOptions(
    voice="hf_alpha", # specific to Kokoro
    speed=1.0,
    lang="en-us"
)

def response(
    audio: tuple[int, np.ndarray],
    user_name: str = "",
    user_email: str = ""
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Process audio input locally, generate response via Groq/LangGraph, and deliver local TTS.
    """
    logger.info("🎙️ Received audio input")
    if user_name or user_email:
        logger.info(f"⌨️ Received context - Name: {user_name}, Email: {user_email}")

    # Convert to bytes
    audio_bytes = audio_to_bytes(audio)

    # Create temporary mp3 file
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        temp_path = tmp.name

    # Faster-whisper transcription
    logger.debug("🔍 Transcribing audio locally with Faster-Whisper...")
    segments, info = local_stt_model.transcribe(
        temp_path,
        beam_size=5,
        vad_filter=True,
    )
    
    detected_lang = info.language
    logger.info(f"🌐 Detected language: {detected_lang} (Prob: {info.language_probability:.2f})")

    # Build transcript
    transcript = " ".join([segment.text for segment in segments]).strip()

    # Delete temp file
    os.remove(temp_path)
    
    # Handle case where transcription is empty or just noise (prevents LLM error)
    if not transcript:
        transcript = " " # Use a space or a default phrase

    #  END: CTRANSLATE2 TEST
    
    logger.info(f'📝 Transcribed: "{transcript}"')
    
    # Force language instruction if Hindi is detected to guide the smaller LLM
    lang_instruction = ""
    if detected_lang == "hi":
        lang_instruction = " (IMPORTANT: Reply in Hindi)"
    
    combined_input = f"{transcript}{lang_instruction}"

    if user_name or user_email:
        context_str = f" (Context: Name: {user_name}, Email: {user_email})"
        combined_input = f"{combined_input}{context_str}"

    # 4. LLM Processing (Still using your Groq-based Agent)
    logger.debug("🧠 Running agent...")
    logger.debug(f"🧠 Agent Input: {combined_input}")
    
    # Store original query for validation 
    from company_support_agent import set_user_query
    set_user_query(transcript)  
    
    agent_response = agent.invoke(
        {"messages": [{"role": "user", "content": combined_input}]}, config=agent_config
    )
    
    # Log agent response details
    logger.debug(f"🧠 Agent returned {len(agent_response['messages'])} messages")
    
    # Extract and log tool usage
    for i, msg in enumerate(agent_response['messages']):
        if hasattr(msg, 'type'):
            if msg.type == 'ai' and hasattr(msg, 'tool_calls') and msg.tool_calls:
                logger.info(f"🤖 Agent made {len(msg.tool_calls)} tool call(s)")
                for tc in msg.tool_calls:
                    logger.debug(f"   - Tool: {tc.get('name', 'unknown')}, Args: {tc.get('args', {})}")
            elif msg.type == 'tool':
                logger.debug(f"   - Tool response: {msg.content[:200]}...")  # First 200 chars
    
    response_text = agent_response["messages"][-1].content
    logger.debug(f"🧠 Raw Agent Response: {response_text}")
    
    # Clean text for TTS (remove markdown, special chars)
    import re
    def clean_text_for_tts(text: str) -> str:
        # Remove bold/italic markers (** or *)
        text = re.sub(r'\*\*|__', '', text)
        text = re.sub(r'\*', '', text)
        # Remove headers (#)
        text = re.sub(r'#+', '', text)
        # Remove list markers (1., -, etc.) but keep the content
        text = re.sub(r'^\s*[\-\*]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        return text.strip()

    response_text = clean_text_for_tts(response_text)
    logger.info(f'💬 Response (Cleaned): "{response_text}"')

    # 5. Local TTS
    # stream_tts_sync yields audio chunks suitable for the stream
    # 5. Local TTS with Dynamic Language
    # Use 'hi' for Hindi, default to 'en-us' for others
    tts_lang = "hi" if detected_lang == "hi" else "en-us"
    
    dynamic_options = KokoroTTSOptions(
        voice=options.voice,
        speed=options.speed,
        lang=tts_lang
    )

    logger.debug(f"🔊 Generating speech locally (Lang: {tts_lang})...")
    for audio_chunk in tts_model.stream_tts_sync(response_text, options=dynamic_options):
        yield audio_chunk

def startup(*args):
    for chunk in tts_model.stream_tts_sync("Hi!, I'm Rena, Renata's support assistant. How can I help you today?", options=options):
        yield chunk

def create_stream() -> Stream:
    """
    Create and configure a Stream instance with audio capabilities.
    """
    return Stream(
        modality="audio",
        mode="send-receive",
        handler=ReplyOnPause(
            response,
            algo_options=AlgoOptions(
                speech_threshold=0.6,
            ),
            model_options=SileroVadOptions(
                threshold=0.7
            ),
            startup_fn=startup
        ),
        additional_inputs=[
            gr.Textbox(label="Name", placeholder="Enter your name"),
            gr.Textbox(label="Email", placeholder="Enter your email")
        ],
        ui_args={"title": "Renata Support Bot"}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RenataAI Voice Agent")
    parser.add_argument("--phone", action="store_true")
    parser.add_argument("--fastphone", action="store_true")
    parser.add_argument("--remote", action="store_true")

    args = parser.parse_args()

    stream = create_stream()
    logger.info("🎧 Stream handler configured")

    if args.remote:
        logger.info("🌍 Launching REMOTE voice endpoint (ngrok + FastAPI)...")

        from pyngrok import ngrok
        from fastapi import FastAPI
        import uvicorn

        # Ensure token is set
        ngrok.set_auth_token(os.getenv("NGROK_AUTH_TOKEN"))

        # Create app
        app = FastAPI()
        stream.mount(app)

        app = gr.mount_gradio_app(
            app,
            stream.ui,
            path="/"
        )

        # Start tunnel FIRST
        public_url = ngrok.connect(8000, "http")
        logger.success(f"🔗 Public Voice Endpoint: {public_url}")

        # bind to 0.0.0.0
        uvicorn.run(app, host="0.0.0.0", port=8000)
        
    elif args.phone:
        logger.info("📞 Launching with phone interface...")

        # Start ngrok tunnel
        from pyngrok import ngrok
        ngrok.set_auth_token(os.getenv("NGROK_AUTH_TOKEN"))
        public_url = ngrok.connect(8000, "http")
        logger.info(f"🌍 Public ngrok URL: {public_url}")

        import uvicorn
        app = FastAPI()
        stream.mount(app)
        # Fix: Removed duplicate uvicorn.run call and redundant reload/workers args for simple execution
        uvicorn.run(app, host="127.0.0.1", port=8000)
    elif args.fastphone:
        stream.fastphone()
    else:
        logger.info("✔️ Launching custom Gradio UI...")
        stream.ui.launch()
