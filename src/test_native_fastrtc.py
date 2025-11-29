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

logger.remove()
logger.add(
    lambda msg: print(msg),
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
)

# 2. Initialize the Local Models (Global scope to load weights once)
# Moonshine is the default STT model in FastRTC
#stt_model = get_stt_model() 

"""
START LOADING CTRANSLATE2 (FASTER-WHISPER)
"""
# Use 'base' or 'small' for best performance/accuracy balance on CPU.
# Use compute_type='int8' for 4x speedup on CPU over vanilla Whisper.
from faster_whisper import WhisperModel
WHISPER_MODEL_SIZE = "small"
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
    voice="af_heart", # specific to Kokoro
    speed=1.0,
    lang="en-us"
)

def response(
    audio: tuple[int, np.ndarray],
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Process audio input locally, generate response via Groq/LangGraph, and deliver local TTS.
    """
    logger.info("🎙️ Received audio input")

    # Convert to bytes
    audio_bytes = audio_to_bytes(audio)

    # Create temporary mp3 file
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        temp_path = tmp.name

    logger.debug(f"📁 Temp audio saved at: {temp_path}")

    # Faster-whisper transcription
    logger.debug("🔍 Transcribing audio locally with Faster-Whisper...")
    segments, _ = local_stt_model.transcribe(
        temp_path,
        language="en",
        beam_size=5,
        vad_filter=True,
    )

    # Build transcript
    transcript = " ".join([segment.text for segment in segments]).strip()

    # Delete temp file
    os.remove(temp_path)
    logger.debug("🧹 Temp file removed.")
    
    # Handle case where transcription is empty or just noise (prevents LLM error)
    if not transcript:
        transcript = " " # Use a space or a default phrase

    #  END: CTRANSLATE2 TEST
    
    logger.info(f' Transcribed: "{transcript}"')

    # 4. LLM Processing (Still using your Groq-based Agent)
    logger.debug("🧠 Running agent...")
    agent_response = agent.invoke(
        {"messages": [{"role": "user", "content": transcript}]}, config=agent_config
    )
    response_text = agent_response["messages"][-1].content
    logger.info(f'💬 Response: "{response_text}"')

    # 5. Local TTS
    # stream_tts_sync yields audio chunks suitable for the stream
    logger.debug("🔊 Generating speech locally...")
    for audio_chunk in tts_model.stream_tts_sync(response_text, options=options):
        yield audio_chunk

def startup():
    for chunk in tts_model.stream_tts_sync("Hi!, I'm Rena, Renata's support assistant. How can I help you today?"):
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
        ui_args={"title": "Renata Support Bot"}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RenataAI Voice Agent")
    parser.add_argument("--phone", action="store_true")
    parser.add_argument("--fastphone", action="store_true")
    args = parser.parse_args()

    stream = create_stream()
    logger.info("🎧 Stream handler configured")

    if args.phone:
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
