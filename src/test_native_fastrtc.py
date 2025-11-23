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
    get_stt_model,
    KokoroTTSOptions,
    SileroVadOptions,
    WebRTC
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
stt_model = get_stt_model() 

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

    # 3. Local STT (No need to convert to bytes)
    # fastrtc STT model accepts the (sample_rate, numpy_array) tuple directly
    logger.debug("🔄 Transcribing audio locally...")
    transcript = stt_model.stt(audio)
    
    logger.info(f'👂 Transcribed: "{transcript}"')

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
        uvicorn.run(app, host="127.0.0.1", port=8000, ssl_keyfile=None, ssl_certfile=None)
        #uvicorn.run(app, host="127.0.0.1", port=8000, ssl_keyfile=None, ssl_certfile=None, reload=True, workers=1)
    else:
        logger.info("✔️ Launching custom Gradio UI...")
        stream.ui.launch()