import argparse
from typing import Generator, Tuple

import numpy as np
import os
from fastrtc import (
    AlgoOptions,
    ReplyOnPause,
    Stream,
    audio_to_bytes,
    get_tts_model,
    KokoroTTSOptions,
    SileroVadOptions
)

from fastapi import FastAPI, WebSocket, Request

from groq import Groq
from loguru import logger
from openai import audio

from process_groq_tts import process_groq_tts
from company_support_agent import agent, agent_config

logger.remove()
logger.add(
    lambda msg: print(msg),
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
)

groq_client = Groq()

# Use a pipeline as a high-level helper
# from transformers import pipeline

# stt_model = pipeline("automatic-speech-recognition", model="openai/whisper-small")

tts_model = get_tts_model(model="kokoro")

options = KokoroTTSOptions(
    voice="af_heart",
    speed=1.0,
    lang="en-us"
)
def response(
    audio: tuple[int, np.ndarray],
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Process audio input, transcribe it, generate a response using LangGraph, and deliver TTS audio.

    Args:
        audio: Tuple containing sample rate and audio data

    Yields:
        Tuples of (sample_rate, audio_array) for audio playback
    """
    logger.info("🎙️ Received audio input")

    logger.debug("🔄 Transcribing audio...")
    transcript = groq_client.audio.transcriptions.create(
        file=("audio-file.mp3", audio_to_bytes(audio)),
        model="whisper-large-v3-turbo",
        response_format="text",
    )
    

    #TRANSFORMERS ATTEMPT
    # Extract the NumPy array of audio data
    # audio_data = audio[1] 
    
    # result = stt_model(audio_data)
    
    # # The pipeline output is usually a dict: {'text': 'The transcribed text.'}
    # transcript = result["text"].strip()
    
    logger.info(f'👂 Transcribed: "{transcript}"')

    logger.debug("🧠 Running agent...")
    agent_response = agent.invoke(
        {"messages": [{"role": "user", "content": transcript}]}, config=agent_config
    )
    response_text = agent_response["messages"][-1].content
    logger.info(f'💬 Response: "{response_text}"')

    logger.debug("🔊 Generating speech...")
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
        # @app.api_route("/incoming-call", methods=["GET", "POST"])
        # async def handle_incoming_call(request: Request):
        #     """Handle incoming call and return TwiML response to connect to Media Stream."""
        #     response = await stream.handle_incoming_call(request)
        #     return response
        # @app.websocket("incoming-call/telephone/handler")
        # async def handle_media_stream(websocket: WebSocket):
        #     """Handle WebSocket connections between Twilio and Media Stream."""
        #     await stream.telephone_handler(websocket)
        
        uvicorn.run(app, host="127.0.0.1", port=8000, ssl_keyfile=None, ssl_certfile=None)
        #uvicorn.run(app, host="127.0.0.1", port=8000, ssl_keyfile=None, ssl_certfile=None, reload=True, workers=1)
    elif args.fastphone:
        stream.fastphone()
    else:
        logger.info("✔️ Launching custom Gradio UI...")
        stream.ui.launch()