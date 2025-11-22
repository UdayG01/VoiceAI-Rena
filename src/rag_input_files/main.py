import argparse
from typing import Generator, Tuple

import numpy as np
import os
from fastrtc import (
    AlgoOptions,
    ReplyOnPause,
    Stream,
    audio_to_bytes,
)
from groq import Groq
from loguru import logger

FFMPEG_BIN_PATH = r"C:\Users\user\Downloads\ffmpeg-2025-11-17-git-e94439e49b-essentials_build\ffmpeg-2025-11-17-git-e94439e49b-essentials_build\bin"
os.environ["PATH"] += os.pathsep + FFMPEG_BIN_PATH

from process_groq_tts import process_groq_tts
#from simple_math_agent import agent, agent_config
from company_support_agent import agent, agent_config

logger.remove()
logger.add(
    lambda msg: print(msg),
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
)

groq_client = Groq()


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
    logger.info(f'👂 Transcribed: "{transcript}"')

    logger.debug("🧠 Running agent...")
    agent_response = agent.invoke(
        {"messages": [{"role": "user", "content": transcript}]}, config=agent_config
    )
    response_text = agent_response["messages"][-1].content
    logger.info(f'💬 Response: "{response_text}"')

    logger.debug("🔊 Generating speech...")
    tts_response = groq_client.audio.speech.create(
        model="playai-tts",
        voice="Cheyenne-PlayAI",
        response_format="wav",
        input=response_text,
    )
    yield from process_groq_tts(tts_response)


def create_stream() -> Stream:
    """
    Create and configure a Stream instance with audio capabilities.

    Returns:
        Stream: Configured FastRTC Stream instance
    """
    return Stream(
        modality="audio",
        mode="send-receive",
        handler=ReplyOnPause(
            response,
            algo_options=AlgoOptions(
                speech_threshold=0.5,
            ),
        ),
    )

import gradio as gr
from fastrtc import WebRTC, ReplyOnPause


import gradio as gr
from fastrtc import WebRTC, ReplyOnPause

def build_custom_ui(stream: Stream):
    with gr.Blocks(theme="soft") as ui:

        gr.Markdown(
            """
            <h1 style="text-align:center; font-size:36px;">
            🤖 Renata Support Bot
            </h1>
            <p style="text-align:center; font-size:18px; margin-top:-10px;">
            Speak freely — the assistant will respond when you pause.
            </p>
            """
        )

        # WebRTC audio stream (MIC)
        audio_stream = WebRTC(
            label="🎤 Tap to Speak",
            mode="send-receive",
            modality="audio",
        )

        # Bind your response generator
        audio_stream.stream(
            fn=ReplyOnPause(response),
            inputs=[audio_stream],
            outputs=[audio_stream],
            time_limit=500
        )

    # Override default UI
    stream.ui = ui



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="FastRTC Groq Voice Agent")
#     parser.add_argument(
#         "--phone",
#         action="store_true",
#         help="Launch with FastRTC phone interface (get a temp phone number)",
#     )
#     args = parser.parse_args()

#     stream = create_stream()
#     logger.info("🎧 Stream handler configured")

#     if args.phone:
#         logger.info("🌈 Launching with FastRTC phone interface...")
#         stream.fastphone()
#     else:
#         logger.info("🌈 Launching with Gradio UI...")
#         stream.ui.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastRTC Groq Voice Agent")
    parser.add_argument("--phone", action="store_true")
    args = parser.parse_args()

    stream = create_stream()
    logger.info("🎧 Stream handler configured")

    if args.phone:
        logger.info("📞 Launching with phone interface...")
        stream.fastphone()
    else:
        logger.info("🌈 Launching custom Gradio UI...")
        build_custom_ui(stream)
        stream.ui.launch()