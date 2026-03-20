import argparse
import os
import re
import tempfile
import torch
import numpy as np
from typing import Generator, Tuple
import gradio as gr
from fastapi import FastAPI
from loguru import logger

# FastRTC Imports
from fastrtc import (
    AlgoOptions,
    ReplyOnPause,
    Stream,
    SileroVadOptions,
    audio_to_bytes,
    get_cloudflare_turn_credentials,
)

# Agent / LLM Imports
from company_support_agent import agent, agent_config, set_user_query
from faster_whisper import WhisperModel

# ==========================================
# 1. INITIALIZE MODELS (STT & TTS)
# ==========================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# A. Load Faster-Whisper (STT)
logger.info(f"⏳ Loading local STT model (medium on {DEVICE})...")
try:
    # Use float16 on GPU for speed, int8 on CPU
    compute_type = "float16" if DEVICE == "cuda" else "int8"
    local_stt_model = WhisperModel("medium", device=DEVICE, compute_type=compute_type)
    logger.info("✅ Local STT model loaded successfully.")
except Exception as e:
    logger.error(f"❌ Failed to load Faster-Whisper model: {e}")
    local_stt_model = None

# B. Load NVIDIA Magpie TTS via NeMo
try:
    from nemo.collections.tts.models import MagpieTTSModel

    logger.info("⏳ Loading Magpie TTS Model (nvidia/magpie_tts_multilingual_357m)...")
    magpie_model = MagpieTTSModel.from_pretrained("nvidia/magpie_tts_multilingual_357m")
    if DEVICE == "cuda":
        magpie_model = magpie_model.cuda()
    magpie_model.eval()
    logger.info("✅ Magpie TTS Model loaded successfully.")
except ImportError:
    logger.error(
        "❌ NeMo toolkit not found! Please install using instructions in docs/Nemo_setup.md"
    )
    magpie_model = None

# ==========================================
# 2. AUDIO GENERATION HELPERS
# ==========================================


def clean_text_for_tts(text: str) -> str:
    """Removes markdown and formatting characters before passing to TTS."""
    text = re.sub(r"\*\*|__", "", text)
    text = re.sub(r"\*", "", text)
    text = re.sub(r"#+", "", text)
    text = re.sub(r"^\s*[\-\*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    return text.strip()


def generate_magpie_audio(text: str, speaker_idx: int = 1) -> np.ndarray:
    """
    Passes a sentence to MagpieTTS, applying Text Normalization and forcing Hindi.
    Returns a 16-bit PCM numpy array at 22050Hz.
    Speakers: 0=John, 1=Sofia, 2=Aria, 3=Jason, 4=Leo.
    """
    if not magpie_model:
        return np.zeros(0, dtype=np.int16)

    with torch.no_grad():
        try:
            # Magpie do_tts outputs (audio_tensor, audio_length)
            # language="hi" ensures it reads Devanagari correctly.
            # apply_TN=True normalizes numbers/currencies into Hindi words.
            audio_tensor, _ = magpie_model.do_tts(
                text, language="hi", apply_TN=True, speaker_index=speaker_idx
            )

            # Convert PyTorch float32 tensor to NumPy array
            audio_np = audio_tensor.squeeze().cpu().numpy()

            # Scale float32 [-1.0, 1.0] to int16 PCM format for WebRTC
            audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
            return audio_int16

        except Exception as e:
            logger.error(f"TTS Generation Error for text '{text}': {e}")
            return np.zeros(0, dtype=np.int16)


# ==========================================
# 3. FASTRTC RESPONSE HANDLER
# ==========================================


def response(
    audio: tuple[int, np.ndarray], user_name: str = "", user_email: str = ""
) -> Generator[Tuple[int, np.ndarray], None, None]:
    logger.info("🎙️ Received audio input")

    # 1. Transcription
    audio_bytes = audio_to_bytes(audio)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        temp_path = tmp.name

    logger.debug("🔍 Transcribing audio locally...")
    segments, info = local_stt_model.transcribe(temp_path, beam_size=5, vad_filter=True)
    transcript = " ".join([segment.text for segment in segments]).strip()
    os.remove(temp_path)

    if not transcript:
        return

    logger.info(f'📝 Transcribed: "{transcript}"')

    # 2. Setup Context for LangGraph Agent
    # Enforce Hindi Devanagari output if Hindi is detected (redundant but safe)
    lang_instruction = (
        " (IMPORTANT: Reply in native Hindi Devanagari script)"
        if info.language == "hi"
        else ""
    )
    combined_input = f"{transcript}{lang_instruction}"

    if user_name or user_email:
        combined_input += f" (Context: Name: {user_name}, Email: {user_email})"

    set_user_query(transcript)

    # 3. Agent Streaming & TTS Chunking
    # MagpieTTS is a non-streaming model by default (processes full strings).
    # We split the LLM response into sentences and yield them dynamically to reduce Time-To-First-Audio.

    # Split pattern looks for Devanagari Purna Viram '।' or standard English punctuation '.?!'
    split_pattern = re.compile(r"(?<=[।.?!])\s+")
    current_buffer = ""
    full_response_text = ""

    try:
        logger.info("🧠 Agent Thinking (Streaming mode)...")
        stream = agent.stream(
            {"messages": [{"role": "user", "content": combined_input}]},
            config=agent_config,
            stream_mode="messages",
        )

        for event in stream:
            msg_chunk = event[0] if isinstance(event, tuple) else event

            if hasattr(msg_chunk, "content") and msg_chunk.content:
                content = msg_chunk.content
                if isinstance(content, str):
                    current_buffer += content
                    full_response_text += content

                    # If we hit a sentence boundary (punctuation followed by space)
                    if re.search(r"[।.?!]\s", current_buffer):
                        parts = split_pattern.split(current_buffer)

                        to_speak = []
                        if len(parts) > 1:
                            to_speak = parts[:-1]
                            current_buffer = parts[-1]
                        elif re.search(r"[।.?!]$", parts[0]):
                            to_speak = [parts[0]]
                            current_buffer = ""

                        # Generate audio for completed sentences
                        for sentence in to_speak:
                            clean_sent = clean_text_for_tts(sentence)
                            if clean_sent:
                                logger.debug(f"🗣️ TTS Generating (Sofia): {clean_sent}")
                                audio_chunk = generate_magpie_audio(
                                    clean_sent, speaker_idx=1
                                )
                                if len(audio_chunk) > 0:
                                    # Magpie outputs 22050Hz (via nano-codec-22khz)
                                    yield (22050, audio_chunk)

        # Process any remaining text left in the buffer
        if current_buffer.strip():
            clean_sent = clean_text_for_tts(current_buffer)
            if clean_sent:
                logger.debug(f"🗣️ TTS Generating (Sofia): {clean_sent}")
                audio_chunk = generate_magpie_audio(clean_sent, speaker_idx=1)
                if len(audio_chunk) > 0:
                    yield (22050, audio_chunk)

        logger.info(f"🧠 Full Response: {full_response_text[:50]}...")

    except Exception as e:
        logger.error(f"Streaming Agent Error: {e}")


# ==========================================
# 4. INITIALIZATION & UI
# ==========================================


def startup(*args) -> Generator[Tuple[int, np.ndarray], None, None]:
    """Greeting generated when WebRTC connection is established."""
    logger.info("🔊 Yielding silence to keep WebRTC alive...")
    yield (22050, np.zeros(22050, dtype=np.int16))

    greeting_text = "नमस्ते, मैं रेना हूँ, रेनाटा की सपोर्ट असिस्टेंट। आज मैं आपकी कैसे मदद कर सकती हूँ?"
    logger.info("🔊 Generating startup greeting with MagpieTTS...")
    audio_chunk = generate_magpie_audio(greeting_text, speaker_idx=1)
    if len(audio_chunk) > 0:
        yield (22050, audio_chunk)


def create_stream() -> Stream:
    return Stream(
        modality="audio",
        mode="send-receive",
        rtc_configuration=get_cloudflare_turn_credentials,
        handler=ReplyOnPause(
            response,
            algo_options=AlgoOptions(speech_threshold=0.6),
            model_options=SileroVadOptions(threshold=0.7),
            startup_fn=startup,
        ),
        additional_inputs=[
            gr.Textbox(label="Name", placeholder="Enter your name"),
            gr.Textbox(label="Email", placeholder="Enter your email"),
        ],
        ui_args={"title": "Renata Support Bot (NVIDIA Magpie TTS)"},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RenataAI Voice Agent with Magpie TTS")
    parser.add_argument("--phone", action="store_true")
    parser.add_argument("--fastphone", action="store_true")
    parser.add_argument("--remote", action="store_true")
    args = parser.parse_args()

    stream = create_stream()
    logger.info(
        "🎧 Stream handler configured with NVIDIA Magpie TTS (Native Devanagari)"
    )

    if args.remote:
        logger.info(
            "🌍 Launching REMOTE voice endpoint (Gradio Share + Cloudflare TURN)..."
        )
        stream.ui.launch(share=True)

    elif args.phone:
        logger.info("📞 Launching with phone interface...")
        from pyngrok import ngrok
        import uvicorn

        ngrok.set_auth_token(os.getenv("NGROK_AUTH_TOKEN"))
        public_url = ngrok.connect(8000, "http")
        logger.info(f"🌍 Public ngrok URL: {public_url}")

        app = FastAPI()
        stream.mount(app)
        uvicorn.run(app, host="127.0.0.1", port=8000)

    elif args.fastphone:
        stream.fastphone()

    else:
        logger.info("✔️ Launching custom Gradio UI...")
        stream.ui.launch()
