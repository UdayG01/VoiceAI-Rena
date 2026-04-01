import argparse
import os
import re
import tempfile
from typing import Generator, Tuple

import numpy as np
import torch
from fastapi import FastAPI
from loguru import logger

from fastrtc import (
    AlgoOptions,
    ReplyOnPause,
    Stream,
    SileroVadOptions,
    WebRTC,
    audio_to_bytes,
    get_cloudflare_turn_credentials,
)
from faster_whisper import WhisperModel

from company_support_agent import (
    agent,
    agent_config,
    ensure_devanagari_sentence,
    set_user_query,
)

# ==========================================
# 1. INITIALIZE MODELS (STT & TTS)
# ==========================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# A. Load Faster-Whisper (STT)
logger.info(f"Loading local STT model (medium on {DEVICE})...")
try:
    compute_type = "float16" if DEVICE == "cuda" else "int8"
    local_stt_model = WhisperModel("medium", device=DEVICE, compute_type=compute_type)
    logger.info("Local STT model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Faster-Whisper model on {DEVICE}: {e}")
    local_stt_model = None
    if DEVICE == "cuda":
        try:
            logger.warning("Retrying Faster-Whisper on CPU with compute_type=int8...")
            local_stt_model = WhisperModel(
                "medium", device="cpu", compute_type="int8"
            )
            logger.info("Local STT model loaded successfully on CPU fallback.")
        except Exception as fallback_error:
            logger.error(
                f"Failed to load Faster-Whisper model on CPU fallback: {fallback_error}"
            )

# B. Load NVIDIA Magpie TTS via NeMo
try:
    from nemo.collections.tts.models import MagpieTTSModel

    logger.info("Loading Magpie TTS Model (nvidia/magpie_tts_multilingual_357m)...")
    magpie_model = MagpieTTSModel.from_pretrained("nvidia/magpie_tts_multilingual_357m")
    if DEVICE == "cuda":
        magpie_model = magpie_model.cuda()
    magpie_model.eval()
    logger.info("Magpie TTS Model loaded successfully.")
except ImportError:
    logger.error(
        "NeMo toolkit not found! Please install using instructions in docs/Nemo_setup.md"
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


def contains_devanagari(text: str) -> bool:
    """Return True when the text includes Devanagari characters."""
    return any("\u0900" <= char <= "\u097F" for char in text)


def extract_complete_sentences(text_buffer: str) -> tuple[list[str], str]:
    """
    Split buffered text into completed sentences while keeping the unfinished tail.
    Sentence boundaries are Hindi purnviram and standard English punctuation.
    """
    parts = re.split(r"(?<=[।.?!])", text_buffer)
    completed: list[str] = []
    remainder = ""

    for part in parts:
        chunk = part.strip()
        if not chunk:
            continue
        if re.search(r"[।.?!]$", chunk):
            completed.append(chunk)
        else:
            remainder = chunk if not remainder else f"{remainder} {chunk}"

    return completed, remainder


def generate_magpie_audio(text: str, speaker_idx: int = 1) -> np.ndarray:
    """
    Passes a sentence to MagpieTTS, applying text normalization and forcing Hindi.
    Returns a 16-bit PCM numpy array at 22050Hz.
    Speakers: 0=John, 1=Sofia, 2=Aria, 3=Jason, 4=Leo.
    """
    if not magpie_model:
        return np.zeros(0, dtype=np.int16)

    with torch.no_grad():
        try:
            audio_tensor, _ = magpie_model.do_tts(
                text, language="hi", apply_TN=True, speaker_index=speaker_idx
            )
            audio_np = audio_tensor.squeeze().cpu().numpy()
            audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
            return audio_int16
        except Exception as e:
            logger.error(f"TTS generation error for text '{text}': {e}")
            return np.zeros(0, dtype=np.int16)


# ==========================================
# 3. FASTRTC RESPONSE HANDLER
# ==========================================


def response(
    audio: tuple[int, np.ndarray] | None, text_input: str = ""
) -> Generator[Tuple[int, np.ndarray], None, None]:
    transcript = text_input.strip()
    detected_language = "hi" if transcript and contains_devanagari(transcript) else "en"

    if transcript:
        logger.info("Received text input")
    else:
        if audio is None:
            logger.warning("No audio or text input received.")
            return
        if local_stt_model is None:
            logger.error("STT model is unavailable, cannot process audio input.")
            return

        logger.info("Received audio input")
        audio_bytes = audio_to_bytes(audio)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(audio_bytes)
                temp_path = tmp.name

            logger.debug("Transcribing audio locally...")
            segments, info = local_stt_model.transcribe(
                temp_path, beam_size=5, vad_filter=True
            )
            transcript = " ".join(segment.text for segment in segments).strip()
            detected_language = info.language
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    if not transcript:
        return

    logger.info(f'Transcribed: "{transcript}"')

    lang_instruction = (
        " (IMPORTANT: Reply in native Hindi Devanagari script)"
        if detected_language == "hi"
        else ""
    )
    combined_input = f"{transcript}{lang_instruction}"

    set_user_query(transcript)

    current_buffer = ""
    full_response_text = ""
    emitted_audio = False

    try:
        logger.info("Agent thinking (streaming mode)...")
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

                    completed_sentences, current_buffer = extract_complete_sentences(
                        current_buffer
                    )

                    for sentence in completed_sentences:
                        rewritten_sentence = ensure_devanagari_sentence(sentence)
                        clean_sent = clean_text_for_tts(rewritten_sentence)
                        if clean_sent:
                            logger.debug(f"TTS generating (Sofia): {clean_sent}")
                            audio_chunk = generate_magpie_audio(
                                clean_sent, speaker_idx=1
                            )
                            if len(audio_chunk) > 0:
                                emitted_audio = True
                                yield (22050, audio_chunk)

        if current_buffer.strip():
            rewritten_sentence = ensure_devanagari_sentence(current_buffer)
            clean_sent = clean_text_for_tts(rewritten_sentence)
            if clean_sent:
                logger.debug(f"TTS generating (Sofia): {clean_sent}")
                audio_chunk = generate_magpie_audio(clean_sent, speaker_idx=1)
                if len(audio_chunk) > 0:
                    emitted_audio = True
                    yield (22050, audio_chunk)

        logger.info(f"Full response: {full_response_text[:50]}...")

    except Exception as e:
        logger.error(f"Streaming agent error: {e}")
        if not emitted_audio:
            logger.warning("Stream failed before audio output. Falling back to invoke.")
            agent_response = agent.invoke(
                {"messages": [{"role": "user", "content": combined_input}]},
                config=agent_config,
            )
            response_text = agent_response["messages"][-1].content
            rewritten_sentence = ensure_devanagari_sentence(response_text)
            clean_sent = clean_text_for_tts(rewritten_sentence)
            if clean_sent:
                audio_chunk = generate_magpie_audio(clean_sent, speaker_idx=1)
                if len(audio_chunk) > 0:
                    yield (22050, audio_chunk)


# ==========================================
# 4. INITIALIZATION & UI
# ==========================================


def startup(*args) -> Generator[Tuple[int, np.ndarray], None, None]:
    """Greeting generated when WebRTC connection is established."""
    logger.info("Yielding silence to keep WebRTC alive...")
    yield (22050, np.zeros(22050, dtype=np.int16))

    greeting_text = (
        "नमस्ते, मैं रेना हूँ, रेनाटा की सपोर्ट असिस्टेंट। "
        "आज मैं आपकी कैसे मदद कर सकता हूँ?"
    )
    logger.info("Generating startup greeting with MagpieTTS...")
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
        ui_args={
            "title": "Renata Support Bot (NVIDIA Magpie TTS)",
            "webrtc": WebRTC(variant="textbox"),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RenataAI Voice Agent with Magpie TTS")
    parser.add_argument("--phone", action="store_true")
    parser.add_argument("--fastphone", action="store_true")
    parser.add_argument("--remote", action="store_true")
    args = parser.parse_args()

    stream = create_stream()
    logger.info("Stream handler configured with NVIDIA Magpie TTS")

    if args.remote:
        logger.info("Launching remote voice endpoint (Gradio Share + Cloudflare TURN)...")
        stream.ui.launch(share=True)

    elif args.phone:
        logger.info("Launching with phone interface...")
        from pyngrok import ngrok
        import uvicorn

        ngrok.set_auth_token(os.getenv("NGROK_AUTH_TOKEN"))
        public_url = ngrok.connect(8000, "http")
        logger.info(f"Public ngrok URL: {public_url}")

        app = FastAPI()
        stream.mount(app)
        uvicorn.run(app, host="127.0.0.1", port=8000)

    elif args.fastphone:
        stream.fastphone()

    else:
        logger.info("Launching custom Gradio UI...")
        stream.ui.launch()
