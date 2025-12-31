import argparse
import asyncio
import os
import re
import tempfile
import numpy as np
import edge_tts
from typing import Generator, Tuple
from faster_whisper import WhisperModel
from loguru import logger
import gradio as gr
from fastapi import FastAPI
from fastrtc import (
    AlgoOptions,
    ReplyOnPause,
    Stream,
    SileroVadOptions,
    audio_to_bytes
)
# Reuse your existing agent setup
from company_support_agent import agent, agent_config, set_user_query

# ==========================================
# CONFIGURATION
# ==========================================
# Voices: https://github.com/rany2/edge-tts/blob/master/src/edge_tts/constants.py
# Good Hindi/English options:
# - en-IN-NeerjaNeural (Female, Indian English)
# - en-IN-PrabhatNeural (Male, Indian English)
# - hi-IN-SwaraNeural (Female, Hindi)
# - hi-IN-MadhurNeural (Male, Hindi)

DEFAULT_VOICE = "en-IN-NeerjaNeural" 
HINDI_VOICE = "hi-IN-SwaraNeural"

SAMPLE_RATE = 24000 

# ==========================================
# STT MODEL (Faster Whisper)
# ==========================================
logger.info("⏳ Loading Whisper (Medium) on GPU...")
try:
    stt_model = WhisperModel("medium", device="cuda", compute_type="float16")
    logger.success("✅ Whisper loaded on GPU.")
except Exception as e:
    logger.error(f"❌ Failed to load Whisper: {e}")
    raise e

# ==========================================
# EDGE TTS HELPERS
# ==========================================

async def generate_edge_audio(text: str, voice: str) -> str:
    """
    Generates audio file using Edge TTS. 
    Returns path to the temporary mp3 file.
    """
    try:
        communicate = edge_tts.Communicate(text, voice)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            temp_path = tmp.name
            
        await communicate.save(temp_path)
        return temp_path
    except Exception as e:
        logger.error(f"EdgeTTS Error: {e}")
        return None

def audio_file_to_pcm(file_path: str, target_sr: int = 24000):
    """
    Reads an audio file and converts it to PCM numpy array + resamples if needed.
    """
    try:
        import librosa
        # librosa handles resampling automatically
        y, _ = librosa.load(file_path, sr=target_sr, mono=True)
        return y
    except Exception as e:
        logger.error(f"Audio Decode Error: {e}")
        return None

def clean_text_for_tts(text: str) -> str:
    # Basic cleanup
    text = re.sub(r'\*\*|__', '', text) 
    text = re.sub(r'[#\*]', '', text)
    return text.strip()

def detect_language_simple(text: str):
    """
    Simple heuristic: if text contains Devanagari chars, treat as Hindi.
    """
    # Range for Devanagari: U+0900 to U+097F
    for char in text:
        if '\u0900' <= char <= '\u097F':
            return "hi"
    return "en"

# ==========================================
# FASTRTC HANDLER
# ==========================================

def response(
    audio: tuple[int, np.ndarray],
    user_name: str = "",
    user_email: str = ""
) -> Generator[Tuple[int, np.ndarray], None, None]:
    
    logger.info("🎙️ Audio received.")
    audio_bytes = audio_to_bytes(audio)

    # 1. Transcribe
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name

        segments, _ = stt_model.transcribe(temp_path, beam_size=5)
        transcript = " ".join([segment.text for segment in segments]).strip()
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    if not transcript:
        return 

    logger.info(f'📝 User: "{transcript}"')
    
    # 2. Agent Logic
    set_user_query(transcript)
    combined_input = transcript
    if user_name: 
        combined_input += f" (User: {user_name})"

    try:
        agent_res = agent.invoke(
            {"messages": [{"role": "user", "content": combined_input}]}, 
            config=agent_config
        )
        raw_text = agent_res["messages"][-1].content
    except Exception as e:
        logger.error(f"Agent Error: {e}")
        raw_text = "Sorry, I encountered an error."

    # 3. TTS Generation (Edge TTS)
    clean_text = clean_text_for_tts(raw_text)
    logger.info(f'🤖 Bot: "{clean_text}"')
    
    # Select Voice based on script
    lang = detect_language_simple(clean_text)
    selected_voice = HINDI_VOICE if lang == "hi" else DEFAULT_VOICE
    logger.debug(f"   Selected Voice: {selected_voice} (Lang: {lang})")

    # Edge TTS requires async, but fastrtc response is a sync generator.
    # We run the async function in the event loop or use asyncio.run (if safe)
    try:
        mp3_path = asyncio.run(generate_edge_audio(clean_text, selected_voice))
        
        if mp3_path:
            pcm_audio = audio_file_to_pcm(mp3_path, target_sr=SAMPLE_RATE)
            if pcm_audio is not None:
                # remove file
                try: os.remove(mp3_path) 
                except: pass
                
                # Yield the full audio
                yield (SAMPLE_RATE, pcm_audio)
    except Exception as e:
        logger.error(f"TTS pipeline failed: {e}")

def startup(*args):
    logger.info("🔊 Generating Startup Greeting...")
    yield (SAMPLE_RATE, np.zeros(SAMPLE_RATE)) # small silence to prime
    try:
        mp3_path = asyncio.run(generate_edge_audio("Namaste! I am Rena. How can I help you?", DEFAULT_VOICE))
        if mp3_path:
            pcm = audio_file_to_pcm(mp3_path, SAMPLE_RATE)
            yield (SAMPLE_RATE, pcm)
            try: os.remove(mp3_path)
            except: pass
    except:
        pass

def create_stream() -> Stream:
    return Stream(
        modality="audio",
        mode="send-receive",
        handler=ReplyOnPause(
            response,
            algo_options=AlgoOptions(speech_threshold=0.6),
            model_options=SileroVadOptions(threshold=0.6),
            startup_fn=startup
        ),
        additional_inputs=[
            gr.Textbox(label="Name"),
            gr.Textbox(label="Email")
        ],
        ui_args={"title": "Renata Support Bot (Edge TTS)"}
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phone", action="store_true")
    parser.add_argument("--remote", action="store_true")
    args = parser.parse_args()

    stream = create_stream()
    
    if args.remote:
        # Remote config (similar to your other files)
        from pyngrok import ngrok
        import uvicorn
        ngrok.set_auth_token(os.getenv("NGROK_AUTH_TOKEN"))
        app = FastAPI()
        stream.mount(app)
        app = gr.mount_gradio_app(app, stream.ui, path="/")
        public_url = ngrok.connect(8000, "http")
        print(f"URL: {public_url}")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    elif args.phone:
         # Phone config
        from pyngrok import ngrok
        import uvicorn
        ngrok.set_auth_token(os.getenv("NGROK_AUTH_TOKEN"))
        public_url = ngrok.connect(8000, "http")
        print(f"URL: {public_url}")
        app = FastAPI()
        stream.mount(app)
        uvicorn.run(app, host="127.0.0.1", port=8000)
    else:
        stream.ui.launch()
