import argparse
from typing import Generator, Tuple
import numpy as np
import os
import re
import tempfile
from contextlib import asynccontextmanager

# FastRTC and Web/App imports
from fastrtc import (
    AlgoOptions,
    ReplyOnPause,
    Stream,
    SileroVadOptions,
    audio_to_bytes,
    get_cloudflare_turn_credentials # Added for remote TURN Server tunneling
)
from fastapi import FastAPI
import gradio as gr
from loguru import logger

# Agent imports
from company_support_agent import agent, agent_config, set_user_query

# ML / Audio imports
import torch
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from snac import SNAC

# --------------------------------------------------------------------
# 1. Initialize the Local Models (STT & TTS)
# --------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Faster-Whisper for STT
WHISPER_MODEL_SIZE = "medium"
WHISPER_DEVICE = "cpu"
try:
    logger.info(f"⏳ Loading local STT model ({WHISPER_MODEL_SIZE} on {WHISPER_DEVICE})...")
    local_stt_model = WhisperModel(
        WHISPER_MODEL_SIZE, 
        device=WHISPER_DEVICE, 
        compute_type="int8"
    )
    logger.info("✅ Local STT model loaded successfully.")
except Exception as e:
    logger.error(f"❌ Failed to load Faster-Whisper model: {e}")

# Load Custom Orpheus 3B TTS and SNAC Decoder
try:
    custom_tts_model_name = "UdayG01/orpheus_3b-hi-ft"
    logger.info("⏳ Loading Tokenizer and Orpheus LLM...")
    tokenizer = AutoTokenizer.from_pretrained(custom_tts_model_name)
    tts_llm = AutoModelForCausalLM.from_pretrained(
        custom_tts_model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    logger.info("⏳ Loading SNAC Audio Decoder...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)
    snac_model.eval()
    logger.info("✅ Custom TTS models loaded successfully.")
except Exception as e:
    logger.error(f"❌ Failed to load Custom TTS models: {e}")


# --------------------------------------------------------------------
# 2. Custom TTS Generation Logic
# --------------------------------------------------------------------

def generate_custom_tts(text: str, voice: str = "tara") -> Tuple[int, np.ndarray]:
    """
    Generates audio using the fine-tuned Orpheus 3B model and SNAC decoder.
    """
    # Special token offsets
    tokeniser_length = 128256
    start_of_text = 128000
    end_of_text = 128009
    
    start_of_speech = tokeniser_length + 1
    end_of_speech = tokeniser_length + 2
    start_of_human = tokeniser_length + 3
    end_of_human = tokeniser_length + 4
    start_of_ai = tokeniser_length + 5
    
    # Format prompt
    text_prompt = f"{voice}: {text}" if voice else text
    text_ids = tokenizer.encode(text_prompt, add_special_tokens=True) + [end_of_text]
    
    # Construct the input sequence
    input_ids = [start_of_human] + text_ids + [end_of_human, start_of_ai, start_of_speech]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # Generate tokens representing audio codes
    with torch.no_grad():
        generated_ids = tts_llm.generate(
            input_ids=input_tensor,
            max_new_tokens=4096,
            eos_token_id=end_of_speech,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
            use_cache=True
        )
        
    # Extract only newly generated tokens
    new_tokens = generated_ids[0][len(input_ids):].tolist()
    audio_tokens = [t for t in new_tokens if t >= 128266]
    
    # Ensure length is a multiple of 7
    length = (len(audio_tokens) // 7) * 7
    if length == 0:
        return (24000, np.zeros(0, dtype=np.int16))
        
    audio_tokens = audio_tokens[:length]
    
    # Demultiplex tokens back into SNAC hierarchical layers
    codes_0, codes_1, codes_2 = [], [], []
    for i in range(0, length, 7):
        codes_0.append(audio_tokens[i] - 128266)
        codes_1.extend([
            audio_tokens[i+1] - 128266 - 4096, 
            audio_tokens[i+4] - 128266 - 16384
        ])
        codes_2.extend([
            audio_tokens[i+2] - 128266 - 8192, 
            audio_tokens[i+3] - 128266 - 12288, 
            audio_tokens[i+5] - 128266 - 20480, 
            audio_tokens[i+6] - 128266 - 24576
        ])
        
    c0_tensor = torch.tensor([codes_0], dtype=torch.int32).to(device)
    c1_tensor = torch.tensor([codes_1], dtype=torch.int32).to(device)
    c2_tensor = torch.tensor([codes_2], dtype=torch.int32).to(device)
    
    # Decode audio using SNAC
    with torch.no_grad():
        audio_waveform = snac_model.decode([c0_tensor, c1_tensor, c2_tensor])
        
    audio_data = audio_waveform.squeeze().cpu().numpy()
    
    # REQUIRED: Scale float32 to int16 PCM format for WebRTC compatibility
    audio_data = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
    
    return (24000, audio_data)


# --------------------------------------------------------------------
# 3. FastRTC Handlers
# --------------------------------------------------------------------

def clean_text_for_tts(text: str) -> str:
    """Removes markdown and special chars before passing to TTS."""
    text = re.sub(r'\*\*|__', '', text)
    text = re.sub(r'\*', '', text)
    text = re.sub(r'#+', '', text)
    text = re.sub(r'^\s*[\-\*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    return text.strip()


def response(
    audio: tuple[int, np.ndarray],
    user_name: str = "",
    user_email: str = ""
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Process audio input locally, generate response via Agent, and deliver custom TTS.
    """
    logger.info("🎙️ Received audio input")
    if user_name or user_email:
        logger.info(f"⌨️ Received context - Name: {user_name}, Email: {user_email}")

    audio_bytes = audio_to_bytes(audio)

    # Transcription
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        temp_path = tmp.name

    logger.debug("🔍 Transcribing audio locally with Faster-Whisper...")
    segments, info = local_stt_model.transcribe(temp_path, beam_size=5, vad_filter=True)
    detected_lang = info.language
    transcript = " ".join([segment.text for segment in segments]).strip()
    os.remove(temp_path)
    
    if not transcript:
        transcript = " "

    logger.info(f'📝 Transcribed: "{transcript}"')
    
    # Explicitly instruct the LLM to output Devanagari if Hindi is detected
    lang_instruction = " (IMPORTANT: Reply in native Hindi Devanagari script)" if detected_lang == "hi" else ""
    combined_input = f"{transcript}{lang_instruction}"

    if user_name or user_email:
        combined_input += f" (Context: Name: {user_name}, Email: {user_email})"

    # LLM Agent execution
    logger.debug(f"🧠 Agent Input: {combined_input}")
    set_user_query(transcript)  
    
    agent_response = agent.invoke(
        {"messages": [{"role": "user", "content": combined_input}]}, config=agent_config
    )
    
    response_text = agent_response["messages"][-1].content
    cleaned_text = clean_text_for_tts(response_text)
    logger.info(f'💬 Response (Cleaned, Devanagari): "{cleaned_text}"')

    # Custom TTS Generation (Transliteration removed, expecting LLM native Devanagari)
    logger.debug("🔊 Generating speech using Custom Orpheus TTS...")
    audio_chunk = generate_custom_tts(cleaned_text, voice="tara")
    yield audio_chunk


def startup(*args) -> Generator[Tuple[int, np.ndarray], None, None]:
    """Initial greeting when connection is established."""
    # 1. Yield a dummy silent int16 array instantly to satisfy the WebRTC handshake timeout
    logger.debug("🔊 Sending dummy audio to keep WebRTC handshake alive...")
    yield (24000, np.zeros(24000, dtype=np.int16))
    
    # 2. Then generate and yield the actual greeting
    greeting_text = "नमस्ते, मैं रेना हूँ, रेनाटा की सपोर्ट असिस्टेंट। आज मैं आपकी कैसे मदद कर सकती हूँ?"
    logger.debug("🔊 Generating startup speech...")
    yield generate_custom_tts(greeting_text, voice="tara")


def create_stream() -> Stream:
    """Create and configure the FastRTC Stream instance."""
    return Stream(
        modality="audio",
        mode="send-receive",
        rtc_configuration=get_cloudflare_turn_credentials, # Cloudflare Relay for strict firewalls
        handler=ReplyOnPause(
            response,
            algo_options=AlgoOptions(speech_threshold=0.6),
            model_options=SileroVadOptions(threshold=0.7),
            startup_fn=startup
        ),
        additional_inputs=[
            gr.Textbox(label="Name", placeholder="Enter your name"),
            gr.Textbox(label="Email", placeholder="Enter your email")
        ],
        ui_args={"title": "Renata Support Bot"}
    )


# --------------------------------------------------------------------
# 4. Entry Point
# --------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RenataAI Voice Agent with Custom TTS")
    parser.add_argument("--phone", action="store_true")
    parser.add_argument("--fastphone", action="store_true")
    parser.add_argument("--remote", action="store_true")
    args = parser.parse_args()

    stream = create_stream()
    logger.info("🎧 Stream handler configured with Custom TTS (Native Devanagari)")

    if args.remote:
        # Replaced ngrok with Gradio Share + TURN Server for proper UDP routing
        logger.info("🌍 Launching REMOTE voice endpoint (Gradio Share + Cloudflare TURN)...")
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