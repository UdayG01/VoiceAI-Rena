import argparse
import time
import os
import re
import tempfile
import torch
import numpy as np
from typing import Generator, Tuple
from contextlib import asynccontextmanager
import gradio as gr

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from snac import SNAC
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

# Keep your agent imports
from company_support_agent import agent, agent_config, set_user_query

# ==========================================
# SYSTEM & GPU CHECK
# ==========================================
def check_gpu_status():
    logger.info("🔍 Checking System Resources...")
    if not torch.cuda.is_available():
        logger.error("❌ CRITICAL: CUDA is NOT available. PyTorch is using CPU.")
        logger.error("   Ensure you have installed the correct torch version for your CUDA drivers.")
        logger.error("   Run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 (or your version)")
        raise RuntimeError("CUDA not found. Aborting to prevent infinite CPU hang.")
    
    device_name = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    vram_allocated = torch.cuda.memory_allocated(0) / 1024**3
    
    logger.success(f"✅ GPU Detected: {device_name}")
    logger.info(f"   VRAM Total: {vram_total:.2f} GB")
    logger.info(f"   VRAM Currently Allocated: {vram_allocated:.2f} GB")
    
    return "cuda"

DEVICE = check_gpu_status()

# ==========================================
# CONFIGURATION
# ==========================================
SPEAKER_NAME = "kavya"
SAMPLE_RATE = 24000

WHISPER_MODEL_SIZE = "large-v3" # shifted to this from the smaller model "medium"
WHISPER_COMPUTE_TYPE = "float16" 

# Veena Special Tokens
START_OF_SPEECH_TOKEN = 128257
END_OF_SPEECH_TOKEN = 128258
START_OF_HUMAN_TOKEN = 128259
END_OF_HUMAN_TOKEN = 128260
START_OF_AI_TOKEN = 128261
END_OF_AI_TOKEN = 128262
AUDIO_CODE_BASE_OFFSET = 128266

# ==========================================
# MODEL LOADING
# ==========================================

# 1. Load Whisper (STT) on GPU
logger.info(f"⏳ Loading Whisper ({WHISPER_MODEL_SIZE}) on GPU...")
try:
    local_stt_model = WhisperModel(
        WHISPER_MODEL_SIZE, 
        device="cuda", 
        compute_type=WHISPER_COMPUTE_TYPE
    )
    logger.success("✅ Whisper loaded on GPU.")
except Exception as e:
    logger.error(f"❌ Failed to load Whisper on GPU: {e}")
    raise e

# 2. Load Veena (TTS) in BFloat16 for speed (since we have 16GB VRAM)
logger.info("⏳ Loading Veena TTS models in BFloat16...")
try:
    # device_map="auto" usually finds the GPU, but we verify later
    veena_model = AutoModelForCausalLM.from_pretrained(
        "maya-research/veena-tts",
        torch_dtype=torch.bfloat16, # Much faster than 4-bit if VRAM allows
        device_map={"": torch.cuda.current_device()}, 
        trust_remote_code=True,
    )
    veena_tokenizer = AutoTokenizer.from_pretrained("maya-research/veena-tts", trust_remote_code=True)
    
    # Verify Veena placement
    logger.info(f"   Veena Model Device Map: {veena_model.hf_device_map}")

    # 3. Load SNAC (Audio Decoder)
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()
    logger.success("✅ Veena & SNAC loaded successfully on GPU (BFloat16).")

except Exception as e:
    logger.error(f"❌ Failed to load TTS models: {e}")
    raise e

# ==========================================
# CORE FUNCTIONS
# ==========================================

def decode_snac_tokens(snac_tokens, snac_model, silent=False):
    """De-interleave and decode SNAC tokens to audio"""
    if not snac_tokens:
        return None

    # FIX: Ensure tokens are divisible by 7 (discard incomplete frames)
    remainder = len(snac_tokens) % 7
    if remainder != 0:
        if not silent:
            logger.warning(f"⚠️ Trimming {remainder} incomplete tokens from audio stream.")
        snac_tokens = snac_tokens[:-remainder]
    
    if not snac_tokens:
        if not silent:
            logger.warning("⚠️ No valid audio tokens remained after trimming.")
        return None

    snac_device = next(snac_model.parameters()).device
    
    # De-interleave
    codes_lvl = [[] for _ in range(3)]
    llm_codebook_offsets = [AUDIO_CODE_BASE_OFFSET + i * 4096 for i in range(7)]

    for i in range(0, len(snac_tokens), 7):
        codes_lvl[0].append(snac_tokens[i] - llm_codebook_offsets[0])
        codes_lvl[1].append(snac_tokens[i+1] - llm_codebook_offsets[1])
        codes_lvl[1].append(snac_tokens[i+4] - llm_codebook_offsets[4])
        codes_lvl[2].append(snac_tokens[i+2] - llm_codebook_offsets[2])
        codes_lvl[2].append(snac_tokens[i+3] - llm_codebook_offsets[3])
        codes_lvl[2].append(snac_tokens[i+5] - llm_codebook_offsets[5])
        codes_lvl[2].append(snac_tokens[i+6] - llm_codebook_offsets[6])

    hierarchical_codes = []
    for lvl_codes in codes_lvl:
        tensor = torch.tensor(lvl_codes, dtype=torch.int32, device=snac_device).unsqueeze(0)
        hierarchical_codes.append(tensor)

    with torch.no_grad():
        audio_hat = snac_model.decode(hierarchical_codes)

    return audio_hat.squeeze().clamp(-1, 1).cpu().numpy()

    return audio_hat.squeeze().clamp(-1, 1).cpu().numpy()

def generate_speech(text, speaker=SPEAKER_NAME, temperature=0.1):
    """Generate speech (Blocking/Batch Mode)"""
    t0 = time.time()
    
    # Ensure speaker prompt is consistent
    prompt = f"<spk_{speaker}> {text}"
    prompt_tokens = veena_tokenizer.encode(prompt, add_special_tokens=False)

    input_tokens = [
        START_OF_HUMAN_TOKEN,
        *prompt_tokens,
        END_OF_HUMAN_TOKEN,
        START_OF_AI_TOKEN,
        START_OF_SPEECH_TOKEN
    ]

    input_ids = torch.tensor([input_tokens], device=veena_model.device)
    
    # Approx 7 audio tokens per text char
    # Increased max buffer slightly to prevent truncation
    max_tokens = min(int(len(text) * 1.5) * 7 + 100, 1024) 

    logger.debug(f"🏃 Starting Veena Generation (Input len: {len(text)} chars, Speaker: {speaker})...")
    
    with torch.no_grad():
        output = veena_model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False, # CHANGED: Greedy Decoding for speed
            temperature=None, # Temperature is not used in greedy
            top_p=None,      # Top-P is not used in greedy
            repetition_penalty=1.0, # Disable penalty to speed up (model is usually stable enough)
            pad_token_id=veena_tokenizer.pad_token_id,
            eos_token_id=[END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN]
        )

    t1 = time.time()
    logger.info(f"⚡ Generation Time: {t1 - t0:.2f}s")

    generated_ids = output[0][len(input_tokens):].tolist()
    
    # Extract only audio tokens
    snac_tokens = [
        token_id for token_id in generated_ids
        if AUDIO_CODE_BASE_OFFSET <= token_id < (AUDIO_CODE_BASE_OFFSET + 7 * 4096)
    ]

    if not snac_tokens:
        logger.warning(f"⚠️ Generation finished but NO audio tokens found. Raw output length: {len(generated_ids)}")
        decoded_text = veena_tokenizer.decode(generated_ids, skip_special_tokens=True)
        logger.debug(f"   Model Hallucination Check: {decoded_text}")
        return None

    audio = decode_snac_tokens(snac_tokens, snac_model)
    return audio

def clean_text_for_tts(text: str) -> str:
    # Remove markdown that might confuse Veena
    text = re.sub(r'\*\*|__', '', text) 
    text = re.sub(r'[#\*]', '', text)
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()

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

    # 1. Transcribe (GPU)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name

        segments, _ = local_stt_model.transcribe(
            temp_path, beam_size=5, vad_filter=True
        )
        transcript = " ".join([segment.text for segment in segments]).strip()
    except Exception as e:
        logger.error(f"Transcribe Error: {e}")
        transcript = ""
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    if not transcript:
        logger.info("Start of speech check - Silence detected.")
        return # Exit if silence

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
        raw_text = "I encountered a system error."

    # 3. TTS Generation (GPU) - BLOCKING
    clean_text = clean_text_for_tts(raw_text)
    logger.info(f'🤖 Bot: "{clean_text}"')

    try:
        # Generate full audio at once
        audio_array = generate_speech(clean_text, speaker=SPEAKER_NAME)
        if audio_array is not None:
            yield (SAMPLE_RATE, audio_array)
        else:
            logger.warning("TTS returned None (Silence).")
            
    except Exception as e:
        logger.error(f"TTS Generation Error: {e}")

def startup(*args):
    logger.info("🔊 Generating Startup Greeting...")
    try:
        # Generate full startup audio
        audio = generate_speech("Hi! I am Rena. How can I help you?", speaker=SPEAKER_NAME)
        if audio is not None:
            yield (SAMPLE_RATE, audio)
    except Exception as e:
        logger.error(f"Startup Audio Error: {e}")

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
        ui_args={"title": "Renata Support Bot (Veena TTS)"}
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