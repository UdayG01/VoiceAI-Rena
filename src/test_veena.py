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

# 2. Load Veena (TTS) in BFloat16 with Flash Attention 2
logger.info("⏳ Loading Veena TTS models in BFloat16 with Flash Attention 2...")
try:
    # device_map="auto" usually finds the GPU, but we verify later
    veena_model = AutoModelForCausalLM.from_pretrained(
        "maya-research/veena-tts",
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2",  # ENABLE FLASH ATTENTION 2
        device_map={"": torch.cuda.current_device()}, 
        trust_remote_code=True,
    )
    veena_tokenizer = AutoTokenizer.from_pretrained("maya-research/veena-tts", trust_remote_code=True)
    
    # Verify Veena placement
    logger.info(f"   Veena Model Device Map: {veena_model.hf_device_map}")

    # 3. Load SNAC (Audio Decoder)
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()
    logger.success("✅ Veena & SNAC loaded successfully on GPU (BFloat16 + FA2).")

except Exception as e:
    logger.error(f"❌ Failed to load TTS models: {e}")
    logger.warning("⚠️ If Flash Attention 2 failed, ensure you have 'flash-attn' installed.")
    raise e

# ==========================================
# CORE FUNCTIONS
# ==========================================

def decode_snac_tokens(snac_tokens, snac_model, silent=False):
    """De-interleave and decode SNAC tokens to audio"""
    if not snac_tokens:
        return None

    # Ensure tokens are divisible by 7 (discard incomplete frames)
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

# ==========================================
# STREAMING INFRASTRUCTURE
# ==========================================
from transformers import TextStreamer
from queue import Queue
from threading import Thread

class AudioTokenStreamer(TextStreamer):
    """
    Custom Streamer that captures tokens and puts them in a queue.
    It ignores text tokens and only captures audio code tokens.
    """
    def __init__(self, tokenizer, token_queue: Queue):
        super().__init__(tokenizer, skip_prompt=True)
        self.token_queue = token_queue
        
    def on_finalized_text(self, text: str, stream_end: bool = False):
        # We don't care about text decoding for audio generation
        pass

    def put(self, value):
        # Value is a tensor of token_ids
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("AudioTokenStreamer only supports batch size 1")
            
        token_ids = value.flatten().tolist()
        
        for token_id in token_ids:
            # Check if it is an audio token
            if AUDIO_CODE_BASE_OFFSET <= token_id < (AUDIO_CODE_BASE_OFFSET + 7 * 4096):
                self.token_queue.put(token_id)
                
    def end(self):
        self.token_queue.put(None) # Signal end of stream

def stream_audio_from_text(text: str, speaker=SPEAKER_NAME) -> Generator[np.ndarray, None, None]:
    """
    Generates audio in a streaming fashion.
    1. Starts model.generate in a separate thread.
    2. Consumes tokens from a queue.
    3. Decodes audio in small chunks (e.g., 28 tokens = 4 SNAC frames).
    """
    # 1. Prepare Prompt
    prompt = f"<spk_{speaker}> {text}"
    prompt_tokens = veena_tokenizer.encode(prompt, add_special_tokens=False)
    input_tokens = [
        START_OF_HUMAN_TOKEN, *prompt_tokens, END_OF_HUMAN_TOKEN,
        START_OF_AI_TOKEN, START_OF_SPEECH_TOKEN
    ]
    input_ids = torch.tensor([input_tokens], device=veena_model.device)
    
    # 2. Prepare Streamer & Queue
    token_queue = Queue()
    streamer = AudioTokenStreamer(veena_tokenizer, token_queue)
    
    # 3. Define Generation Config
    max_tokens = min(int(len(text) * 1.5) * 7 + 100, 2048)
    gen_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_tokens,
        do_sample=False, # FAST greedy decoding
        repetition_penalty=1.0, 
        pad_token_id=veena_tokenizer.pad_token_id,
        eos_token_id=[END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN],
        streamer=streamer
    )
    
    # 4. Start Generation Thread
    thread = Thread(target=veena_model.generate, kwargs=gen_kwargs)
    thread.start()
    
    # 5. Consume Tokens & Yield Audio
    buffer = []
    CHUNK_SIZE = 28 # 28 tokens = 4 SNAC frames (divisible by 7)
    
    while True:
        token = token_queue.get()
        if token is None: # End of stream
            break
            
        buffer.append(token)
        
        if len(buffer) >= CHUNK_SIZE:
            # Decode chunk
            audio_chunk = decode_snac_tokens(buffer, snac_model, silent=True)
            if audio_chunk is not None:
                yield audio_chunk
            buffer = []
            
    # Decode remaining tokens
    if buffer:
        audio_chunk = decode_snac_tokens(buffer, snac_model, silent=True)
        if audio_chunk is not None:
            yield audio_chunk
            
    thread.join()

def clean_text_for_tts(text: str) -> str:
    text = re.sub(r'\*\*|__', '', text) 
    text = re.sub(r'[#\*]', '', text)
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

    # 1. Transcribe
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
        return 

    logger.info(f'📝 User: "{transcript}"')
    
    # 2. Agent Logic Setup
    set_user_query(transcript)
    combined_input = transcript
    if user_name: 
        combined_input += f" (User: {user_name})"

    # 3. STREAMING LLM & TTS PIPELINE
    # Patterns to split sentences: . ? ! 
    split_pattern = re.compile(r'(?<=[.?!])\s+')
    
    current_buffer = ""
    full_response_text = ""
    
    try:
        logger.info("🧠 Agent Thinking (Streaming mode)...")
        
        # Use stream_mode="messages" to get message chunks
        stream = agent.stream(
            {"messages": [{"role": "user", "content": combined_input}]}, 
            config=agent_config,
            stream_mode="messages"
        )

        for event in stream:
            # We are looking for (message, metadata) tuples or just messages depending on the version
            # In recent LangGraph, stream_mode="messages" yields (message, metadata) tuples or just message objects

            msg_chunk = None
            if isinstance(event, tuple):
                 msg_chunk = event[0]
            else:
                 msg_chunk = event

            
            if hasattr(msg_chunk, 'content') and msg_chunk.content:
                # Filter out Tool calls or non-final-answer chunks if necessary
                # Simple heuristic: If it has content and is NOT a tool call
                content = msg_chunk.content
                if isinstance(content, str):
                    current_buffer += content
                    full_response_text += content
                    
                    # Check for sentence boundaries in buffer
                    # We look for punctuation followed by space or end of string
                    if re.search(r'[.?!]\s', current_buffer):
                        # Split buffer into sentences
                        parts = split_pattern.split(current_buffer)
                        
                        # Process all complete sentences (all except the last potentially incomplete one)
                        # BUT: If the last part ends with punctuation, it is also complete.
                        
                        to_speak = []
                        if len(parts) > 1:
                            to_speak = parts[:-1]
                            current_buffer = parts[-1]
                        elif re.search(r'[.?!]$', parts[0]):
                             to_speak = [parts[0]]
                             current_buffer = ""
                        
                        for sentence in to_speak:
                            sentence = sentence.strip()
                            if sentence:
                                clean_sent = clean_text_for_tts(sentence)
                                if clean_sent:
                                    logger.debug(f"🗣️ TTS Streaming Sentence: {clean_sent}")
                                    for audio_chunk in stream_audio_from_text(clean_sent, speaker=SPEAKER_NAME):
                                        yield (SAMPLE_RATE, audio_chunk)

        # Process any remaining text in buffer
        if current_buffer.strip():
            clean_sent = clean_text_for_tts(current_buffer)
            if clean_sent:
                logger.debug(f"🗣️ TTS Streaming Final Sentence: {clean_sent}")
                for audio_chunk in stream_audio_from_text(clean_sent, speaker=SPEAKER_NAME):
                    yield (SAMPLE_RATE, audio_chunk)

        logger.info(f"🧠 Full Response: {full_response_text[:50]}...")

    except Exception as e:
        logger.error(f"Streaming Error: {e}")
        # Only fallback if absolutely nothing was generated
        if not full_response_text:
             logger.warning("⚠️ Stream failed, attempting fallback invoke.")
             try:
                agent_res = agent.invoke(
                    {"messages": [{"role": "user", "content": combined_input}]}, 
                    config=agent_config
                )
                raw_text = agent_res["messages"][-1].content
                # ... existing fallback processing ...
                clean_text = clean_text_for_tts(raw_text)
                for audio_chunk in stream_audio_from_text(clean_text, speaker=SPEAKER_NAME):
                    yield (SAMPLE_RATE, audio_chunk)
             except Exception as fallback_err:
                 logger.error(f"Fallback Error: {fallback_err}")

def startup(*args):
    logger.info("🔊 Generating Startup Greeting...")
    try:
        # Stream startup too
        for audio_chunk in stream_audio_from_text("Hi! I am Rena. How can I help you?", speaker=SPEAKER_NAME):
            yield (SAMPLE_RATE, audio_chunk)
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
        ui_args={"title": "Renata Support Bot (Veena TTS - True Streaming)"}
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