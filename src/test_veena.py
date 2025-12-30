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
    # get_tts_model, # Removed Kokoro
    # get_stt_model, # Commented out as we are using faster_whisper
    # KokoroTTSOptions, # Removed Kokoro
    SileroVadOptions,
    WebRTC,
    audio_to_bytes
)
from fastapi import FastAPI

import gradio as gr
from loguru import logger

# Keep agent imports for the LLM part
from company_support_agent import agent, agent_config

# Veena TTS Imports
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from snac import SNAC

# Note: Logger is configured in company_support_agent.py
# We import the configured logger, so no need to reconfigure here

# 2. Initialize the Local Models (Global scope to load weights once)
# Moonshine is the default STT model in FastRTC
#stt_model = get_stt_model() 

"""
START LOADING CTRANSLATE2 (FASTER-WHISPER)
"""

# Use compute_type='int8' for 4x speedup on CPU over vanilla Whisper.
from faster_whisper import WhisperModel
WHISPER_MODEL_SIZE = "medium"
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

"""
START LOADING VEENA TTS
"""
# Control token IDs (fixed for Veena)
START_OF_SPEECH_TOKEN = 128257
END_OF_SPEECH_TOKEN = 128258
START_OF_HUMAN_TOKEN = 128259
END_OF_HUMAN_TOKEN = 128260
START_OF_AI_TOKEN = 128261
END_OF_AI_TOKEN = 128262
AUDIO_CODE_BASE_OFFSET = 128266

logger.info("⏳ Loading Veena TTS models...")
try:
    # Model configuration for 4-bit inference
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model and tokenizer
    # Using device_map="auto" which usually puts it on GPU if available
    veena_model = AutoModelForCausalLM.from_pretrained(
        "maya-research/veena-tts",
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    veena_tokenizer = AutoTokenizer.from_pretrained("maya-research/veena-tts", trust_remote_code=True)

    # Initialize SNAC decoder
    # Assuming CUDA availability for consistency with test_hinglish_tts.py
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()
    
    logger.info(" Veena TTS models loaded successfully.")
except Exception as e:
    logger.error(f" Failed to load Veena TTS models: {e}")
    # This will likely crash if models fail to load

"""
END LOADING VEENA TTS
"""

def decode_snac_tokens(snac_tokens, snac_model):
    """De-interleave and decode SNAC tokens to audio"""
    if not snac_tokens or len(snac_tokens) % 7 != 0:
        return None

    # Get the device of the SNAC model.
    snac_device = next(snac_model.parameters()).device

    # De-interleave tokens into 3 hierarchical levels
    codes_lvl = [[] for _ in range(3)]
    llm_codebook_offsets = [AUDIO_CODE_BASE_OFFSET + i * 4096 for i in range(7)]

    for i in range(0, len(snac_tokens), 7):
        # Level 0: Coarse (1 token)
        codes_lvl[0].append(snac_tokens[i] - llm_codebook_offsets[0])
        # Level 1: Medium (2 tokens)
        codes_lvl[1].append(snac_tokens[i+1] - llm_codebook_offsets[1])
        codes_lvl[1].append(snac_tokens[i+4] - llm_codebook_offsets[4])
        # Level 2: Fine (4 tokens)
        codes_lvl[2].append(snac_tokens[i+2] - llm_codebook_offsets[2])
        codes_lvl[2].append(snac_tokens[i+3] - llm_codebook_offsets[3])
        codes_lvl[2].append(snac_tokens[i+5] - llm_codebook_offsets[5])
        codes_lvl[2].append(snac_tokens[i+6] - llm_codebook_offsets[6])

    # Convert to tensors for SNAC decoder
    hierarchical_codes = []
    for lvl_codes in codes_lvl:
        tensor = torch.tensor(lvl_codes, dtype=torch.int32, device=snac_device).unsqueeze(0)
        if torch.any((tensor < 0) | (tensor > 4095)):
            raise ValueError("Invalid SNAC token values")
        hierarchical_codes.append(tensor)

    # Decode with SNAC
    with torch.no_grad():
        audio_hat = snac_model.decode(hierarchical_codes)

    return audio_hat.squeeze().clamp(-1, 1).cpu().numpy()

def generate_speech(text, speaker="kavya", temperature=0.4, top_p=0.9):
    """Generate speech from text using specified speaker voice"""

    # Prepare input with speaker token
    prompt = f"<spk_{speaker}> {text}"
    prompt_tokens = veena_tokenizer.encode(prompt, add_special_tokens=False)

    # Construct full sequence: [HUMAN] <spk_speaker> text [/HUMAN] [AI] [SPEECH]
    input_tokens = [
        START_OF_HUMAN_TOKEN,
        *prompt_tokens,
        END_OF_HUMAN_TOKEN,
        START_OF_AI_TOKEN,
        START_OF_SPEECH_TOKEN
    ]

    input_ids = torch.tensor([input_tokens], device=veena_model.device)

    # Calculate max tokens based on text length
    max_tokens = min(int(len(text) * 1.3) * 7 + 21, 700)

    # Generate audio tokens
    with torch.no_grad():
        output = veena_model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.05,
            pad_token_id=veena_tokenizer.pad_token_id,
            eos_token_id=[END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN]
        )

    # Extract SNAC tokens
    generated_ids = output[0][len(input_tokens):].tolist()
    snac_tokens = [
        token_id for token_id in generated_ids
        if AUDIO_CODE_BASE_OFFSET <= token_id < (AUDIO_CODE_BASE_OFFSET + 7 * 4096)
    ]

    if not snac_tokens:
        raise ValueError("No audio tokens generated")

    # Decode audio
    audio = decode_snac_tokens(snac_tokens, snac_model)
    return audio

def response(
    audio: tuple[int, np.ndarray],
    user_name: str = "",
    user_email: str = ""
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Process audio input locally, generate response via Groq/LangGraph, and deliver local TTS using Veena.
    """
    logger.info("🎙️ Received audio input")
    if user_name or user_email:
        logger.info(f"⌨️ Received context - Name: {user_name}, Email: {user_email}")

    # Convert to bytes
    audio_bytes = audio_to_bytes(audio)

    # Create temporary mp3 file
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        temp_path = tmp.name

    # Faster-whisper transcription
    logger.debug("🔍 Transcribing audio locally with Faster-Whisper...")
    segments, _ = local_stt_model.transcribe(
        temp_path,
        language="en",
        # language="hi",
        beam_size=5,
        vad_filter=True,
    )

    # Build transcript
    transcript = " ".join([segment.text for segment in segments]).strip()

    # Delete temp file
    os.remove(temp_path)
    
    # Handle case where transcription is empty or just noise (prevents LLM error)
    if not transcript:
        transcript = " " # Use a space or a default phrase

    #  END: CTRANSLATE2 TEST
    
    logger.info(f'📝 Transcribed: "{transcript}"')
    
    combined_input = transcript
    if user_name or user_email:
        context_str = f"(Context: Name: {user_name}, Email: {user_email})"
        combined_input = f"{transcript} {context_str}"

    # 4. LLM Processing (Still using your Groq-based Agent)
    logger.debug("🧠 Running agent...")
    logger.debug(f"🧠 Agent Input: {combined_input}")
    
    # Store original query for validation 
    from company_support_agent import set_user_query
    set_user_query(transcript)  
    
    agent_response = agent.invoke(
        {"messages": [{"role": "user", "content": combined_input}]}, config=agent_config
    )
    
    # Log agent response details
    logger.debug(f"🧠 Agent returned {len(agent_response['messages'])} messages")
    
    # Extract and log tool usage
    for i, msg in enumerate(agent_response['messages']):
        if hasattr(msg, 'type'):
            if msg.type == 'ai' and hasattr(msg, 'tool_calls') and msg.tool_calls:
                logger.info(f"🤖 Agent made {len(msg.tool_calls)} tool call(s)")
                for tc in msg.tool_calls:
                    logger.debug(f"   - Tool: {tc.get('name', 'unknown')}, Args: {tc.get('args', {})}")
            elif msg.type == 'tool':
                logger.debug(f"   - Tool response: {msg.content[:200]}...")  # First 200 chars
    
    response_text = agent_response["messages"][-1].content
    logger.debug(f"🧠 Raw Agent Response: {response_text}")
    
    # Clean text for TTS (remove markdown, special chars)
    import re
    def clean_text_for_tts(text: str) -> str:
        # Remove bold/italic markers (** or *)
        text = re.sub(r'\*\*|__', '', text)
        text = re.sub(r'\*', '', text)
        # Remove headers (#)
        text = re.sub(r'#+', '', text)
        # Remove list markers (1., -, etc.) but keep the content
        text = re.sub(r'^\s*[\-\*]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        return text.strip()

    response_text = clean_text_for_tts(response_text)
    logger.info(f'💬 Response (Cleaned): "{response_text}"')

    # 5. Local Veena TTS
    logger.debug("🔊 Generating speech locally with VeenaTTS...")
    
    # Fixed speaker as per request
    speaker = "kavya"
    
    try:
        audio_array = generate_speech(response_text, speaker=speaker)
        
        # Veena generates 24kHz audio
        sample_rate = 24000
        
        if audio_array is not None:
             # FastRTC expects a generator yielding (sample_rate, audio_chunk)
            yield (sample_rate, audio_array)
    except Exception as e:
        logger.error(f"Error generating Veena TTS audio: {e}")
        # Could maybe yield an error message or empty chunk but better to just log

def startup(*args):
    """Startup function to process initial Greeting"""
    logger.info("🔊 Generating startup speech with VeenaTTS...")
    welcome_text = "Hi!, I'm Rena, Renata's support assistant. How can I help you today?"
    speaker = "kavya"
    try:
        audio_array = generate_speech(welcome_text, speaker=speaker)
        if audio_array is not None:
            yield (24000, audio_array)
    except Exception as e:
        logger.error(f"Error generating startup audio: {e}")

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
        additional_inputs=[
            gr.Textbox(label="Name", placeholder="Enter your name"),
            gr.Textbox(label="Email", placeholder="Enter your email")
        ],
        ui_args={"title": "Renata Support Bot (Veena TTS)"}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RenataAI Voice Agent - Veena TTS Version")
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
