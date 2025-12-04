# Rena - RenataAI's Support Assistant

Creating an AI voice assistant prototype for RenataAI team.

## Setup

1. Set up Python environment and install dependencies:
   ```
   uv venv
   source .venv/bin/activate
   uv sync
   ```

2. Copy the `.env.example` to `.env` and add your Groq API key from [Groq Console](https://console.groq.com/keys)

## Running the Application

Run with web UI:
```
python src/main.py
```
Run with Twilio contact:
```
python src/main.py --phone
```
Run with FastRTC phone (quick trial):
```
python src/main.py --fastphone
```

## First look
[Rena_Demo](https://github.com/user-attachments/assets/5e2037b3-f652-4910-ba17-abd73ea7d059)

## Remaining features
[x]. Local deployment of
  - LLM (Ollama - Qwen3:0.6B worked perfectly on my system - (i5 12500H 16GB RAM) (RTX 3050 6GB VRAM)
  - TTS - explorint CoquiTTS, KokoroTTS, and some other hugging face options like XTTS
  - STT - moonshine/base not giving accurate results, we will probably move to Whisper
[]. Fallback/Failsafe for tools
  - If valid email not received, the LLM should again ask for a valid email address, and repeat it, currently, there is no failsafe, even if correct email is not received, it generates a ticket and terminates.
  - Async voice output whilst RAG outputs are recieved to keep the user engaged.
[]. Alternate input method for name and email
  - A failsafe for when the transcription fails

