# FastRTC Groq Voice Agent

This project demonstrates voice interactions with AI assistants using FastRTC and Groq.

## Setup

1. Set up Python environment and install dependencies:
   ```
   uv venv
   source .venv/bin/activate
   uv sync
   ```

2. Copy the `.env.example` to `.env` and add your Groq API key from [Groq Console](https://console.groq.com/keys)

## Running the Application

Navigate to the src directory:
```
cd src
```

Run with web UI (default):
```
python fastrtc_groq_voice_stream.py
```

Run with phone interface (get a temporary phone number):
```
python fastrtc_groq_voice_stream.py --phone
```

## Usage Examples

### Math Agent Commands

- "What is 5 plus 7?"
- "Can you multiply 12 and 4?"
- "Calculate the sum of 123 and 456"
"# VoiceAI-Rena" 
