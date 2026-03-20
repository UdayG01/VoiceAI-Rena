# Setting up NVIDIA NeMo & Magpie TTS for Windows/WSL2

NVIDIA's **Magpie TTS** is built on the [NeMo Framework](https://github.com/NVIDIA/NeMo), which is heavily optimized for Linux environments. Installing NeMo directly on native Windows can be extremely challenging due to dependencies on PyTorch audio bindings, Cython, and Linux-specific C++ compilers.

This guide provides the two best methods to set up your environment to run the `nemo_magpie_tts.py` script.

---

## Prerequisites
1. **Windows Subsystem for Linux (WSL2)**: Ensure you have WSL2 installed with an Ubuntu distribution.
2. **NVIDIA GPU Drivers**: Install the latest NVIDIA Game Ready or Studio drivers on Windows. (WSL2 automatically gets GPU passthrough; do *not* install separate display drivers inside WSL2).
3. **Docker Desktop (For Method A)**: Ensure Docker Desktop is installed and the WSL2 integration is enabled for your Ubuntu distro.

---

## Method A: Using NVIDIA's Official Docker Container (Highly Recommended)
Using the official NeMo Docker container avoids 99% of dependency hell.

### 1. Launch the NeMo Container
Open your WSL2 terminal (Ubuntu) and run the following command to pull and run the NeMo container. This maps your project directory to `/workspace/app` inside the container and exposes port `8000`.

```bash
docker run --gpus all -it --rm \
  -v /mnt/c/Work/Renata/VoiceAI/fastrtc-groq-voice-agent:/workspace/app \
  -p 8000:8000 \
  -p 7860:7860 \
  nvcr.io/nvidia/nemo:24.05.01.framework bash
```
*(Note: You do not need an NGC API key to pull this public framework container).*

### 2. Install Project Dependencies using `uv`
Once inside the Docker container, navigate to your app directory. We will install the `uv` package manager and use it to sync your project.

```bash
cd /workspace/app

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Create a virtual environment that inherits the container's pre-installed NeMo packages
uv venv --system-site-packages

# Sync the remaining project dependencies
uv sync
```

### 3. Run the Bot
Now you can start the FastRTC agent using `uv run`:

```bash
uv run python src/nemo_magpie_tts.py --remote
```

---

## Method B: Native WSL2 using `uv` (Alternative)
If you prefer not to use Docker, you can use `uv` directly in your WSL2 Ubuntu instance to manage everything, including the Python installation.

### 1. Install `uv`
Open your WSL2 terminal and run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 2. Sync Dependencies
Navigate to your project folder inside WSL2 (usually `/mnt/c/Work/Renata/VoiceAI/fastrtc-groq-voice-agent`). Run `uv sync` to automatically download the correct Python version, set up the virtual environment, and install all dependencies (including PyTorch and NeMo from GitHub):
```bash
uv sync
```

### 3. Run the Bot
Use `uv run` to execute the script within the managed virtual environment:
```bash
uv run python src/nemo_magpie_tts.py --remote
```

---

## Configuration Details in `nemo_magpie_tts.py`
- **Language**: Forced to `language="hi"`. This tells Magpie to expect Hindi text and accurately interpret the Devanagari script output by your LangGraph agent.
- **Text Normalization (`apply_TN=True`)**: NeMo will automatically spell out numbers, currencies, and abbreviations into Devanagari words (e.g., `150` -> `एक सौ पचास`), preventing silence or stuttering when the agent outputs numeric IDs.
- **Speaker**: Defaults to `speaker_index=1` (Sofia). Other options: John (0), Aria (2), Jason (3), Leo (4).

## Troubleshooting
- **CUDA Out of Memory**: MagpieTTS (357M parameters) requires ~2-3GB of VRAM. Faster-Whisper (medium) requires ~2GB. Ensure you have at least 6GB of available VRAM. If you run out of memory, change Faster-Whisper's compute type to `int8` or run it on the CPU.
- **Silent Audio**: If the bot returns silent audio chunks, verify that the LLM is correctly outputting Devanagari text. Magpie expects Devanagari when `language="hi"` is specified.
