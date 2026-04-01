import gradio as gr
import subprocess
import threading
import os
import signal
import re
import queue
import urllib.request
import urllib.error
import time
import sys

current_process = None


def stop_engine():
    global current_process
    if current_process is not None:
        try:
            if os.name == "nt":
                subprocess.call(
                    ["taskkill", "/F", "/T", "/PID", str(current_process.pid)]
                )
            else:
                os.killpg(os.getpgid(current_process.pid), signal.SIGTERM)
        except Exception as e:
            print(f"Error stopping process: {e}")
        current_process = None
    return "<div style='color:gray; padding:20px; border:1px solid #333; border-radius:8px;'>No engine running. Select an engine and click Launch.</div>"


def start_engine(engine_name):
    global current_process

    # Ensure any previous instance is killed first
    stop_engine()

    script_map = {
        "Kokoro (Native FastRTC)": "src/native_fastrtc_kokorro_tts.py",
        "Magpie (Nemo)": "src/nemo_magpie_tts.py",
        "Veena": "src/veena_tts.py",
        "Orpheus 3B (Custom)": "src/custom_tts_rena.py",
    }

    script_path = script_map.get(engine_name)
    if not script_path:
        yield "<div style='color:red;'>Invalid engine selected.</div>"
        return

    # Render loading message in UI
    loading_html = f"""
    <div style='padding:20px; border:1px solid #333; border-radius:8px; text-align:center;'>
        <h3 style='color:#eab308;'>⏳ Starting {engine_name}...</h3>
        <p>Please wait while AI models are loaded into the GPU. This may take 15-30 seconds.</p>
    </div>
    """
    yield loading_html

    # We use uv to run the underlying application cleanly
    cmd = ["uv", "run", "python", script_path]

    is_shared = "--share" in sys.argv
    if is_shared:
        cmd.append("--remote")

    try:
        # Pre-execution setup to group processes so they can be cleanly killed on Linux
        kwargs = {}
        if os.name != "nt":
            kwargs["preexec_fn"] = os.setsid

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["GRADIO_SERVER_PORT"] = "7861"  # Force the child process to use port 7861

        current_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout so we capture everything
            text=True,
            bufsize=1,  # Line-buffered
            env=env,
            **kwargs,
        )
    except Exception as e:
        yield f"<div style='color:red;'>Error launching process: {e}</div>"
        return

    url = "http://127.0.0.1:7861"
    public_url = None

    start_time = time.time()

    def enqueue_output(out, q):
        for line in iter(out.readline, ""):
            q.put(line)
        out.close()

    q = queue.Queue()
    t = threading.Thread(target=enqueue_output, args=(current_process.stdout, q))
    t.daemon = True
    t.start()

    logs = []
    engine_ready = False

    public_url_pattern = re.compile(
        r"(https://[a-zA-Z0-9-]+\.(?:ngrok-free\.app|gradio\.live|ngrok\.io|ngrok\.app|loca\.lt))"
    )

    engine_ready_time = None

    # Poll until we can connect to the Gradio app on port 7861
    while True:
        # 1. Grab all available log lines from queue
        while True:
            try:
                line = q.get_nowait()
                print(f"[{engine_name}] {line}", end="")
                logs.append(line)

                # If we're shared, actively search for the public URL in the logs
                if is_shared and not public_url:
                    match = public_url_pattern.search(line)
                    if match:
                        public_url = match.group(1)
                        print(f"[{engine_name}] Found public URL: {public_url}")
            except queue.Empty:
                break

        # Show terminal progress in UI for long-loading models
        if len(logs) > 0:
            recent_logs = "<br>".join(logs[-8:])
            yield f"""
            <div style='padding:20px; border:1px solid #333; border-radius:8px;'>
                <h3 style='color:#eab308;'>⏳ Loading {engine_name}...</h3>
                <div style='font-family:monospace; font-size:12px; color:#aaa; margin-top:10px;'>{recent_logs}</div>
            </div>
            """

        # 2. Check if process crashed
        if current_process.poll() is not None:
            err_logs = "<br>".join(logs[-15:])
            yield f"""
            <div style='padding:20px; border:1px solid #ef4444; border-radius:8px;'>
                <h3 style='color:#ef4444;'>❌ Engine Crashed</h3>
                <div style='font-family:monospace; font-size:12px; color:#aaa; margin-top:10px;'>{err_logs}</div>
            </div>
            """
            return

        # 3. Try connecting to the webserver if not yet ready
        if not engine_ready:
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=1) as response:
                    if response.getcode() == 200:
                        engine_ready = True
                        engine_ready_time = time.time()
            except Exception:
                pass

        # If ready, check if we need to wait for public URL
        if engine_ready:
            if not is_shared:
                break
            elif is_shared and public_url:
                break
            # If shared but no public url yet, we keep looping until timeout!
            if engine_ready_time is not None and time.time() - engine_ready_time > 30:
                print(f"[{engine_name}] Timeout waiting for public URL generation.")
                break  # We give up on the public URL and just load the local one

        # 4. Timeout (approx 300 seconds to allow huge models to download and load)
        if time.time() - start_time > 300:
            stop_engine()
            yield f"<div style='color:red; padding:20px;'>Timeout waiting for engine. It took too long to start (over 5 minutes).</div>"
            return

        time.sleep(1)

    if engine_ready:
        final_url = public_url if (is_shared and public_url) else url

        # Success! Render iframe pointing to the newly opened port
        # VERY IMPORTANT: allow="microphone; camera; autoplay" is required for WebRTC audio
        iframe_html = f'<iframe src="{final_url}" allow="microphone; camera; autoplay" width="100%" height="800px" style="border:none; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.5);"></iframe>'

        if is_shared and not public_url:
            iframe_html = (
                f"<div style='color:orange;'>Warning: Started in share mode, but couldn't find public URL for the engine. Attempting to use local URL.</div>"
                + iframe_html
            )

        yield iframe_html

        # Keep routing background logs to main console
        def consume_rest():
            while current_process and current_process.poll() is None:
                try:
                    line = q.get(timeout=1)
                    print(f"[{engine_name}] {line}", end="")
                except queue.Empty:
                    pass

        threading.Thread(target=consume_rest, daemon=True).start()
    else:
        yield "<div style='color:red;'>Failed to extract local URL.</div>"


theme = gr.themes.Monochrome()

js_dark = """
function() {
    document.documentElement.classList.add('dark');
}
"""

with gr.Blocks(theme=theme, title="Renata Voice AI Testing Hub", js=js_dark) as demo:
    gr.Markdown("# 🎙️ Renata Voice AI - Testing Hub")
    gr.Markdown(
        "Select a TTS engine to test. The hub automatically manages GPU VRAM by ensuring only one heavy model pipeline runs at a time."
    )

    with gr.Row():
        with gr.Column(scale=1):
            engine_radio = gr.Radio(
                choices=[
                    "Kokoro (Native FastRTC)",
                    "Magpie (Nemo)",
                    "Veena",
                    "Orpheus 3B (Custom)",
                ],
                value="Kokoro (Native FastRTC)",
                label="Select TTS Engine",
            )
            launch_btn = gr.Button("🚀 Launch Engine", variant="primary")
            stop_btn = gr.Button("🛑 Stop / Clear VRAM", variant="stop")

            gr.Markdown("### Instructions")
            gr.Markdown("- Choose an engine and click **Launch**.")
            gr.Markdown(
                "- **Wait 15-45s** for the models (Whisper, LLM, TTS) to load into VRAM. Terminal logs will stream below."
            )
            gr.Markdown(
                "- To switch engines, simply select a new one and click **Launch**. The previous engine will be cleanly killed automatically."
            )

        with gr.Column(scale=3):
            iframe_container = gr.HTML(
                "<div style='color:gray; padding:20px; border:1px solid #333; border-radius:8px;'>No engine running. Select an engine and click Launch.</div>"
            )

    launch_btn.click(fn=start_engine, inputs=[engine_radio], outputs=[iframe_container])

    stop_btn.click(fn=stop_engine, inputs=[], outputs=[iframe_container])

if __name__ == "__main__":
    is_shared_hub = "--share" in sys.argv
    # Host on 127.0.0.1 so it provides a clickable link in terminal
    demo.launch(server_name="127.0.0.1", server_port=7860, share=is_shared_hub)
