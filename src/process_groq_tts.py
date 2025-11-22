import os
import tempfile
import wave
from typing import Any, Generator, Tuple

import numpy as np


def process_groq_tts(
    tts_response: Any,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Process Groq TTS response into a complete audio segment.

    This function reads the entire audio file and yields it as one piece.

    Args:
        tts_response: Groq TTS API response object

    Yields:
        A single tuple of (sample_rate, audio_array) for audio playback
    """
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file_path = temp_file.name
    temp_file.close()

    try:
        tts_response.write_to_file(temp_file_path)

        with wave.open(temp_file_path, "rb") as wf:
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            audio_data = wf.readframes(n_frames)

        audio_array = np.frombuffer(audio_data, dtype=np.int16).reshape(1, -1)
        yield (sample_rate, audio_array)
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)