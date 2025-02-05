import io

import numpy as np
import sounddevice as sd
import soundfile as sf
from pydub import AudioSegment


def read_audio_from_bytes_sf(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    with io.BytesIO(audio_bytes) as audio_buffer:
        data, samplerate = sf.read(audio_buffer)

    return data, samplerate


def read_audio_from_bytes_pydub(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes))

    return np.array(audio.get_array_of_samples()), audio.frame_rate


def play_audio(data: np.ndarray, samplerate: int) -> None:
    sd.play(data, samplerate)
    sd.wait()
