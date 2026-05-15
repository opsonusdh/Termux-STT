import os
import queue
import re
import subprocess
import tempfile
import wave
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd


# CONFIG

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_DURATION = 0.25
SILENCE_TIMEOUT = 1.0

ENERGY_THRESHOLD = 0.05
MIN_SPEECH_FRAMES = 3
PRE_ROLL_FRAMES = 4

BASE_DIR = Path(__file__).resolve().parent

WHISPER_PATH = str(
    BASE_DIR / "whisper.cpp" / "build" / "bin" / "whisper-cli"
)

MODEL_PATH = str(
    BASE_DIR / "whisper.cpp" / "models" / "ggml-tiny.en.bin"
)

NOISE_RE = re.compile(r"^\s*\([^)]*\)\s*")


def _clean_text(text: str) -> str:
    text = text.strip()
    text = NOISE_RE.sub("", text, count=1).strip()
    return text


def _transcribe_wav(wav_path: str, cleaned: bool = True) -> str:
    result = subprocess.run(
        [
            WHISPER_PATH,
            "-m",
            MODEL_PATH,
            "-f",
            wav_path,
            "-nt",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        err = result.stderr.strip()
        if err:
            print(err)
        return ""

    raw_text = result.stdout.strip()
    return _clean_text(raw_text) if cleaned else raw_text


def _listen_generator(debug: bool = False, cleaned: bool = True):
    audio_queue = queue.Queue()

    recording = []
    is_recording = False
    silence_time = 0.0
    speech_frames = 0
    pre_roll = deque(maxlen=PRE_ROLL_FRAMES)

    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        audio_queue.put(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=int(SAMPLE_RATE * BLOCK_DURATION),
        callback=audio_callback,
    ):
        while True:
            chunk = audio_queue.get()
            audio = chunk.flatten()

            pre_roll.append(audio.copy())

            energy = float(np.sqrt(np.mean(audio**2)))

            if debug:
                print(
                    f"[DEBUG] energy={energy:.5f} "
                    f"recording={is_recording} "
                    f"speech_frames={speech_frames} "
                    f"silence_time={silence_time:.2f}"
                )

            # speech detected
            if energy >= ENERGY_THRESHOLD:
                speech_frames += 1

                if speech_frames >= MIN_SPEECH_FRAMES:
                    if not is_recording:
                        is_recording = True
                        recording = []

                        if debug:
                            print("[DEBUG] Recording started")

                        for frame in pre_roll:
                            recording.extend(frame)

                    recording.extend(audio)
                    silence_time = 0.0

                continue

            # silence
            speech_frames = 0

            if is_recording:
                recording.extend(audio)
                silence_time += BLOCK_DURATION

                if debug:
                    print(f"[DEBUG] silence_time={silence_time:.2f}s")

                if silence_time >= SILENCE_TIMEOUT:
                    if debug:
                        print("[DEBUG] Transcribing audio...")

                    with tempfile.NamedTemporaryFile(
                        suffix=".wav",
                        delete=False,
                    ) as f:
                        wav_path = f.name

                    try:
                        with wave.open(wav_path, "wb") as wf:
                            wf.setnchannels(CHANNELS)
                            wf.setsampwidth(2)
                            wf.setframerate(SAMPLE_RATE)

                            int_audio = (
                                np.array(recording) * 32767
                            ).astype(np.int16)

                            wf.writeframes(int_audio.tobytes())

                        text = _transcribe_wav(wav_path, cleaned)

                    finally:
                        try:
                            os.remove(wav_path)
                        except FileNotFoundError:
                            pass

                    recording = []
                    is_recording = False
                    silence_time = 0.0
                    pre_roll.clear()

                    if debug:
                        print(f"[DEBUG] transcript={text!r}")

                    if text:
                        yield text


def listen(as_module: bool = True, once: bool = False, debug: bool = False, cleaned: bool = True):
    """
    as_module=True  -> return generator / single text for use in other projects
    as_module=False -> print transcripts in terminal
    once=True       -> stop after first transcript
    """
    generator = _listen_generator(debug=debug, cleaned=cleaned)

    if as_module:
        if once:
            return next(generator, "")
        return generator

    print("Listening...")

    try:
        if once:
            text = next(generator, "")
            if text:
                print(text)
            return text

        for text in generator:
            print(text)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    listen(as_module=False, once=False, debug=True, cleaned=False)