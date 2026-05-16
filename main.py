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

#  CONFIG 

SAMPLE_RATE     = 16000
CHANNELS        = 1
BLOCK_DURATION  = 0.10          # 100 ms blocks → more responsive VAD

SILENCE_TIMEOUT = 1.2           # seconds of silence before committing
MIN_SPEECH_SEC  = 0.4           # skip clips shorter than this (avoids noise hits)

ENERGY_THRESHOLD  = 0.02        # RMS amplitude threshold — tune to your mic
MIN_SPEECH_FRAMES = 4           # consecutive loud frames needed to open recording
PRE_ROLL_FRAMES   = 6           # frames prepended before speech onset

BASE_DIR     = Path(__file__).resolve().parent
WHISPER_PATH = str(BASE_DIR / "whisper.cpp" / "build" / "bin" / "whisper-cli")
MODEL_PATH   = str(BASE_DIR / "whisper.cpp" / "models" / "ggml-tiny.en.bin")

#  TEXT CLEANING 

# Whisper noise tokens: (Music), [BLANK_AUDIO], [noise], etc. — anywhere
_NOISE_RE  = re.compile(r"[\[\(][^\]\)]{0,40}[\]\)]", re.IGNORECASE)
# Leading dashes / whitespace whisper sometimes emits
_LEADER_RE = re.compile(r"^[\s\-–—]+")
# Collapse internal whitespace runs
_SPACE_RE  = re.compile(r"\s{2,}")


def _clean_text(text: str) -> str:
    text = _NOISE_RE.sub("", text)
    text = _LEADER_RE.sub("", text)
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


#  AUDIO HELPERS 

def _frames_to_wav(frames: list, path: str) -> None:
    """Normalize and write a float32 frame list to a 16-bit WAV file."""
    audio = np.array(frames, dtype=np.float32)
    peak  = np.abs(audio).max()
    if peak > 1e-6:
        audio = audio * min(1.0, 0.9 / peak)   # normalise; never clip
    int_audio = (audio * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(int_audio.tobytes())


def _transcribe_wav(wav_path: str, cleaned: bool = True) -> str:
    cmd = [
        WHISPER_PATH,
        "-m", MODEL_PATH,
        "-f", wav_path,
        "-nt",          # no timestamps
        "-l", "en",     # force language (helps tiny model)
        "-t", "2",      # threads — safe on most Android devices
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
    except subprocess.TimeoutExpired:
        print("[whisper] Timeout — skipping segment")
        return ""

    if result.returncode != 0:
        err = result.stderr.strip()
        if err:
            print(f"[whisper error] {err}")
        return ""

    raw = result.stdout.strip()
    return _clean_text(raw) if cleaned else raw


#  CORE GENERATOR 

def _listen_generator(debug: bool = False, cleaned: bool = True):
    audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

    # VAD state
    recording:     list  = []
    is_recording:  bool  = False
    silence_time:  float = 0.0
    speech_frames: int   = 0
    speech_time:   float = 0.0
    pre_roll               = deque(maxlen=PRE_ROLL_FRAMES)

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}")
        audio_queue.put(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=int(SAMPLE_RATE * BLOCK_DURATION),
        callback=audio_callback,
    ):
        while True:
            # Timeout allows KeyboardInterrupt to surface cleanly
            try:
                chunk = audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            audio  = chunk.flatten()
            energy = float(np.sqrt(np.mean(audio ** 2)))

            if debug:
                bar = "█" * int(min(energy * 300, 40))
                print(
                    f"[DBG] E={energy:.4f} rec={int(is_recording)} "
                    f"sf={speech_frames:02d} sil={silence_time:.2f}s | {bar}"
                )

            above = energy >= ENERGY_THRESHOLD

            #  SPEECH BRANCH 
            if above:
                speech_frames += 1
                # Always update pre-roll (captures onset context)
                pre_roll.append(audio.copy())

                if speech_frames >= MIN_SPEECH_FRAMES:
                    if not is_recording:
                        #  Open recording 
                        is_recording = True
                        speech_time  = 0.0
                        # Pre-roll already contains the current frame, so
                        # just drain it — no separate recording.extend(audio)
                        recording = []
                        for frame in pre_roll:
                            recording.extend(frame)
                        if debug:
                            print("[DBG] ▶ Recording started")
                    else:
                        #  Continue recording 
                        recording.extend(audio)
                        speech_time += BLOCK_DURATION

                    silence_time = 0.0

            #  SILENCE BRANCH 
            else:
                # Gradual decay prevents a single quiet frame from resetting
                # the detector mid-word
                speech_frames = max(0, speech_frames - 1)
                pre_roll.append(audio.copy())

                if is_recording:
                    recording.extend(audio)          # keep trailing silence
                    silence_time += BLOCK_DURATION
                    if debug:
                        print(f"[DBG]   silence_time={silence_time:.2f}s")

            #  COMMIT BRANCH 
            if is_recording and silence_time >= SILENCE_TIMEOUT:
                total_sec = len(recording) / SAMPLE_RATE
                if debug:
                    print(f"[DBG] ⏹ Committing {total_sec:.2f}s of audio")

                text = ""
                if total_sec >= MIN_SPEECH_SEC:
                    fd, wav_path = tempfile.mkstemp(suffix=".wav")
                    os.close(fd)
                    try:
                        _frames_to_wav(recording, wav_path)
                        text = _transcribe_wav(wav_path, cleaned)
                    finally:
                        try:
                            os.remove(wav_path)
                        except FileNotFoundError:
                            pass
                elif debug:
                    print(f"[DBG] Skipped — too short ({total_sec:.2f}s)")

                if debug:
                    print(f"[DBG] transcript={text!r}")

                # Reset all state
                recording     = []
                is_recording  = False
                silence_time  = 0.0
                speech_frames = 0
                speech_time   = 0.0
                pre_roll.clear()

                if text:
                    yield text


#  PUBLIC API 

def listen(
    as_module: bool = True,
    once:      bool = False,
    debug:     bool = False,
    cleaned:   bool = True,
):
    """
    as_module=True  → return a generator, or a single string if once=True
    as_module=False → print transcripts to stdout until Ctrl-C
    once=True       → stop after the first transcript
    """
    gen = _listen_generator(debug=debug, cleaned=cleaned)

    if as_module:
        return next(gen, "") if once else gen

    print("Listening… (Ctrl-C to stop)\n")
    try:
        if once:
            text = next(gen, "")
            if text:
                print(text)
        else:
            for text in gen:
                print(text)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    listen(as_module=False, once=False, debug=True, cleaned=False)
