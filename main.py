import os
import queue
import re
import subprocess
import tempfile
import wave
import warnings as _warnings
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd

try:
    # webrtcvad imports pkg_resources internally for its own version check.
    # pkg_resources is deprecated in newer setuptools and emits a UserWarning.
    # We suppress it by matching the warning message text, since the warning
    # originates from webrtcvad.py itself (not from the pkg_resources module).
    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", message=".*pkg_resources.*")
        _warnings.filterwarnings("ignore", category=DeprecationWarning)
        import webrtcvad as _webrtcvad
    _VAD_AVAILABLE = True
except ImportError:
    _VAD_AVAILABLE = False
    print("[warn] webrtcvad not found — falling back to energy-only VAD")
    print("[warn] Install with: pip install webrtcvad --break-system-packages")

#  CONFIG 

SAMPLE_RATE     = 16000
CHANNELS        = 1
BLOCK_DURATION  = 0.10          # 100 ms blocks → more responsive VAD

SILENCE_TIMEOUT = 1.2           # seconds of silence before committing
MIN_SPEECH_SEC  = 0.4           # skip clips shorter than this (avoids noise hits)

ENERGY_THRESHOLD  = 0.02        # fallback if calibration is skipped
MIN_SPEECH_FRAMES = 2           # loud frames needed before we'll open a recording
PRE_ROLL_FRAMES   = 6           # frames prepended before speech onset

#  CALIBRATION 
# At startup we sample CALIBRATION_SEC seconds of silence, measure the 95th
# percentile RMS, then set the live threshold to that × THRESHOLD_MULTIPLIER.
# This adapts to any mic and room automatically.
CALIBRATION_SEC      = 1.5
THRESHOLD_MULTIPLIER = 3.0   # gap between noise floor and speech; raise if
                              # ambient noise still leaks through
THRESHOLD_FLOOR      = 0.015 # never go below this (near-silent room protection)
THRESHOLD_CEIL       = 0.12  # never go above this (very noisy room protection)

# webrtcvad aggressiveness: 0 = least aggressive, 3 = most aggressive.
# 2 is a good balance — 3 cuts off word onsets before voicing is established.
VAD_AGGRESSIVENESS = 2
# What fraction of 30 ms sub-frames in a block must be speech for the
# block to count as speech (majority vote across the 100 ms window)
VAD_SPEECH_RATIO   = 0.7

BASE_DIR     = Path(__file__).resolve().parent
WHISPER_PATH = str(BASE_DIR / "whisper.cpp" / "build" / "bin" / "whisper-cli")
MODELS_DIR   = BASE_DIR / "whisper.cpp" / "models"

# Pick the best available model automatically (priority: large → tiny)
MODEL_PRIORITY = [
    "ggml-large.bin",
    "ggml-large-v3.bin",
    "ggml-medium.en.bin",
    "ggml-medium.bin",
    "ggml-base.en.bin",
    "ggml-base.bin",
    "ggml-small.en.bin",
    "ggml-small.bin",
    "ggml-tiny.en.bin",
    "ggml-tiny.bin",
]

MODEL_PATH = None
for _model in MODEL_PRIORITY:
    _path = MODELS_DIR / _model
    if _path.exists():
        MODEL_PATH = str(_path)
        break

if MODEL_PATH is None:
    raise FileNotFoundError(
        f"No Whisper model found in {MODELS_DIR}. "
        "Download one with: bash whisper.cpp/models/download-ggml-model.sh base.en"
    )



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


#  VOICE ACTIVITY DETECTION 

# Initialise once at module level so there's no per-frame overhead
_vad = _webrtcvad.Vad(VAD_AGGRESSIVENESS) if _VAD_AVAILABLE else None

# webrtcvad only accepts exactly 10 / 20 / 30 ms frames at 16 kHz
_VAD_FRAME_MS      = 30
_VAD_FRAME_SAMPLES = int(SAMPLE_RATE * _VAD_FRAME_MS / 1000)   # 480 samples
_VAD_FRAME_BYTES   = _VAD_FRAME_SAMPLES * 2                     # int16 → 2 bytes


def _is_voice(audio: np.ndarray) -> bool:
    """
    Return True if the audio block contains human voice.

    Strategy: split the 100 ms block into 30 ms sub-frames, run WebRTC VAD
    on each, and require at least VAD_SPEECH_RATIO of them to be classified
    as speech (majority vote).  Falls back to True (trust energy alone)
    when webrtcvad is not installed.
    """
    if _vad is None:
        return True  # graceful degradation

    pcm = (audio * 32767).astype(np.int16).tobytes()
    total = speech = 0
    for start in range(0, len(pcm) - _VAD_FRAME_BYTES + 1, _VAD_FRAME_BYTES):
        frame = pcm[start : start + _VAD_FRAME_BYTES]
        total += 1
        try:
            if _vad.is_speech(frame, SAMPLE_RATE):
                speech += 1
        except Exception:
            speech += 1  # if VAD errors on a frame, don't penalise it

    if total == 0:
        return False
    ratio = speech / total
    return ratio >= VAD_SPEECH_RATIO



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

def _calibrate(debug: bool = False) -> float:
    """
    Sample CALIBRATION_SEC seconds of ambient audio and return a threshold
    set safely above the measured noise floor.
    """
    block_size = int(SAMPLE_RATE * BLOCK_DURATION)
    n_blocks   = int(CALIBRATION_SEC / BLOCK_DURATION)
    energies   = []

    print(f"Calibrating… stay silent for {CALIBRATION_SEC:.0f}s", flush=True)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=block_size,
    ) as stream:
        for _ in range(n_blocks):
            block, _ = stream.read(block_size)
            rms = float(np.sqrt(np.mean(block.flatten() ** 2)))
            energies.append(rms)

    # Use 95th percentile so the occasional spike doesn't inflate the floor
    noise_floor = float(np.percentile(energies, 95))
    threshold   = noise_floor * THRESHOLD_MULTIPLIER
    threshold   = max(THRESHOLD_FLOOR, min(THRESHOLD_CEIL, threshold))

    if debug:
        print(
            f"[calibrate] noise_floor(p95)={noise_floor:.5f} → "
            f"threshold={threshold:.5f}"
        )
    else:
        print(f"[calibrate] threshold set to {threshold:.4f}")

    return threshold


def _listen_generator(debug: bool = False, cleaned: bool = True):
    # Measure the room noise floor before opening the main mic stream
    energy_threshold = _calibrate(debug=debug)

    audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

    # VAD state
    recording:     list  = []
    is_recording:  bool  = False
    silence_time:  float = 0.0
    speech_frames: int   = 0
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

            above = energy >= energy_threshold
            # Two-gate: energy first (cheap), then WebRTC VAD (only if loud)
            voice = above and _is_voice(audio)

            if debug:
                bar = "█" * int(min(energy * 300, 40))
                tag = "VOICE" if voice else ("loud " if above else "     ")
                print(
                    f"[DBG] E={energy:.4f} T={energy_threshold:.4f} "
                    f"rec={int(is_recording)} sf={speech_frames:02d} "
                    f"sil={silence_time:.2f}s [{tag}] | {bar}"
                )

            #  SPEECH BRANCH 
            if above:
                # Count any loud frame toward the onset accumulator (energy is
                # a reliable onset detector; VAD can reject word-initial frames
                # at high aggressiveness before steady voicing is established)
                speech_frames += 1
                pre_roll.append(audio.copy())

                if not is_recording:
                    # Only open recording once we have enough loud frames AND
                    # the current frame is confirmed voice by WebRTC VAD
                    if speech_frames >= MIN_SPEECH_FRAMES and voice:
                        is_recording = True
                        recording = []
                        for frame in pre_roll:
                            recording.extend(frame)
                        if debug:
                            print("[DBG] ▶ Recording started")
                else:
                    # Already recording — add every loud frame regardless of
                    # VAD (unvoiced consonants, plosives, etc. must be kept)
                    recording.extend(audio)
                    silence_time = 0.0

            #  SILENCE BRANCH 
            else:  # not above threshold
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
    if debug:
        print(f"Using model: {MODEL_PATH}")
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
