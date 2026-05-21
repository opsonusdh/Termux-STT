import atexit
import http.client
import io
import json
import os
import queue
import re
import subprocess
import tempfile
import time
import wave
import warnings as _warnings
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd

try:
    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", message=".*pkg_resources.*")
        _warnings.filterwarnings("ignore", category=DeprecationWarning)
        import webrtcvad as _webrtcvad
    _VAD_AVAILABLE = True
except ImportError:
    _VAD_AVAILABLE = False
    print("[warn] webrtcvad not found — falling back to energy-only VAD")
    print("[warn] Install with: pip install webrtcvad --break-system-packages")

#  GROQ CLIENT (optional)

try:
    from openai import OpenAI as _OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    print("[warn] openai package not found — Groq backend disabled")
    print("[warn] Install with: pip install openai --break-system-packages")

#  CONFIG

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_DURATION = 0.10  # 100 ms blocks

# Normal endpointing: wait for silence before committing.
SILENCE_TIMEOUT = 1.5

# When the stream grows too long, we do NOT cut mid-word.
# We only arm a pending commit and then wait for the next silence edge.
MAX_RECORDING_SEC_STREAM = 20.0
FORCE_COMMIT_SILENCE_TIMEOUT = 0.35

# Audio queue: bounded to prevent unbounded RAM growth during realtime processing.
# If transcription stalls (CPU throttle, server delay, long speech), queue won't grow infinitely.
# Drops oldest frames when full to preserve freshest audio and maintain low latency.
MAX_AUDIO_QUEUE_SIZE = 5

MIN_SPEECH_SEC = 0.5
ENERGY_THRESHOLD = 0.02
MIN_SPEECH_FRAMES = 1
PRE_ROLL_FRAMES = 6

# Adaptive thread scaling: use half available cores, capped at 2-8
WHISPER_THREADS = min(max(2, (os.cpu_count() or 2) // 2), 8)

# CALIBRATION
CALIBRATION_SEC = 1.5
THRESHOLD_MULTIPLIER = 3.0
THRESHOLD_FLOOR = 0.015
THRESHOLD_CEIL = 0.2

# webrtcvad
VAD_AGGRESSIVENESS = 2
VAD_SPEECH_RATIO = 0.7

# Whisper-server (persistent process — eliminates per-clip model-load overhead)
_SERVER_HOST = "127.0.0.1"
_SERVER_PORT = 8178

# Groq
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "distil-whisper-large-v3-en"
GROQ_RESPONSE_FORMAT = "text"
GROQ_LANGUAGE = "en"

#  PATHS
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "whisper.cpp" / "models"

MODEL_PRIORITY = [
    "ggml-large.bin", "ggml-large-v3.bin",
    "ggml-medium.en.bin", "ggml-medium.bin",
    "ggml-base.en.bin",  "ggml-base.bin",
    "ggml-small.en.bin", "ggml-small.bin",
    "ggml-tiny.en.bin",  "ggml-tiny.bin",
]

# Resolved lazily on first local-whisper use. never touched if Groq handles everything.
WHISPER_PATH: str | None = None
MODEL_PATH:   str | None = None
_local_whisper_resolved: bool = False

#  TEXT CLEANING

_NOISE_RE = re.compile(r"[\[\(][^\]\)]{0,40}[\]\)]", re.IGNORECASE)
_LEADER_RE = re.compile(r"^[\s\-–—]+")
_SPACE_RE = re.compile(r"\s{2,}")


def _clean_text(text: str) -> str:
    text = _NOISE_RE.sub("", text)
    text = _LEADER_RE.sub("", text)
    text = _SPACE_RE.sub(" ", text)
    return text.strip()

#  LAZY LOCAL WHISPER RESOLVER

def _resolve_local_whisper() -> bool:
    global WHISPER_PATH, MODEL_PATH, _local_whisper_resolved
    if _local_whisper_resolved:
        return WHISPER_PATH is not None and MODEL_PATH is not None

    _local_whisper_resolved = True  # mark regardless — don't scan twice on failure

    for _bin in ("whisper-cli", "main"):
        _p = BASE_DIR / "whisper.cpp" / "build" / "bin" / _bin
        if _p.exists():
            WHISPER_PATH = str(_p)
            break

    for _model in MODEL_PRIORITY:
        _path = MODELS_DIR / _model
        if _path.exists():
            MODEL_PATH = str(_path)
            break

    if WHISPER_PATH is None:
        print(
            "[warn] whisper binary not found in whisper.cpp/build/bin/. "
            "Looked for: whisper-cli, main"
        )
    else:
        print(f"[local] binary  → {WHISPER_PATH}")

    if MODEL_PATH is None:
        print(
            f"[warn] no whisper model found in {MODELS_DIR}. "
            "Download one with: bash whisper.cpp/models/download-ggml-model.sh base.en"
        )
    else:
        print(f"[local] model   → {MODEL_PATH}")

    return WHISPER_PATH is not None and MODEL_PATH is not None


#  VAD

_vad = _webrtcvad.Vad(VAD_AGGRESSIVENESS) if _VAD_AVAILABLE else None

_VAD_FRAME_MS = 30
_VAD_FRAME_SAMPLES = int(SAMPLE_RATE * _VAD_FRAME_MS / 1000)  # 480 samples
_VAD_FRAME_BYTES = _VAD_FRAME_SAMPLES * 2  # int16 → 2 bytes


def _is_voice(audio: np.ndarray) -> bool:
    if _vad is None:
        return True
    pcm = (audio * 32767).astype(np.int16).tobytes()
    total = speech = 0
    for start in range(0, len(pcm) - _VAD_FRAME_BYTES + 1, _VAD_FRAME_BYTES):
        frame = pcm[start : start + _VAD_FRAME_BYTES]
        total += 1
        try:
            if _vad.is_speech(frame, SAMPLE_RATE):
                speech += 1
        except Exception:
            speech += 1
    return (speech / total) >= VAD_SPEECH_RATIO if total else False

#  AUDIO / WAV

def _frames_to_wav_bytes(frames: list) -> bytes:
    """Convert a list of float32 numpy chunks to WAV bytes held in memory."""
    if not frames:
        audio = np.array([], dtype=np.float32)
    else:
        audio = np.concatenate(frames, dtype=np.float32)

    peak = np.abs(audio).max() if audio.size else 0.0
    if peak > 1e-6:
        audio *= min(1.0, 0.9 / peak)

    int_audio = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(int_audio.tobytes())
    return buf.getvalue()

#  GROQ TRANSCRIBER

class _GroqTranscriber:
    #  tunables 
    _KEYS_FILENAME      = "api.keys"
    _MAX_NONRL_FAILURES = 3       # consecutive non-RL errors before suspension
    _NONRL_SUSPEND_SEC  = 60.0    # how long to suspend after max non-RL errors
    _POOL_RESET_SLEEP   = 30.0    # cool-down when ALL keys are rate-limited
    _RL_RETRY_SLEEP     = 3.0     # brief pause between per-key 429 retries

    def __init__(self):
        self._keys:        list[str]  = []   # full pool loaded from file/env
        self._bad_keys:    set[str]   = set()
        self._cursor:      int        = 0

        self._enabled:     bool       = False
        self._nonrl_fails: int        = 0
        self._suspend_until: float    = 0.0  # monotonic epoch; 0 = not suspended

    #  key-file loading 

    @classmethod
    def _load_keys_from_file(cls, debug: bool = False) -> list[str]:
        path = BASE_DIR / cls._KEYS_FILENAME
        if not path.exists():
            if debug:
                print(f"[groq] {cls._KEYS_FILENAME} not found at {path}")
            return []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            keys  = [
                ln.strip() for ln in lines
                if ln.strip() and not ln.strip().startswith("#")
            ]
            if debug:
                print(f"[groq] loaded {len(keys)} key(s) from {cls._KEYS_FILENAME}")
            return keys
        except OSError as e:
            print(f"[groq] could not read {cls._KEYS_FILENAME}: {e}")
            return []

    #  key rotation 

    def _available_keys(self) -> list[str]:
        return [k for k in self._keys if k not in self._bad_keys]

    def _next_key(self) -> str | None:
        pool = self._available_keys()
        if not pool:
            return None
        key = pool[self._cursor % len(pool)]
        self._cursor += 1
        return key

    def _mark_bad(self, key: str) -> None:
        self._bad_keys.add(key)

    def _reset_pool(self) -> None:
        self._bad_keys.clear()
        self._cursor = 0

    def _key_count(self) -> int:
        return len(self._available_keys())

    #  setup 

    def setup(self, debug: bool = False) -> bool:
        """
        Load keys and verify at least one is present.
        Call once before the listen loop; returns True when Groq is usable.
        """
        if not _OPENAI_AVAILABLE:
            return False

        # 1. Try key file
        self._keys = self._load_keys_from_file(debug=debug)

        # 2. Fall back to env-var
        if not self._keys:
            env_key = os.environ.get("GROQ_API_KEY", "").strip()
            if env_key:
                self._keys = [env_key]
                if debug:
                    print("[groq] using key from GROQ_API_KEY env var")
            else:
                if debug:
                    print(
                        "[groq] no keys found in api.keys or GROQ_API_KEY. "
                        "Groq backend disabled"
                    )
                return False

        self._bad_keys.clear()
        self._cursor      = 0
        self._enabled     = True
        self._nonrl_fails = 0
        self._suspend_until = 0.0

        if debug:
            print(
                f"[groq] ready  keys={len(self._keys)}  "
                f"model={GROQ_MODEL}  lang={GROQ_LANGUAGE or 'auto'}"
            )
        return True

    #  availability 

    @property
    def available(self) -> bool:
        """True when Groq has keys loaded and is not suspended."""
        return (
            self._enabled
            and bool(self._keys)
            and time.monotonic() >= self._suspend_until
        )

    #  transcription 

    def transcribe(self, wav_bytes: bytes, cleaned: bool = True) -> str:
        if not self.available:
            return ""

        # How many distinct keys can we try this call
        attempts = max(1, len(self._keys))

        for attempt in range(attempts):
            key = self._next_key()
            if key is None:
                # If all keys currently bad then reset pool and cool down
                print(
                    f"[groq] all {len(self._keys)} key(s) rate-limited. "
                    f"Resetting pool and cooling down {self._POOL_RESET_SLEEP:.0f}s…"
                )
                self._reset_pool()
                time.sleep(self._POOL_RESET_SLEEP)
                key = self._next_key()
                if key is None:
                    return ""

            audio_file      = io.BytesIO(wav_bytes)
            audio_file.name = "audio.wav"

            try:
                client = _OpenAI(base_url=GROQ_BASE_URL, api_key=key)
                kwargs: dict = dict(
                    model=GROQ_MODEL,
                    file=audio_file,
                    response_format=GROQ_RESPONSE_FORMAT,
                )
                if GROQ_LANGUAGE:
                    kwargs["language"] = GROQ_LANGUAGE

                result = client.audio.transcriptions.create(**kwargs)

                text = result if isinstance(result, str) else (result.text or "")
                text = text.strip()

                # Success. clear non-RL failure counter
                self._nonrl_fails = 0
                return _clean_text(text) if cleaned else text

            except Exception as e:
                err = str(e)
                is_rate_limit = any(
                    tok in err for tok in ("429", "rate_limit", "RATE_LIMIT", "rate limit")
                )

                if is_rate_limit:
                    self._mark_bad(key)
                    remaining = self._key_count()
                    print(
                        f"[groq] 429 on key …{key[-6:]}. "
                        f"Keys still available: {remaining}."
                    )
                    if remaining > 0 and attempt < attempts - 1:
                        time.sleep(self._RL_RETRY_SLEEP)
                        continue
                    return ""

                # Non-rate-limit error
                self._nonrl_fails += 1
                print(
                    f"[groq] error (attempt {attempt + 1}/{attempts}): "
                    f"{err[:120]}"
                )
                if self._nonrl_fails >= self._MAX_NONRL_FAILURES:
                    self._suspend_until = time.monotonic() + self._NONRL_SUSPEND_SEC
                    print(
                        f"[groq] {self._MAX_NONRL_FAILURES} consecutive non-RL errors — "
                        f"suspending for {self._NONRL_SUSPEND_SEC:.0f}s, "
                        f"falling back to local whisper"
                    )
                return ""

        return ""


_groq = _GroqTranscriber()


#  PERSISTENT WHISPER SERVER

class _WhisperServer:
    """Manages a persistent whisper-server subprocess."""

    def __init__(self):
        self._proc = None
        self._ready = False
        _bin = BASE_DIR / "whisper.cpp" / "build" / "bin" / "whisper-server"
        self._bin = str(_bin) if _bin.exists() else None

    @property
    def available(self) -> bool:
        return self._bin is not None

    def start(self, debug: bool = False, cleaned: bool = True) -> bool:
        if not self.available:
            return False
        if self.healthy():
            return True
        
        if MODEL_PATH is None:
            print("[server] MODEL_PATH not resolved — cannot start whisper-server")
            return False
        if cleaned:
            cmd = [
                self._bin,
                "-m", MODEL_PATH,
                "-t", str(WHISPER_THREADS),
                "--host", _SERVER_HOST,
                "--port", str(_SERVER_PORT),
                "-l", "en", "-sns",
            ]
        else:
            cmd = [
                self._bin,
                "-m", MODEL_PATH,
                "-t", str(WHISPER_THREADS),
                "--host", _SERVER_HOST,
                "--port", str(_SERVER_PORT),
                "-l", "en",
            ]
        if debug:
            print(f"[server] starting: {' '.join(cmd)}")

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as e:
            print(f"[server] failed to start process: {e}")
            return False

        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                print("[server] process exited unexpectedly during startup")
                return False
            try:
                conn = http.client.HTTPConnection(_SERVER_HOST, _SERVER_PORT, timeout=1)
                conn.request("GET", "/health")
                resp = conn.getresponse()
                resp.read()
                conn.close()
                if resp.status == 200:
                    self._ready = True
                    print("[server] whisper-server ready")
                    return True
            except Exception:
                pass
            time.sleep(0.5)

        print("[server] timed out waiting for ready — falling back to subprocess mode")
        self.stop()
        return False

    def transcribe(self, wav_bytes: bytes, cleaned: bool = True) -> str:
        """POST wav_bytes to /inference, return transcript string."""
        boundary = "termuxsttbdy"
        header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode("ascii")
        footer = f"\r\n--{boundary}--\r\n".encode("ascii")
        body = header + wav_bytes + footer

        try:
            conn = http.client.HTTPConnection(_SERVER_HOST, _SERVER_PORT, timeout=30)
            conn.request(
                "POST", "/inference",
                body=body,
                headers={
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "Content-Length": str(len(body)),
                },
            )
            resp = conn.getresponse()
            raw = resp.read().decode("utf-8")
            conn.close()
            if resp.status == 200:
                text = json.loads(raw).get("text", "").strip()
                return _clean_text(text) if cleaned else text
            print(f"[server] HTTP {resp.status}: {raw[:120]}")
        except Exception as e:
            print(f"[server] request failed: {e} — marking unhealthy")
            self._ready = False

        return ""

    def stop(self):
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                pass
            self._proc = None
        self._ready = False

    def healthy(self) -> bool:
        return (
            self._ready
            and self._proc is not None
            and self._proc.poll() is None
        )


_server = _WhisperServer()
atexit.register(_server.stop)

#  TRANSCRIPTION ENTRY POINT

def _transcribe(frames: list, cleaned: bool = True) -> str:
    wav_bytes = _frames_to_wav_bytes(frames)

    #  1. Groq 
    if _groq.available:
        text = _groq.transcribe(wav_bytes, cleaned)
        if text:
            return text
            
    if not _resolve_local_whisper():
        print("[error] local whisper unavailable and Groq failed — no transcript")
        return ""

    #  2. Local whisper-server 
    if not _server.healthy() and _server.available:
        _server.start()

    if _server.healthy():
        return _server.transcribe(wav_bytes, cleaned)

    #  3. Subprocess fallback 
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    try:
        os.close(fd)
        with open(wav_path, "wb") as f:
            f.write(wav_bytes)
        return _transcribe_subprocess(wav_path, cleaned)
    finally:
        try:
            os.remove(wav_path)
        except FileNotFoundError:
            pass


def _transcribe_subprocess(wav_path: str, cleaned: bool = True) -> str:
    """Original subprocess path. kept as fallback."""
    cmd = [
        WHISPER_PATH,
        "-m", MODEL_PATH,
        "-f", wav_path,
        "-nt",
        "-l", "en",
        "-t", str(WHISPER_THREADS),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
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

#  CALIBRATION

def _calibrate(debug: bool = False) -> float:
    block_size = int(SAMPLE_RATE * BLOCK_DURATION)
    n_blocks = int(CALIBRATION_SEC / BLOCK_DURATION)
    energies = []

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

    noise_floor = float(np.median(energies))
    threshold = noise_floor * THRESHOLD_MULTIPLIER
    threshold = max(THRESHOLD_FLOOR, min(THRESHOLD_CEIL, threshold))

    if debug:
        print(f"[calibrate] noise_floor(p95)={noise_floor:.5f} → threshold={threshold:.5f}")
    else:
        print(f"[calibrate] threshold set to {threshold:.4f}")

    return threshold

#  CORE GENERATOR

_SILENCE_THRESHOLDS = [
    (3.0, 1.0),
    (10.0, 1.5),
    (MAX_RECORDING_SEC_STREAM, 1.8),
]


def _silence_timeout_for_clip(clip_elapsed: float, pending_force_commit: bool) -> float:
    if pending_force_commit:
        return FORCE_COMMIT_SILENCE_TIMEOUT
    for threshold, timeout in _SILENCE_THRESHOLDS:
        if clip_elapsed < threshold:
            return timeout
    return 1.8


def _listen_generator(
    debug: bool = False,
    cleaned: bool = True,
    energy_threshold=None,
):
    if energy_threshold is None:
        energy_threshold = _calibrate(debug=debug)

    audio_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=MAX_AUDIO_QUEUE_SIZE)

    _is_voice_func = _is_voice

    recording: list = []
    recording_samples: int = 0
    is_recording: bool = False
    silence_time: float = 0.0
    clip_elapsed: float = 0.0
    speech_frames: int = 0
    pending_force_commit: bool = False
    pre_roll = deque(maxlen=PRE_ROLL_FRAMES)

    def reset_state():
        nonlocal recording, recording_samples, is_recording, silence_time
        nonlocal clip_elapsed, speech_frames, pending_force_commit
        recording = []
        recording_samples = 0
        is_recording = False
        silence_time = 0.0
        clip_elapsed = 0.0
        speech_frames = 0
        pending_force_commit = False
        pre_roll.clear()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}")
        if audio_queue.full():
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                pass
        audio_queue.put_nowait(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=int(SAMPLE_RATE * BLOCK_DURATION),
        callback=audio_callback,
    ):
        while True:
            try:
                chunk = audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[error] audio queue: {e}")
                reset_state()
                continue

            try:
                audio = chunk.flatten()
                energy = float(np.sqrt(np.mean(audio ** 2)))
                above = energy >= energy_threshold
                voice = above and _is_voice_func(audio)

                if debug:
                    bar = "█" * int(min(energy * 300, 40))
                    tag = "VOICE" if voice else ("loud " if above else "     ")
                    backend = (
                        "groq"
                        if _groq.available
                        else ("server" if _server.healthy() else "subproc")
                    )
                    print(
                        f"[DBG] E={energy:.4f} T={energy_threshold:.4f} "
                        f"rec={int(is_recording)} sf={speech_frames:02d} "
                        f"sil={silence_time:.2f}s el={clip_elapsed:.2f}s "
                        f"pend={int(pending_force_commit)} [{tag}] "
                        f"backend={backend} | {bar}"
                    )

                #  SPEECH BRANCH 
                if above:
                    speech_frames += 1
                    pre_roll.append(audio)

                    if not is_recording:
                        if speech_frames >= MIN_SPEECH_FRAMES and voice:
                            is_recording = True
                            recording = list(pre_roll)
                            recording_samples = sum(len(f) for f in recording)
                            clip_elapsed = recording_samples / SAMPLE_RATE
                            if debug:
                                print("[DBG] ▶ Recording started")
                    else:
                        recording.append(audio)
                        recording_samples += len(audio)
                        clip_elapsed += BLOCK_DURATION
                        silence_time = 0.0

                #  SILENCE BRANCH 
                else:
                    speech_frames = max(0, speech_frames - 1)
                    pre_roll.append(audio)

                    if is_recording:
                        recording.append(audio)
                        recording_samples += len(audio)
                        clip_elapsed += BLOCK_DURATION
                        silence_time += BLOCK_DURATION

                        if debug:
                            print(f"[DBG] silence_time={silence_time:.2f}s")

                #  SOFT CAP ARMING 
                if is_recording and clip_elapsed >= MAX_RECORDING_SEC_STREAM:
                    pending_force_commit = True

                #  COMMIT BRANCH 
                current_silence_timeout = _silence_timeout_for_clip(
                    clip_elapsed=clip_elapsed,
                    pending_force_commit=pending_force_commit,
                )

                should_commit = is_recording and silence_time >= current_silence_timeout

                if should_commit:
                    total_sec = recording_samples / SAMPLE_RATE
                    if debug:
                        print(f"[DBG] Committing {total_sec:.2f}s of audio")

                    text = ""
                    if total_sec >= MIN_SPEECH_SEC:
                        text = _transcribe(recording, cleaned)
                    elif debug:
                        print(f"[DBG] Skipped — too short ({total_sec:.2f}s)")

                    if debug:
                        print(f"[DBG] transcript={text!r}")

                    reset_state()

                    if text:
                        yield text

            except Exception as e:
                print(f"[error] processing: {e}")
                reset_state()

#  PUBLIC API

def listen(
    as_module: bool = True,
    once: bool = False,
    debug: bool = False,
    cleaned: bool = True,
    calibrate_once: bool = True,
    use_groq: bool = True,
):
    
    if debug:
        print(f"Using model:  {MODEL_PATH}")
        print(f"Using binary: {WHISPER_PATH}")

    #  Groq setup 
    if use_groq:
        ok = _groq.setup(debug=debug)
        if ok:
            print(
                f"[groq] backend active "
                f"{len(_groq._keys)} key(s) loaded, "
                f"model={GROQ_MODEL}"
            )
        else:
            if debug:
                print("[groq] not available — using local backends")
    else:
        if debug:
            print("[groq] disabled by caller")


    #  Calibration 
    energy_threshold = None
    if calibrate_once:
        energy_threshold = _calibrate(debug=debug)

    #  Run 
    if as_module:
        if once:
            gen = _listen_generator(
                debug=debug, cleaned=cleaned, energy_threshold=energy_threshold
            )
            try:
                return next(gen, "")
            finally:
                gen.close()
        return _listen_generator(
            debug=debug, cleaned=cleaned, energy_threshold=energy_threshold
        )

    print("Listening… (Ctrl-C to stop)\n")
    try:
        gen = _listen_generator(
            debug=debug, cleaned=cleaned, energy_threshold=energy_threshold
        )
        if once:
            try:
                text = next(gen, "")
                if text:
                    print(text)
            finally:
                gen.close()
        else:
            for text in gen:
                print(text)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    listen(as_module=False, once=False, debug=True, cleaned=False, calibrate_once=True, use_groq=False)
