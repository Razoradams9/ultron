"""ULTRON local voice service — Chatterbox TTS on CPU, port 8001.

Endpoints:
    GET  /health      -> model/device/sample status
    POST /tts         {"text": "..."}                      -> wav (stock voice)
    POST /clone       multipart file=<ref.wav> [name=...]   -> registers cloned voice
    POST /speak       {"text": "..."}                       -> wav (cloned voice if set, else stock)
    DELETE /clone     -> drop the cloned voice

The model loads lazily on first synthesis (~20-40s first hit on CPU, then cached).
Reference samples live in voice/samples/.
"""
from __future__ import annotations

import io
import os
import struct
import sys
import wave
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

HERE = Path(__file__).resolve().parent
SAMPLES = HERE / "samples"
SAMPLES.mkdir(exist_ok=True)
REF_PATH = SAMPLES / "reference.wav"

import hashlib
import re
import threading

app = FastAPI(title="ultron-voice")

_state = {"model": None, "sr": 24000, "device": None}
_load_lock = threading.Lock()
_gen_lock = threading.Lock()  # CPU synthesis is strictly one-at-a-time

CACHE_DIR = HERE / "cache"
CACHE_LIMIT = 60 * 1024 * 1024  # 60MB of rendered lines

# Voiceprint length: Chatterbox conditions its output length on the prompt;
# a 21s prompt invites runaway generations. Trim to a tight 12s.
MAX_PROMPT_S = 12.0


def _log(msg: str) -> None:
    print(f"[voice] {msg}", flush=True)


def _pick_device() -> str:
    """VOICE_DEVICE env wins; otherwise auto-detect CUDA (Colab GPU) then CPU."""
    forced = (os.environ.get("VOICE_DEVICE") or "").strip().lower()
    if forced in ("cpu", "cuda"):
        return forced
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def get_model():
    """Lazy-load Chatterbox; cached for the process lifetime.

    On CPU: uses all cores and int8-quantizes the transformer (~35% faster).
    On CUDA (e.g. a Colab GPU): loads on the GPU and skips int8 quantization,
    which is a CPU-only path. Set VOICE_DEVICE=cpu|cuda to force it."""
    with _load_lock:
        if _state["model"] is None:
            import torch

            device = _pick_device()
            if device == "cpu":
                cores = os.cpu_count() or 4
                torch.set_num_threads(cores)      # torch defaults to half here
                try:
                    torch.set_num_interop_threads(1)
                except RuntimeError:
                    pass  # already initialized
                _log(f"loading ChatterboxTTS on cpu, {cores} threads...")
            else:
                _log(f"loading ChatterboxTTS on {device}...")
            from chatterbox.tts import ChatterboxTTS  # heavy import, deferred

            m = ChatterboxTTS.from_pretrained(device=device)
            if device == "cpu":
                try:
                    import torch.quantization as q

                    m.t3.tfmr = q.quantize_dynamic(
                        m.t3.tfmr, {torch.nn.Linear}, dtype=torch.qint8)
                    m.t3.speech_head = q.quantize_dynamic(
                        m.t3.speech_head, {torch.nn.Linear}, dtype=torch.qint8)
                    _log("int8 quantization applied to t3 transformer")
                except Exception as e:  # noqa: BLE001
                    _log(f"int8 quantization skipped: {e!r}")
            _state["model"] = m
            _state["sr"] = m.sr
            _state["device"] = device
            _log(f"model ready on {device}, sample rate {_state['sr']}")
    return _state["model"]


def _cache_key(text: str) -> Path:
    """Stable per-(text, voiceprint) cache path; changes when the clone changes."""
    ref_sig = ""
    if REF_PATH.exists():
        ref_sig = f"{REF_PATH.stat().st_mtime_ns}"
    h = hashlib.sha1(f"{text}|{ref_sig}".encode()).hexdigest()[:20]
    return CACHE_DIR / f"{h}.wav"


def _cache_prune() -> None:
    """Oldest-first eviction past CACHE_LIMIT."""
    files = sorted(CACHE_DIR.glob("*.wav"), key=lambda p: p.stat().st_atime_ns)
    total = sum(p.stat().st_size for p in files)
    for p in files:
        if total <= CACHE_LIMIT:
            break
        total -= p.stat().st_size
        p.unlink(missing_ok=True)


def _preload() -> None:
    """Warm the model at boot so the first real request doesn't pay the 6-minute load."""
    try:
        get_model()
        _log("preload complete — service warm")
    except Exception as e:  # noqa: BLE001
        _log(f"preload failed (will retry on first request): {e!r}")


@app.on_event("startup")
def _warm() -> None:
    threading.Thread(target=_preload, daemon=True).start()


def tensor_to_wav(wav_tensor, sr: int) -> bytes:
    """Chatterbox returns a (1, T) float tensor — pack it into a WAV in-memory."""
    import torch

    audio = wav_tensor.detach().cpu().flatten()
    audio = torch.clamp(audio, -1.0, 1.0)
    return _pcm_wav(audio.numpy(), sr)


@app.get("/health")
def health():
    return {
        "ready": _state["model"] is not None,
        "device": _state["device"] or _pick_device(),
        "cloned": REF_PATH.exists(),
        "sample": str(REF_PATH) if REF_PATH.exists() else None,
        "python": sys.version.split()[0],
    }


def _sentences(text: str, cap: int = 280) -> list[str]:
    """Split into sentence chunks under `cap` chars so each generate() call
    stays small and bounded (long single calls can run away on CPU)."""
    parts = re.split(r"(?<=[.!?;:])\s+", text[:1400])
    chunks: list[str] = []
    cur = ""
    for p in parts:
        if cur and len(cur) + 1 + len(p) > cap:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        chunks.append(cur)
    return chunks or [text[:cap]]


def _trim_prompt(path: Path) -> str:
    """Return a wav path whose audio is at most MAX_PROMPT_S (cached beside it)."""
    import soundfile as sf

    trimmed = path.with_suffix(".trim.wav")
    if not trimmed.exists():
        y, sr = sf.read(str(path), dtype="float32")
        n = int(MAX_PROMPT_S * sr)
        sf.write(str(trimmed), y[:n], sr, subtype="PCM_16")
        _log(f"voiceprint trimmed to {MAX_PROMPT_S}s -> {trimmed.name}")
    return str(trimmed)


def _synthesize(text: str) -> bytes:
    """Sentence-chunked synthesis with a hard per-call guard; concat to one WAV.
    Full renders are cached to disk, so repeats are instant."""
    import numpy as np
    import soundfile as sf

    CACHE_DIR.mkdir(exist_ok=True)
    ck = _cache_key(text)
    if ck.exists():
        _log(f"cache hit {ck.name}")
        p = ck.stat().st_atime_ns
        os.utime(ck, ns=(p, p))  # touch for LRU
        return ck.read_bytes()

    model = get_model()
    chunks = _sentences(text)
    prompt = _trim_prompt(REF_PATH) if REF_PATH.exists() else None
    sr = _state["sr"]
    pieces: list[np.ndarray] = []
    with _gen_lock:
        for i, c in enumerate(chunks):
            _log(f"chunk {i+1}/{len(chunks)} ({len(c)} chars)")
            if prompt:
                wav = model.generate(c, audio_prompt_path=prompt)
            else:
                wav = model.generate(c)
            pieces.append(wav.detach().cpu().flatten().clamp(-1, 1).numpy())
    audio = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
    out = _pcm_wav(audio, sr)
    if len(out) < 30_000_000:  # don't cache pathological runaway renders
        ck.write_bytes(out)
        _cache_prune()
    return out


def _pcm_wav(audio, sr: int) -> bytes:
    import numpy as np

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


@app.post("/tts")
def tts(body: dict):
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "text is required")
    _log(f"synthesizing {len(text)} chars (stock voice)")
    return Response(
        content=_synthesize(text),
        media_type="audio/wav",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/clone")
async def clone(file: UploadFile = File(...), name: str = Form("ultron")):
    """Store an uploaded reference clip; it becomes the voice of /speak."""
    data = await file.read()
    if len(data) < 20_000:
        raise HTTPException(400, "reference clip too small — need at least a few seconds of clean speech")
    # accept wav/mp3/ogg/flac/m4a; decode anything non-wav into wav via librosa
    suffix = Path(file.filename or "ref.wav").suffix.lower()
    if suffix in (".wav", ""):
        try:
            with wave.open(io.BytesIO(data), "rb") as w:
                if w.getnframes() < w.getframerate():  # <1s
                    raise HTTPException(400, "reference clip under 1 second")
        except wave.Error:
            suffix = ".bin"  # mislabeled container — fall through to decode
    if suffix not in (".wav", ""):
        import librosa

        y, sr = librosa.load(io.BytesIO(data), sr=None, mono=True)
        if len(y) < sr:
            raise HTTPException(400, "reference clip under 1 second")
        buf = io.BytesIO()
        import soundfile as sf

        sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
        data = buf.getvalue()

    REF_PATH.write_bytes(data)
    trimmed = REF_PATH.with_suffix(".trim.wav")
    if trimmed.exists():
        trimmed.unlink()  # old voiceprint's cache is now invalid
    # drop the cached model only if one is loaded; generate() picks up the new
    # prompt via audio_prompt_path argument, so no reload is actually needed.
    _log(f"reference clip saved ({len(data)} bytes) as '{name}' — /speak now uses it")
    return {"ok": True, "voice": name, "bytes": len(data)}


@app.post("/speak")
def speak(body: dict):
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "text is required")
    if REF_PATH.exists():
        _log(f"synthesizing {len(text)} chars (CLONED voice)")
    else:
        _log(f"synthesizing {len(text)} chars (stock voice — no reference yet)")
    return Response(content=_synthesize(text), media_type="audio/wav")


@app.delete("/clone")
def unclone():
    if REF_PATH.exists():
        REF_PATH.unlink()
        return {"ok": True, "removed": True}
    return JSONResponse({"ok": True, "removed": False})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
