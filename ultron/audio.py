"""Groq audio bridge via curl subprocess.

urllib is Cloudflare-blocked (error 1010) on Groq's audio endpoints;
Windows' system curl passes, so all audio traffic rides it.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile

from . import config  # noqa: F401  (loads .env before any key access)

BASE = "https://api.groq.com/openai/v1"
CURL = "curl"  # Windows ships curl.exe; Git Bash has it too


def _key() -> str:
    k = os.environ.get("GROQ_API_KEY", "").strip()
    if not k:
        raise RuntimeError("GROQ_API_KEY not configured")
    return k


def _run_curl(args: list, timeout: int) -> tuple[int, bytes]:
    p = subprocess.run(
        [CURL, "-s", "--max-time", str(timeout), *args],
        capture_output=True, timeout=timeout + 10,
    )
    return p.returncode, p.stdout


def transcribe(data: bytes, filename: str, content_type: str) -> dict:
    """Speech-to-text via whisper-large-v3-turbo."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "upload.bin")
        with open(path, "wb") as f:
            f.write(data)
        code, out = _run_curl(
            [
                BASE + "/audio/transcriptions",
                "-H", "Authorization: Bearer " + _key(),
                "-F", "model=" + os.environ.get("GROQ_STT_MODEL", "whisper-large-v3-turbo"),
                "-F", "response_format=json",
                "-F", f"file=@{path};type={content_type or 'audio/webm'}",
            ],
            timeout=90,
        )
    try:
        res = json.loads(out.decode(errors="replace"))
    except json.JSONDecodeError:
        return {"error": f"stt bad response ({code}): {out[:300]!r}"}
    if "error" in res:
        return {"error": "stt: " + json.dumps(res["error"])[:400]}
    return {"text": (res.get("text") or "").strip()}


def speak(text: str) -> dict:
    """Text-to-speech via orpheus; returns wav bytes."""
    payload = json.dumps({
        "model": os.environ.get("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english"),
        "input": text[:2000],
        "response_format": "wav",
        "voice": os.environ.get("GROQ_TTS_VOICE", "atlas"),
    })
    code, out = _run_curl(
        [
            BASE + "/audio/speech",
            "-H", "Authorization: Bearer " + _key(),
            "-H", "Content-Type: application/json",
            "-d", payload,
        ],
        timeout=180,
    )
    if out[:1] == b"{":
        try:
            res = json.loads(out.decode(errors="replace"))
            if "error" in res:
                return {"error": "tts: " + json.dumps(res["error"])[:400]}
        except json.JSONDecodeError:
            pass
    if code != 0 or not out:
        return {"error": f"tts failed (curl rc={code}, {len(out)} bytes)"}
    return {"audio": out, "content_type": "audio/wav"}
