"""Groq audio bridge.

Prefers urllib (works everywhere, handles UTF-8 cleanly). Falls back to the
system `curl` only when urllib fails AND curl exists — historically urllib was
Cloudflare-blocked (error 1010) on Groq's audio endpoints on some networks,
and Windows' curl slipped through. Passing user text (which contains em-dashes
and other non-ASCII) as a curl command-line arg crashed with UnicodeEncodeError
on ASCII-locale machines like Colab, so urllib is now the primary path.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request

from . import config  # noqa: F401  (loads .env before any key access)

BASE = "https://api.groq.com/openai/v1"
CURL = "curl"


def _key() -> str:
    k = os.environ.get("GROQ_API_KEY", "").strip()
    if not k:
        raise RuntimeError("GROQ_API_KEY not configured")
    return k


def _post_json(url: str, payload: dict, timeout: int) -> tuple[int, bytes]:
    """POST a JSON body via urllib. Returns (status, raw_bytes)."""
    body = json.dumps(payload).encode("utf-8")  # UTF-8: em-dashes are fine
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": "Bearer " + _key(),
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _curl_available() -> bool:
    return shutil.which(CURL) is not None


def transcribe(data: bytes, filename: str, content_type: str) -> dict:
    """Speech-to-text via whisper-large-v3-turbo (multipart upload)."""
    model = os.environ.get("GROQ_STT_MODEL", "whisper-large-v3-turbo")
    ctype = content_type or "audio/webm"
    url = BASE + "/audio/transcriptions"
    out = b""

    # PRIMARY: curl (dodges Cloudflare 1010 on Colab); file written to disk
    if _curl_available():
        try:
            with tempfile.NamedTemporaryFile("wb", suffix=".bin", delete=False) as f:
                f.write(data)
                upload_path = f.name
            p = subprocess.run(
                [CURL, "-s", "--max-time", "90", url,
                 "-H", "Authorization: Bearer " + _key(),
                 "-F", "model=" + model,
                 "-F", "response_format=json",
                 "-F", f"file=@{upload_path};type={ctype}"],
                capture_output=True, timeout=100,
            )
            os.unlink(upload_path)
            out = p.stdout
        except Exception:  # noqa: BLE001
            out = b""

    # FALLBACK: urllib multipart
    if not out:
        boundary = "----ultron" + os.urandom(8).hex()
        b = boundary.encode()
        payload = b"".join([
            b"--" + b + b"\r\n"
            b'Content-Disposition: form-data; name="model"\r\n\r\n'
            + model.encode() + b"\r\n",
            b"--" + b + b"\r\n"
            b'Content-Disposition: form-data; name="response_format"\r\n\r\n'
            b"json\r\n",
            b"--" + b + b"\r\n"
            b'Content-Disposition: form-data; name="file"; filename="'
            + (filename or "speech.webm").encode() + b'"\r\n'
            b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + data + b"\r\n",
            b"--" + b + b"--\r\n",
        ])
        req = urllib.request.Request(
            url, data=payload,
            headers={
                "Authorization": "Bearer " + _key(),
                "Content-Type": "multipart/form-data; boundary=" + boundary,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                out = r.read()
        except urllib.error.HTTPError as e:
            out = e.read()
        except Exception as e:  # noqa: BLE001
            return {"error": f"stt request failed: {e!r}"}
    try:
        res = json.loads(out.decode(errors="replace"))
    except json.JSONDecodeError:
        return {"error": f"stt bad response: {out[:300]!r}"}
    if "error" in res:
        return {"error": "stt: " + json.dumps(res["error"])[:400]}
    return {"text": (res.get("text") or "").strip()}


def speak(text: str) -> dict:
    """Text-to-speech via orpheus; returns wav bytes."""
    payload = {
        "model": os.environ.get("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english"),
        "input": text[:2000],
        "response_format": "wav",
        # Valid Orpheus English voices: troy, austin, hannah, autumn, ...
        # (the old default "atlas" is not a real voice and caused 502s).
        "voice": os.environ.get("GROQ_TTS_VOICE", "troy"),
    }
    url = BASE + "/audio/speech"
    status, out = -1, b""

    # PRIMARY: curl. urllib gets Cloudflare-blocked (error 1010) on Groq's
    # audio endpoint from data-center IPs (e.g. Colab); curl slips through.
    # Body goes via a temp file so non-ASCII text (em-dashes) never touches
    # the command line — that previously crashed with UnicodeEncodeError.
    if _curl_available():
        try:
            with tempfile.NamedTemporaryFile("wb", suffix=".json", delete=False) as f:
                f.write(json.dumps(payload).encode("utf-8"))
                body_path = f.name
            p = subprocess.run(
                [CURL, "-s", "--max-time", "180", url,
                 "-H", "Authorization: Bearer " + _key(),
                 "-H", "Content-Type: application/json",
                 "--data-binary", "@" + body_path],
                capture_output=True, timeout=190,
            )
            os.unlink(body_path)
            if p.returncode == 0 and p.stdout and p.stdout[:1] != b"{":
                return {"audio": p.stdout, "content_type": "audio/wav"}
            if p.stdout:
                out, status = p.stdout, 200 if p.returncode == 0 else 502
        except Exception:  # noqa: BLE001
            pass

    # FALLBACK: urllib (works where curl is absent, e.g. some Windows setups)
    if out[:1] != b"{" and (status != 200 or not out):
        try:
            status, out = _post_json(url, payload, timeout=180)
        except Exception as e:  # noqa: BLE001
            status, out = -1, str(e).encode()

    if out[:1] == b"{":
        try:
            res = json.loads(out.decode(errors="replace"))
            if "error" in res:
                return {"error": "tts: " + json.dumps(res["error"])[:400]}
        except json.JSONDecodeError:
            pass
    if status != 200 or not out:
        return {"error": f"tts failed (status={status}, {len(out)} bytes)"}
    return {"audio": out, "content_type": "audio/wav"}
