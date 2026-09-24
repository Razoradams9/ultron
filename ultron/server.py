"""FastAPI server: SSE chat, vitals, history, memory APIs + dashboard."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from . import audio, brain, memory, tools

app = FastAPI(title="ULTRON", version="0.1.0")

STATIC_DIR = Path(__file__).resolve().parent / "static"


class ChatIn(BaseModel):
    message: str
    voice_mode: bool = False  # true when VOICE is on: replies spoken aloud


class TTSIn(BaseModel):
    text: str
    voice: str = "local"  # "local" (Chatterbox on :8001) | "orpheus" (Groq)


@app.on_event("startup")
def _startup() -> None:
    memory.init()


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/status")
def status() -> dict:
    return {**brain.provider_status(), "memory": memory.stats()}


@app.post("/api/chat")
async def chat(body: ChatIn) -> StreamingResponse:
    async def event_stream() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue[dict | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def pump() -> None:
            try:
                for ev in brain.respond(body.message, voice_mode=body.voice_mode):
                    loop.call_soon_threadsafe(queue.put_nowait, ev)
            except Exception as e:  # noqa: BLE001
                loop.call_soon_threadsafe(
                    queue.put_nowait, {"event": "error", "data": repr(e)}
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        task = loop.run_in_executor(None, pump)
        while True:
            ev = await queue.get()
            if ev is None:
                break
            yield f"event: {ev['event']}\ndata: {json.dumps(ev['data'])}\n\n"
        await task

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/stt")
def stt(file: UploadFile = File(...)) -> dict:
    """Speech-to-text: browser audio blob -> whisper transcript."""
    data = file.file.read()
    if not data:
        raise HTTPException(400, "empty audio upload")
    if len(data) > 20_000_000:
        raise HTTPException(413, "audio too large (>20MB)")
    res = audio.transcribe(data, file.filename or "speech.webm", file.content_type or "")
    if "error" in res:
        raise HTTPException(502, res["error"])
    return res


import os

VOICE_URL = os.environ.get("VOICE_URL", "http://127.0.0.1:8001").rstrip("/")


def _voice_health(timeout: float = 2.0) -> dict | None:
    """Ping the local Chatterbox service; None when it's down."""
    try:
        import urllib.request

        with urllib.request.urlopen(VOICE_URL + "/health", timeout=timeout) as r:
            import json

            return json.loads(r.read().decode())
    except Exception:  # noqa: BLE001
        return None


def _voice_speak(text: str, timeout: float = 240.0) -> bytes | None:
    """Synthesize via the local voice service; None when unavailable."""
    import json
    import urllib.request

    payload = json.dumps({"text": text[:1000]}).encode()
    req = urllib.request.Request(
        VOICE_URL + "/speak",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception:  # noqa: BLE001
        return None


@app.post("/api/tts")
def tts(body: TTSIn) -> Response:
    """Text-to-speech: local Chatterbox first (clone if voiceprint exists),
    falling back to Groq orpheus."""
    if not body.text.strip():
        raise HTTPException(400, "empty text")
    if body.voice == "local":
        wav = _voice_speak(body.text)
        if wav is not None:
            return Response(content=wav, media_type="audio/wav")
    res = audio.speak(body.text)  # orpheus fallback / explicit choice
    if "error" in res:
        detail = res["error"]
        if body.voice == "local":
            detail = "local voice service offline; " + detail
        raise HTTPException(502, detail)
    return Response(content=res["audio"], media_type=res["content_type"])


@app.get("/api/voice/status")
def voice_status() -> dict:
    h = _voice_health()
    if h is None:
        return {"online": False}
    return {"online": True, **h}


@app.post("/api/voice/clone")
def voice_clone(file: UploadFile = File(...)) -> dict:
    """Proxy a reference clip to the local voice service (same-origin for the HUD)."""
    data = file.file.read()
    if not data:
        raise HTTPException(400, "empty upload")
    if len(data) > 30_000_000:
        raise HTTPException(413, "reference clip too large (>30MB)")
    import secrets
    import urllib.request

    b = secrets.token_hex(8)
    body = (
        f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{file.filename or 'ref.wav'}\"\r\n"
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + data + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(
        VOICE_URL + "/clone",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={b}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            import json

            return json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"voice service unavailable: {e}") from e


@app.delete("/api/voice/clone")
def voice_unclone() -> dict:
    import urllib.request

    req = urllib.request.Request(VOICE_URL + "/clone", method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            import json

            return json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"voice service unavailable: {e}") from e


@app.get("/api/vitals")
def vitals() -> dict:
    return tools.system_stats()


@app.get("/api/history")
def history(limit: int = 50) -> dict:
    return {"messages": memory.recent_messages(limit)}


@app.get("/api/facts")
def facts(limit: int = 100) -> dict:
    return {"facts": memory.search_facts("", limit)}


@app.delete("/api/memory")
def wipe() -> dict:
    memory.clear_all()
    return {"ok": True}
