# Run doc — ULTRON preview

Two Python FastAPI services (no npm):

- **Main server** — uvicorn via `run.py`, port **8000** (dashboard, chat, vitals, STT/TTS proxy)
- **Voice service** — `voice/server.py`, port **8001** (Chatterbox TTS + voice cloning, own venv)

## Reproduce artifacts (fresh checkout)

1. Main venv (system Python):
   ```powershell
   py -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```
2. Voice venv — needs **Python 3.12** (3.14 has no wheels for the ML stack).
   `uv` is installed at `C:\Users\razor\.local\bin\uv.exe` and manages its own
   Python interpreters:
   ```powershell
   $env:Path = "C:\Users\razor\.local\bin;$env:Path"
   uv venv voice/.venv --python 3.12
   uv pip install --python voice/.venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cpu
   uv pip install --python voice/.venv/Scripts/python.exe chatterbox-tts fastapi "uvicorn[standard]" soundfile
   ```
   First Chatterbox run downloads ~1GB of weights to
   `%USERPROFILE%\.cache\huggingface` (already cached on this machine — after
   that, model load at boot is seconds).
3. Environment file: none required. `.env` (copy from main checkout) supplies
   `GROQ_API_KEY` for the brain; without it chat fails gracefully. Optional:
   `GROQ_TTS_MODEL`, `GROQ_TTS_VOICE` for the orpheus fallback path.
4. Cloned voiceprint lives at `voice/samples/reference.wav` (uploaded via the
   HUD CLONE button or `POST /api/voice/clone`). Delete the file to revert to
   the stock voice.

## Run the servers

```powershell
powershell -NoProfile -Command "(Start-Process -FilePath '<ABS>\voice\.venv\Scripts\python.exe' -ArgumentList '\"<ABS>\voice\server.py\"' -RedirectStandardOutput '<log>' -RedirectStandardError '<log>.err' -WindowStyle Hidden -PassThru).Id"
powershell -NoProfile -Command "(Start-Process -FilePath '<ABS>\.venv\Scripts\python.exe' -ArgumentList '\"<ABS>\run.py\"' -RedirectStandardOutput '<log>' -RedirectStandardError '<log>.err' -WindowStyle Hidden -PassThru).Id"
```

- stdout and stderr must go to DIFFERENT files; uvicorn logs to the `.err` file.
- `ArgumentList` paths MUST be inner-quoted (`'"<ABS>\run.py"'`) — the space in
  "Projects 2.0" otherwise splits the argument and the process dies instantly
  with `can't find '__main__' module in 'E:\Projects'`.
- PS `Start-Process` often prints its pid slower than a 30-60s tool timeout;
  the process usually starts anyway — verify by port, not by command exit.
- Verify: `netstat -ano | findstr ":8000 :8001"` shows LISTENING;
  `curl http://127.0.0.1:8001/health` returns `ready/cloned` JSON;
  `curl http://127.0.0.1:8000/api/voice/status` mirrors it.

## Performance reality (Intel Iris Xe, CPU-only inference)

- Model load: ~6.5 min first ever boot (weights download); seconds once cached.
- Service now runs torch on ALL cores (default was half) + int8-quantizes the
  t3 transformer: measured 43s -> ~29s stock-voice per short sentence.
- Cloned-voice lines render ~55s; disk cache (voice/cache, 60MB LRU) makes
  repeated lines instant (~30ms).
- Chatterbox Turbo (350M) has inference code in the package but no loader for
  turbo weights in 0.1.7 — revisit when a release ships from_pretrained support
  (HF: ResembleAI/chatterbox-turbo).
- Voices are cloned by reference audio only — no training step involved.
