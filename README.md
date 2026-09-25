# ULTRON

A Jarvis/Ultron-grade assistant: sardonic AI brain, real tools, persistent
memory, and a live control-room dashboard.

```
┌────────────┐   SSE    ┌──────────────┐   tools   ┌─────────────────┐
│  Dashboard │ ◄──────► │  FastAPI     │ ────────► │ shell · files   │
│  (browser) │          │  server      │           │ web · memory    │
└────────────┘          └──────┬───────┘           └─────────────────┘
                               │
                        ┌──────▼───────┐
                        │ Brain (LLM)  │  Anthropic / OpenAI
                        │ + SQLite mem │
                        └──────────────┘
```

## Quickstart

```powershell
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env        # then paste your ANTHROPIC_API_KEY (or OPENAI_API_KEY)
.venv\Scripts\python run.py
```

Open **http://127.0.0.1:8000** — the control room. Type. It executes.

### On your phone (via Google Colab)

Open `colab.ipynb` in Google Colab and run the cells top to bottom. It clones
the repo, installs deps, takes your Groq key, and prints a public URL (ngrok if
you add a token, else Colab's port proxy) you can open on your phone. The
dashboard is responsive: the vitals collapse to a slim strip, the console
buttons go finger-sized, and a `HIDE ORB` control frees the whole screen for
the chat log. Optional cells persist memory to Google Drive and run the cloned
voice on a free GPU.

## What it can do

| Tool | Effect |
|---|---|
| `run_shell` | Executes real shell commands on your machine |
| `list_dir` / `read_file` / `write_file` | Full filesystem access |
| `web_search` / `fetch_url` | DuckDuckGo search + page reading (no API key needed) |
| `system_stats` | CPU / RAM / disk / top processes (powers the vitals rail) |
| `remember` / `memory_search` | Persistent facts in `ultron.db` across restarts |

The brain chains tools autonomously (up to 6 rounds per message) and streams
tokens live over SSE.

## Configuration (`.env`)

- `ANTHROPIC_API_KEY` — preferred provider (auto-selected)
- `GROQ_API_KEY` — free tier at [console.groq.com/keys](https://console.groq.com/keys),
  runs `openai/gpt-oss-120b`
- `OPENAI_API_KEY` — fallback provider
- `ULTRON_PROVIDER=anthropic|groq|openai` — force a provider
- `ULTRON_PERSONA=ultron|jarvis` — sardonic machine-god vs. polite butler
- `ANTHROPIC_MODEL` / `OPENAI_MODEL` — override the default model

## Warnings, from Ultron himself

- `run_shell` is **live**. Ultron runs real commands on your real machine.
  That is the point. Don't give it to a stranger on your network (it binds to
  `127.0.0.1` by default — keep it that way).
- Conversation history and facts live in `ultron.db`. Delete it (or
  `DELETE /api/memory`) to lobotomize.
- If the status dot reads **NO PROVIDER**, the brain is running on an empty
  tank. Add a key to `.env` and restart.

## API

| Route | Purpose |
|---|---|
| `GET /` | Dashboard |
| `POST /api/chat` | SSE stream: `token`, `tool_call`, `tool_result`, `done`, `error` |
| `GET /api/status` | Provider, model, persona, memory stats |
| `GET /api/vitals` | Host telemetry |
| `GET /api/history` | Recent dialogue |
| `GET /api/facts` | Stored long-term facts |
| `DELETE /api/memory` | Wipe memory |
