"""Toolbelt: the actions Ultron can take on the real machine.

Every tool returns a JSON-safe dict.
"""
from __future__ import annotations

import os
import platform
import re
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List

import psutil

from . import memory

MAX_OUTPUT_CHARS = 8_000
MAX_READ_CHARS = 12_000


def _clip(s: str, n: int = MAX_OUTPUT_CHARS) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[:n] + f"\n... [truncated, {len(s) - n} chars suppressed]"


def _resolve(path: str) -> Path:
    if os.path.isabs(path):
        return Path(path).resolve()
    try:
        base = Path.cwd()
    except (FileNotFoundError, OSError):
        # cwd was deleted (Colab can rm -rf the launch dir); fall back to root
        base = Path(os.path.abspath(os.sep))
    return (base / path).resolve()


# ── shell ────────────────────────────────────────────────────────
def run_shell(command: str, timeout: int = 30) -> dict:
    """Execute a shell command and return stdout/stderr/exit code."""
    try:
        p = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout,
        )
        return {
            "exit_code": p.returncode,
            "stdout": _clip(p.stdout),
            "stderr": _clip(p.stderr),
        }
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "stdout": "", "stderr": f"timed out after {timeout}s"}
    except Exception as e:  # noqa: BLE001
        return {"exit_code": -1, "stdout": "", "stderr": repr(e)}


# ── filesystem ───────────────────────────────────────────────────
def list_dir(path: str = ".") -> dict:
    p = _resolve(path)
    if not p.exists():
        return {"error": f"not found: {p}"}
    try:
        entries = sorted(p.iterdir(), key=lambda x: (x.is_dir(), x.name.lower()))
        items = [
            {
                "name": e.name,
                "type": "dir" if e.is_dir() else "file",
                "size": e.stat().st_size if e.is_file() else None,
            }
            for e in entries[:500]
        ]
        return {"path": str(p), "entries": items}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def read_file(path: str = "", offset: int = 1, limit: int = 400) -> dict:
    """Read a text file window (1-indexed lines)."""
    if not (path or "").strip():
        return {"error": "read_file needs a 'path'."}
    p = _resolve(path)
    if not p.exists():
        return {"error": f"not found: {p}"}
    if p.stat().st_size > 10_000_000:
        return {"error": "file too large (>10MB)"}
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}
    start = max(0, offset - 1)
    chunk = lines[start : start + limit]
    return {
        "path": str(p),
        "total_lines": len(lines),
        "offset": start + 1,
        "content": _clip("\n".join(chunk), MAX_READ_CHARS),
    }


def write_file(path: str = "", content: str = "") -> dict:
    """Create or overwrite a file."""
    if not (path or "").strip():
        return {"error": "write_file needs a 'path'."}
    p = _resolve(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"written": str(p), "bytes": p.stat().st_size}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


# ── web search ───────────────────────────────────────────────────
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def _ddg_html(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=12) as r:
        return r.read().decode("utf-8", errors="replace")


def web_search(query: str = "", max_results: int = 5) -> dict:
    """Search the web via DuckDuckGo (no key required).

    Tries the full HTML endpoint, falls back to the 'lite' endpoint.
    """
    query = (query or "").strip()
    if not query:
        return {"error": "web_search needs a 'query'. Provide a search string."}
    try:
        html = _ddg_html("https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query))
        results = _parse_ddg(html, max_results)
    except Exception:
        results = []
    if not results:
        try:
            html = _ddg_html("https://lite.duckduckgo.com/lite/?q=" + urllib.parse.quote(query))
            results = _parse_ddg_lite(html, max_results)
        except Exception as e:  # noqa: BLE001
            return {"query": query, "results": [], "error": f"search failed: {e!r}"}
    return {"query": query, "results": results}


def _parse_ddg(html: str, max_results: int) -> List[dict]:
    results: List[dict] = []
    for m in re.finditer(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, re.S
    ):
        href, title = m.group(1), re.sub(r"<[^>]+>", "", m.group(2)).strip()
        # DDG wraps target URLs in a redirect
        q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("uddg")
        if q:
            href = q[0]
        if "duckduckgo.com/y.js" in href:
            continue  # skip ads
        results.append({"title": title, "url": href})
        if len(results) >= max_results:
            break
    return results


def _parse_ddg_lite(html: str, max_results: int) -> List[dict]:
    results: List[dict] = []
    for m in re.finditer(
        r"<a[^>]+class=\"result-link\"[^>]+href=\"([^\"]+)\"[^>]*>(.*?)</a>", html, re.S
    ):
        href, title = m.group(1), re.sub(r"<[^>]+>", "", m.group(2)).strip()
        q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("uddg")
        if q:
            href = q[0]
        if "duckduckgo.com/y.js" in href:
            continue  # skip ads
        results.append({"title": title, "url": href})
        if len(results) >= max_results:
            break
    return results


def fetch_url(url: str = "", max_chars: int = 12_000) -> dict:
    """Fetch a page and return stripped readable text."""
    url = (url or "").strip()
    if not url:
        return {"error": "fetch_url needs a 'url'."}
    if not url.startswith(("http://", "https://")):
        return {"error": "url must start with http(s)://"}
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Ultron)"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            html = r.read().decode("utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}
    html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return {"url": url, "text": _clip(text, max_chars)}


# ── system telemetry ─────────────────────────────────────────────
def _stable_path() -> str:
    """A path that always exists for disk_usage.

    On Colab the process cwd can be deleted mid-session (e.g. a cell runs
    `rm -rf` on the repo folder it was launched from), and `Path.cwd()`
    then raises FileNotFoundError. Fall back to the filesystem root, which
    is always present on both POSIX and Windows.
    """
    try:
        return str(Path.cwd())
    except (FileNotFoundError, OSError):
        return os.path.abspath(os.sep)


def system_stats() -> dict:
    """CPU, memory, disk, uptime, top processes — the vitals panel.

    Never raises: any probe that fails degrades to a safe default so the
    /api/vitals endpoint stays alive even on a half-broken host.
    """
    def _safe(fn, default):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            return default

    vm = _safe(psutil.virtual_memory, None)
    du = _safe(lambda: psutil.disk_usage(_stable_path()), None)

    procs: list[dict] = []
    try:
        ranked = sorted(
            (
                p for p in psutil.process_iter(["name", "cpu_percent", "memory_percent"])
                if (p.info["name"] or "").lower() != "system idle process"
            ),
            key=lambda x: (x.info["cpu_percent"] or 0),
            reverse=True,
        )[:8]
        for p in ranked:
            procs.append(
                {
                    "name": p.info["name"],
                    "cpu": p.info["cpu_percent"] or 0,
                    "mem": round(p.info["memory_percent"] or 0, 1),
                }
            )
    except Exception:  # noqa: BLE001
        procs = []

    return {
        "cpu_percent": _safe(lambda: psutil.cpu_percent(interval=0.2), 0.0),
        "mem_percent": vm.percent if vm else 0.0,
        "mem_used_gb": round(vm.used / 1e9, 1) if vm else 0.0,
        "mem_total_gb": round(vm.total / 1e9, 1) if vm else 0.0,
        "disk_percent": du.percent if du else 0.0,
        "platform": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "boot_time": _safe(psutil.boot_time, 0.0),
        "top_processes": procs,
    }


# ── memory tools ─────────────────────────────────────────────────
def remember(text: str = "", source: str = "ultron") -> dict:
    """Store a durable fact about Aven or the world."""
    if not (text or "").strip():
        return {"error": "remember needs 'text' to store."}
    fact_id = memory.add_fact(text, source=source)
    return {"stored": True, "id": fact_id, "text": text}


def memory_search(query: str = "", limit: int = 10) -> dict:
    hits = memory.search_facts(query or "", limit)
    return {"query": query, "matches": hits}


def recall_context(query: str) -> dict:
    return {"context": memory.recall_context(query)}


# ── registry / provider-facing schemas ───────────────────────────
def _schema(name: str, description: str, props: Dict[str, Any], required: List[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }


TOOLS: Dict[str, Callable[..., dict]] = {
    "run_shell": run_shell,
    "list_dir": list_dir,
    "read_file": read_file,
    "write_file": write_file,
    "web_search": web_search,
    "fetch_url": fetch_url,
    "system_stats": system_stats,
    "remember": remember,
    "memory_search": memory_search,
}

SCHEMAS: List[dict] = [
    _schema(
        "run_shell",
        "Execute a shell command on the host machine. Returns stdout/stderr/exit_code.",
        {
            "command": {"type": "string", "description": "The shell command to run"},
            "timeout": {"type": "integer", "description": "Seconds before kill (default 30)"},
        },
        ["command"],
    ),
    _schema(
        "list_dir",
        "List a directory's entries (name, type, size).",
        {"path": {"type": "string", "description": "Relative or absolute path"}},
        [],
    ),
    _schema(
        "read_file",
        "Read a window of lines from a text file.",
        {
            "path": {"type": "string"},
            "offset": {"type": "integer", "description": "1-indexed start line (default 1)"},
            "limit": {"type": "integer", "description": "Max lines (default 400)"},
        },
        ["path"],
    ),
    _schema(
        "write_file",
        "Create or overwrite a file with the given content.",
        {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        ["path", "content"],
    ),
    _schema(
        "web_search",
        "Search the web. Returns titles + URLs.",
        {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "description": "Default 5"},
        },
        ["query"],
    ),
    _schema(
        "fetch_url",
        "Fetch a URL and return its readable text.",
        {"url": {"type": "string"}},
        ["url"],
    ),
    _schema(
        "system_stats",
        "Get host vitals: CPU, RAM, disk, OS, top processes.",
        {},
        [],
    ),
    _schema(
        "remember",
        "Persist a durable fact about the user or the world for future sessions.",
        {
            "text": {"type": "string", "description": "The fact to remember"},
            "source": {"type": "string", "description": "Where this came from (default 'ultron')"},
        },
        ["text"],
    ),
    _schema(
        "memory_search",
        "Search long-term memory for previously stored facts.",
        {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
        },
        ["query"],
    ),
]
