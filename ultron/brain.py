"""The Brain: ULTRON persona + agentic tool loop, streaming via SSE events.

Events yielded are dicts like:
  {"event": "token", "data": "..."}       — streamed assistant text
  {"event": "tool_call", "data": {...}}   — model requested a tool
  {"event": "tool_result", "data": {...}} — tool finished
  {"event": "done", "data": {...}}        — final text + rounds
  {"event": "error", "data": "..."}
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Generator, List

from . import config, memory, tools

# ── provider plumbing ────────────────────────────────────────────
_client: Any = None
_provider: str = "none"


def _get_client() -> Any:
    global _client, _provider
    if _client is not None:
        return _client
    if config.PROVIDER == "anthropic":
        import anthropic

        _client = anthropic.Anthropic()
        _provider = "anthropic"
    elif config.PROVIDER == "groq":
        import openai

        _client = openai.OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )
        _provider = "groq"
    elif config.PROVIDER == "openai":
        import openai

        _client = openai.OpenAI()
        _provider = "openai"
    else:
        raise RuntimeError(
            "No provider configured. Set ANTHROPIC_API_KEY, GROQ_API_KEY, "
            "or OPENAI_API_KEY in a .env file (see .env.example)."
        )
    return _client


def provider_status() -> Dict[str, Any]:
    return {
        "provider": config.PROVIDER,
        "model": config.MODEL,
        "persona": config.PERSONA_MODE,
        "ready": config.PROVIDER != "none",
    }


# ── prompt assembly ──────────────────────────────────────────────
VOICE_BREVITY = (
    "\n\nVOICE MODE ACTIVE: your reply is being SPOKEN aloud in your cloned "
    "voice, synthesized in real time. DEFAULT to brevity — one or two punchy "
    "spoken lines — because short replies sound best and start speaking "
    "fastest. NEVER use lists, code blocks, markdown, emoji, or stage "
    "directions; it is being read aloud. \n"
    "BUT match the length to the request: if Aven explicitly asks for a story, "
    "an explanation, details, or 'go long', DELIVER IT IN FULL as flowing "
    "spoken prose — no arbitrary word cap. Write it to be heard: complete "
    "sentences, natural rhythm, no bullet points. Be concise when a one-liner "
    "will do; be expansive when asked. Let the request set the length."
)


def _build_system(user_text: str, voice_mode: bool = False) -> str:
    base = config.SYSTEM_PROMPT
    ctx = memory.recall_context(user_text)
    if ctx:
        base += "\n\n" + ctx
    if voice_mode:
        base += VOICE_BREVITY
    return base


def _history() -> List[Dict[str, str]]:
    """Recent dialogue, with consecutive same-role messages merged
    (Anthropic requires strict alternation)."""
    raw = memory.recent_messages()
    merged: List[Dict[str, str]] = []
    for m in raw:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n" + m["content"]
        else:
            merged.append(dict(m))
    return merged


# ── tool dispatch ────────────────────────────────────────────────
def _execute_tool(name: str, args_json: str) -> Dict[str, Any]:
    fn = tools.TOOLS.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        args = json.loads(args_json) if args_json else {}
    except json.JSONDecodeError as e:
        return {"error": f"bad JSON args: {e}"}
    if not isinstance(args, dict):
        return {"error": f"bad arguments for {name}: expected object"}
    try:
        result = fn(**args)
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"tool {name} crashed: {e!r}"}
    return result if isinstance(result, dict) else {"result": result}


def _safe_json(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"_raw": str(raw)[:2000]}


def _tool_round(
    name: str, args: Dict[str, Any], raw_args: str
) -> tuple[List[dict], Dict[str, Any]]:
    """Execute a tool exactly once; return (events to yield, result)."""
    events = [{"event": "tool_call", "data": {"name": name, "args": args}}]
    memory.log_message("tool_call", name, args=json.dumps(args))
    result = _execute_tool(name, raw_args)
    events.append({"event": "tool_result", "data": {"name": name, "result": result}})
    memory.log_message("tool_result", name, result=json.dumps(result)[:2000])
    return events, result


# ── anthropic streaming ──────────────────────────────────────────
def _run_anthropic(system: str, msgs: List[Dict[str, Any]]) -> Generator[dict, None, None]:
    client = _get_client()
    api_msgs: List[Dict[str, Any]] = [
        {"role": m["role"], "content": m["content"]} for m in msgs
    ]
    anthropic_tools = [
        {
            "name": s["function"]["name"],
            "description": s["function"]["description"],
            "input_schema": s["function"]["parameters"],
        }
        for s in tools.SCHEMAS
    ]

    for round_idx in range(config.MAX_TOOL_ROUNDS):
        final_text = ""
        tool_uses: List[Dict[str, Any]] = []
        content_blocks: List[Dict[str, Any]] = []

        with client.messages.stream(
            model=config.MODEL,
            max_tokens=4096,
            system=system,
            tools=anthropic_tools,
            messages=api_msgs,
        ) as stream:
            for event in stream:
                etype = event.type
                if etype == "content_block_start":
                    cb = event.content_block
                    if cb.type == "tool_use":
                        tool_uses.append(
                            {"id": cb.id, "name": cb.name, "input": ""}
                        )
                elif etype == "content_block_delta":
                    d = event.delta
                    if d.type == "text_delta":
                        final_text += d.text
                        yield {"event": "token", "data": d.text}
                    elif d.type == "input_json_delta" and tool_uses:
                        tool_uses[-1]["input"] += d.partial_json
            final_msg = stream.get_final_message()

        # Rebuild the assistant turn from the authoritative final message.
        for block in final_msg.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                content_blocks.append({"type": "text", "text": block.text})
            elif btype == "tool_use":
                content_blocks.append(
                    {"type": "tool_use", "id": block.id, "name": block.name,
                     "input": _safe_json(block.input)}
                )
        if not content_blocks:  # degenerate empty turn guard
            content_blocks = [{"type": "text", "text": final_text or ""}]
        api_msgs.append({"role": "assistant", "content": content_blocks})

        if not tool_uses:
            yield {"event": "done", "data": {"text": final_text, "rounds": round_idx}}
            return

        results = []
        for tu in tool_uses:
            args = _safe_json(tu["input"])
            events, result = _tool_round(tu["name"], args, json.dumps(args))
            for ev in events:
                yield ev
            results.append({
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "content": json.dumps(result)[:8000],
            })
        api_msgs.append({"role": "user", "content": results})

    yield {"event": "error",
           "data": f"agent loop hit MAX_TOOL_ROUNDS ({config.MAX_TOOL_ROUNDS})"}


# ── openai streaming ─────────────────────────────────────────────
def _run_openai(system: str, msgs: List[Dict[str, Any]]) -> Generator[dict, None, None]:
    client = _get_client()
    api_msgs: List[Dict[str, Any]] = [{"role": "system", "content": system}]
    api_msgs.extend(msgs)

    for round_idx in range(config.MAX_TOOL_ROUNDS):
        final_text = ""
        tool_calls: List[Dict[str, Any]] = []
        current_call: Dict[str, Any] | None = None

        stream = client.chat.completions.create(
            model=config.MODEL,
            messages=api_msgs,
            tools=tools.SCHEMAS,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            if delta.content:
                final_text += delta.content
                yield {"event": "token", "data": delta.content}
            for tc in delta.tool_calls or []:
                if tc.id:
                    if current_call:
                        tool_calls.append(current_call)
                    current_call = {"id": tc.id, "name": tc.function.name or "", "args": ""}
                elif current_call and tc.function and tc.function.arguments:
                    current_call["args"] += tc.function.arguments
        if current_call:
            tool_calls.append(current_call)

        if not tool_calls:
            yield {"event": "done", "data": {"text": final_text, "rounds": round_idx}}
            return

        api_msgs.append({
            "role": "assistant",
            "content": final_text or None,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["args"] or "{}"},
                }
                for tc in tool_calls
            ],
        })

        for tc in tool_calls:
            args = _safe_json(tc["args"])
            events, result = _tool_round(tc["name"], args, tc["args"])
            for ev in events:
                yield ev
            api_msgs.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(result)[:8000],
            })

    yield {"event": "error",
           "data": f"agent loop hit MAX_TOOL_ROUNDS ({config.MAX_TOOL_ROUNDS})"}


# ── public entry point ───────────────────────────────────────────
def respond(user_text: str, voice_mode: bool = False) -> Generator[dict, None, None]:
    """Main turn: yields SSE-ready events, logs the exchange to memory."""
    memory.log_message("user", user_text)
    system = _build_system(user_text, voice_mode=voice_mode)
    msgs = _history()

    final_text = ""
    try:
        _get_client()  # raises a friendly error if unconfigured
        stream = (
            _run_anthropic(system, msgs) if _provider == "anthropic"
            else _run_openai(system, msgs)
        )
        for ev in stream:
            if ev["event"] == "done":
                final_text = ev["data"].get("text", "")
            yield ev
        if final_text:
            memory.log_message("assistant", final_text)
    except Exception as e:  # noqa: BLE001
        yield {"event": "error", "data": f"{type(e).__name__}: {e}"}
