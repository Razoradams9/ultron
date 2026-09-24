"""Pick the cleanest ~20s of Ultron speech from the compilation mp3.

1. Decode to mono 24k WAV chunks (soundfile).
2. Transcribe each chunk on Groq whisper (via curl — urllib is CF-blocked).
3. Score segments by avg_logprob / no_speech_prob, select ~18-22s total.
4. Cut the selected spans -> voice/samples/reference.wav
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = Path("Ultron_ Best Lines & Moments.mp3.mpeg")
OUT = Path("voice/samples/reference.wav")
CHUNK_S = 100
TARGET_S = 20.0
SR = 24000


def key() -> str:
    for line in Path(".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("GROQ_API_KEY="):
            k = line.split("=", 1)[1].strip()
            if k:
                return k
    raise SystemExit("GROQ_API_KEY missing in .env")


def groq_stt(k: str, wav_path: Path, offset: float) -> list[dict]:
    r = subprocess.run(
        [
            "curl", "-s", "-m", "120",
            "https://api.groq.com/openai/v1/audio/transcriptions",
            "-H", f"Authorization: Bearer {k}",
            "-F", f"file=@{wav_path};type=audio/wav",
            "-F", "model=whisper-large-v3-turbo",
            "-F", "response_format=verbose_json",
            "-F", "timestamp_granularities[]=segment",
        ],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    try:
        d = json.loads(r.stdout)
    except Exception:
        print("chunk failed:", r.stdout[:200], r.stderr[:200])
        return []
    segs = []
    for s in d.get("segments", []):
        segs.append({
            "start": s["start"] + offset,
            "end": s["end"] + offset,
            "text": (s.get("text") or "").strip(),
            "lp": s.get("avg_logprob", -9.0),
            "ns": s.get("no_speech_prob", 1.0),
        })
    return segs


def main() -> None:
    audio, sr = sf.read(str(SRC), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # mono
    if sr != SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    total = len(audio) / SR
    print(f"decoded {total:.0f}s @ {SR}Hz mono")

    # chunk + transcribe
    k = key()
    all_segs: list[dict] = []
    tmp = Path(".freebuff/clone_chunks")
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = 0.0
    i = 0
    while t0 < total:
        n = min(CHUNK_S, total - t0)
        a0, a1 = int(t0 * SR), int((t0 + n) * SR)
        cpath = tmp / f"c{i}.wav"
        sf.write(str(cpath), audio[a0:a1], SR, subtype="PCM_16")
        segs = groq_stt(k, cpath, t0)
        print(f"chunk {i} ({t0:.0f}-{t0+n:.0f}s): {len(segs)} segments")
        all_segs.extend(segs)
        t0 += n
        i += 1

    # score: confident, actual speech
    good = [s for s in all_segs if s["text"] and s["ns"] < 0.35 and s["lp"] > -0.6]
    good.sort(key=lambda s: (-s["lp"], s["ns"]))
    print(f"{len(all_segs)} segments total, {len(good)} high-confidence")

    picked: list[dict] = []
    dur = 0.0
    for s in good:
        if dur >= TARGET_S:
            break
        if any(s["start"] < p["end"] and p["start"] < s["end"] for p in picked):
            continue
        picked.append(s)
        dur += s["end"] - s["start"]
    picked.sort(key=lambda s: s["start"])
    print(f"picked {len(picked)} spans, {dur:.1f}s total:")
    for s in picked:
        print(f"  [{s['start']:6.1f}-{s['end']:6.1f}] lp={s['lp']:+.2f}  {s['text'][:70]}")

    # cut from the ORIGINAL decoded audio
    pieces = [audio[int(s["start"] * SR):int(s["end"] * SR)] for s in picked]
    if not pieces:
        raise SystemExit("nothing selectable — fallback needed")
    clip = pieces[0]
    gap = np.zeros(int(0.25 * SR), dtype="float32")
    for p in pieces[1:]:
        clip = np.concatenate([clip, gap, p])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(OUT), clip, SR, subtype="PCM_16")
    print(f"WROTE {OUT} — {len(clip)/SR:.1f}s voiceprint")


if __name__ == "__main__":
    main()
