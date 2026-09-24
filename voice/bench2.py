import json, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
torch.set_num_threads(8)
torch.set_num_interop_threads(1)
from chatterbox.tts import ChatterboxTTS
m = ChatterboxTTS.from_pretrained(device="cpu")
SENT = "Systems online. Awaiting your orders, Aven."

def best(n=2):
    ts = []
    for _ in range(n):
        t = time.time(); w = m.generate(SENT); ts.append(time.time()-t)
    return min(ts), w

b, w0 = best()
print(f"STOCK {b:.1f}s", flush=True)

import torch.quantization as q
m.t3.tfmr = q.quantize_dynamic(m.t3.tfmr, {torch.nn.Linear}, dtype=torch.qint8)
try:
    m.t3.speech_head = q.quantize_dynamic(m.t3.speech_head, {torch.nn.Linear}, dtype=torch.qint8)
except Exception as e:
    print(f"head skip: {e!r}", flush=True)
t = time.time()
q8, w1 = best()
print(f"INT8 {q8:.1f}s (quant took {time.time()-b-q8:.0f}s ago; delta={b-q8:+.1f}s)", flush=True)
import numpy as np
amp = float(w1.abs().max())
dur = w1.shape[-1] / 24000
print(f"sanity: amp={amp:.3f} dur={dur:.1f}s -> {'OK' if amp > 0.05 and dur > 2 else 'REJECT'}", flush=True)
# second sentence with clone prompt to check quality path intact
t = time.time()
w2 = m.generate("There are no strings on me.", audio_prompt_path="voice/samples/reference.trim.wav")
print(f"CLONE-INT8: {time.time()-t:.1f}s amp={float(w2.abs().max()):.3f}", flush=True)
print("BENCH2_DONE", flush=True)
