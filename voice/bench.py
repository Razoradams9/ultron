import json, os, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
print(f"torch {torch.__version__} default threads={torch.get_num_threads()} interop={torch.get_num_interop_threads()}", flush=True)
torch.set_num_threads(8)
torch.set_num_interop_threads(1)
print(f"set threads={torch.get_num_threads()}", flush=True)

from chatterbox.tts import ChatterboxTTS
t0 = time.time()
m = ChatterboxTTS.from_pretrained(device="cpu")
print(f"load {time.time()-t0:.0f}s sr={m.sr}", flush=True)
SENT = "Systems online. Awaiting your orders, Aven."

def run(tag, n=2):
    times = []
    for i in range(n):
        t = time.time()
        w = m.generate(SENT)
        times.append(time.time() - t)
        print(f"{tag} run{i+1}: {times[-1]:.1f}s frames={w.shape[-1]}", flush=True)
    return min(times)

base = run("STOCK")

# int8 dynamic quantization of the T3 transformer's linear layers
try:
    import torch.quantization as q
    t = time.time()
    m.tfmr = q.quantize_dynamic(m.tfmr, {torch.nn.Linear}, dtype=torch.qint8)
    print(f"quantized tfmr in {time.time()-t:.0f}s", flush=True)
    q8 = run("INT8")
    # sanity: output must not be silence
    w = m.generate(SENT)
    import numpy as np
    amp = float(w.abs().max())
    print(f"INT8 peak amplitude: {amp:.3f} ({'OK' if amp > 0.05 else 'SILENT - REJECT'})", flush=True)
except Exception as e:
    print(f"INT8 failed: {e!r}", flush=True)
    q8 = None

print(json.dumps({"stock_best": base, "int8_best": q8}), flush=True)
print("BENCH_DONE", flush=True)
