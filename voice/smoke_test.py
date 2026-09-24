import sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from chatterbox.tts import ChatterboxTTS
import soundfile as sf
t0 = time.time()
print("loading model (weights download on first run)...", flush=True)
m = ChatterboxTTS.from_pretrained(device="cpu")
print(f"model loaded in {time.time()-t0:.0f}s, sr={m.sr}", flush=True)
t1 = time.time()
wav = m.generate("Systems online. All nine layers of my intellect at your disposal, Aven.")
print(f"synthesis took {time.time()-t1:.0f}s, frames={wav.shape}", flush=True)
sf.write("voice/smoke.wav", wav.squeeze(0).numpy(), m.sr)
print("WROTE voice/smoke.wav", flush=True)
