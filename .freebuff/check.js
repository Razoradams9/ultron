
/* ── neural entity — particle network, the movie centerpiece ────── */
const canvas = document.getElementById("entity");
const ctx = canvas.getContext("2d");
let W, H, CX, CY;
function resize() {
  W = canvas.width = canvas.offsetWidth * devicePixelRatio;
  H = canvas.height = canvas.offsetHeight * devicePixelRatio;
  CX = W / 2; CY = H * 0.52;
}
resize();
addEventListener("resize", resize);

const N = 150, LINK = 90 * devicePixelRatio;
const nodes = [];
for (let i = 0; i < N; i++) {
  const core = Math.random() < 0.45;                 // dense inner cluster
  const r = (core ? 0.55 * Math.pow(Math.random(), 1.6) : 0.6 + Math.random() * 0.4);
  nodes.push({
    a: Math.random() * Math.PI * 2,                  // angle
    r,                                               // radius factor
    z: (Math.random() - 0.5),                        // pseudo-depth wobble
    s: 0.4 + Math.random() * 0.6,                    // speed factor
    sz: (core ? 1.6 : 1.0) * devicePixelRatio,
  });
}
let busy = false, t = 0, pulse = 0;

function frame() {
  t += busy ? 0.014 : 0.004;
  ctx.clearRect(0, 0, W, H);
  const R = Math.min(W, H) * 0.34;
  const glow = busy ? 1 : 0.72;
  ctx.globalCompositeOperation = "lighter";

  // positions
  const pts = nodes.map(n => {
    const ang = n.a + t * n.s * (n.r < 0.55 ? 1.7 : 1);
    const wob = Math.sin(t * 2 + n.a * 3) * 8 * devicePixelRatio;
    return {
      x: CX + Math.cos(ang) * n.r * R,
      y: CY + Math.sin(ang) * n.r * R * 0.82 + n.z * wob,
      sz: n.sz,
    };
  });

  // links
  ctx.lineWidth = devicePixelRatio * 0.7;
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      const dx = pts[i].x - pts[j].x, dy = pts[i].y - pts[j].y;
      const d = Math.hypot(dx, dy);
      if (d < LINK) {
        const a = (1 - d / LINK) * 0.5 * glow;
        ctx.strokeStyle = `rgba(90, 190, 255, ${a})`;
        ctx.beginPath(); ctx.moveTo(pts[i].x, pts[i].y); ctx.lineTo(pts[j].x, pts[j].y); ctx.stroke();
      }
    }
  }
  // nodes
  for (const p of pts) {
    ctx.fillStyle = `rgba(160, 225, 255, ${0.85 * glow})`;
    ctx.shadowColor = "rgba(90, 190, 255, .9)";
    ctx.shadowBlur = 6 * devicePixelRatio;
    ctx.beginPath(); ctx.arc(p.x, p.y, p.sz, 0, 7); ctx.fill();
  }
  ctx.shadowBlur = 0;
  // core bloom
  const g = ctx.createRadialGradient(CX, CY, 0, CX, CY, R * 0.5);
  g.addColorStop(0, `rgba(120, 200, 255, ${0.16 * glow + pulse})`);
  g.addColorStop(1, "rgba(120, 200, 255, 0)");
  ctx.fillStyle = g;
  ctx.fillRect(CX - R, CY - R, R * 2, R * 2);
  if (pulse > 0) pulse -= 0.02;

  ctx.globalCompositeOperation = "source-over";
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

/* ── chat plumbing (unchanged protocol) ─────────────────────────── */
const log = document.getElementById("log");
const input = document.getElementById("input");
const send = document.getElementById("send");
let chatBusy = false;

function el(cls, who, body) {
  const m = document.createElement("div");
  m.className = "msg " + cls;
  const w = document.createElement("div");
  w.className = "who"; w.textContent = who;
  const b = document.createElement("div");
  b.className = "body"; if (body) b.textContent = body;
  m.append(w, b); log.appendChild(m);
  log.scrollTop = log.scrollHeight;
  return b;
}
function toolLine(name, label) {
  const t = document.createElement("div");
  t.className = "tool";
  t.innerHTML = "▸ <b>" + name + "</b> " + (label || "");
  log.appendChild(t); log.scrollTop = log.scrollHeight;
}

async function sendMsg() {
  const text = input.value.trim();
  if (!text || chatBusy) return;
  chatBusy = true; send.disabled = true; input.value = "";
  busy = true;
  el("user", "AVEN", text);
  const body = el("ultron", "ULTRON");
  const cursor = document.createElement("span");
  cursor.className = "cursor"; body.appendChild(cursor);
  let acc = "";

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({message: text, voice_mode: voiceOn}),
    });
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const raw = buf.slice(0, idx); buf = buf.slice(idx + 2);
        let ev = null, data = null;
        for (const line of raw.split("\n")) {
          if (line.startsWith("event: ")) ev = line.slice(7).trim();
          else if (line.startsWith("data: ")) data = line.slice(6);
        }
        if (!ev || data == null) continue;
        const d = JSON.parse(data);
        if (ev === "token") {
          acc += d; body.textContent = acc; body.appendChild(cursor);
          log.scrollTop = log.scrollHeight;
        } else if (ev === "tool_call") {
          pulse = 0.12;
          toolLine(d.name, JSON.stringify(d.args).slice(0, 110));
        } else if (ev === "tool_result") {
          const s = JSON.stringify(d.result);
          toolLine("✓ " + d.name, s.length > 110 ? s.slice(0, 110) + "…" : s);
        } else if (ev === "error") {
          body.className = "msg error"; body.textContent = "[FAULT] " + d;
        } else if (ev === "done" && acc) {
          refreshVitals();
          speak(acc);
        }
      }
    }
  } catch (e) {
    body.className = "msg error";
    body.textContent = acc || "[FAULT] connection severed: " + e;
  }
  cursor.remove();
  busy = false; chatBusy = false; send.disabled = false; input.focus();
}

/* spoken pipeline: split final text into sentences, render sentence N while
   sentence N-1 plays — first audio lands in ~1 render instead of N */
let spkQueue = [], spkBusy = false;
function speak(text) {
  if (!voiceOn || !text) return;
  const clean = text.replace(/\*\*/g, "").replace(/```[\s\S]*?```/g, " code omitted ")
                    .replace(/[#>_`|]/g, " ").replace(/\s+/g, " ").trim();
  const sentences = clean.match(/[^.!?]+[.!?]*/g) || [clean];
  spkQueue.push(...sentences.map(s => s.trim()).filter(Boolean));
  pumpSpeech();
}
function pumpSpeech() {
  if (spkBusy || !spkQueue.length) return;
  const text = spkQueue.shift();
  spkBusy = true;
  fetch("/api/tts", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({text: text.slice(0, 280), voice: "local"}),
  })
    .then(r => { if (!r.ok) throw new Error("tts " + r.status); return r.blob(); })
    .then(b => new Promise(res => {
      const a = new Audio(URL.createObjectURL(b));
      document.body.appendChild(a);
      a.play().catch(() => {});
      a.onended = () => { a.remove(); res(); };
    }))
    .catch(err => toolLine("✗ tts", String(err).slice(0, 90)))
    .finally(() => { spkBusy = false; if (voiceOn) pumpSpeech(); });
}
function stopSpeech() {
  spkQueue = []; spkBusy = false;
  document.querySelectorAll("audio").forEach(a => a.remove());
}
send.addEventListener("click", sendMsg);
input.addEventListener("keydown", e => { if (e.key === "Enter") sendMsg(); });

/* ── voice: STT (mic) + TTS (spoken replies) ─────────────────── */
const micBtn = document.getElementById("mic");
const spkBtn = document.getElementById("spk");
let mediaRec = null, recChunks = [], recording = false;
let voiceOn = false;
const cloneBtn = document.getElementById("cloneBtn");
const cloneFile = document.getElementById("cloneFile");

spkBtn.addEventListener("click", () => {
  voiceOn = !voiceOn;
  spkBtn.classList.toggle("on", voiceOn);
  spkBtn.textContent = voiceOn ? "VOICE: ON" : "VOICE: OFF";
  if (!voiceOn) stopSpeech();
});



/* ── local voice clone: upload sample -> Chatterbox voiceprint ─── */
async function voiceStatus() {
  try {
    const r = await fetch("/api/voice/status");
    const d = await r.json();
    if (d.online) {
      cloneBtn.textContent = d.cloned ? "CLONED" : "CLONE";
      cloneBtn.classList.toggle("on", !!d.cloned);
      cloneBtn.title = d.cloned
        ? "voiceprint active: " + d.sample.split(/[\\/]/).pop() + " — upload again to replace"
        : "upload a voice sample (5-15s clean speech) — cloned locally";
    } else {
      cloneBtn.title = "local voice service offline — replies will fall back to orpheus";
    }
  } catch {}
}
cloneBtn.addEventListener("click", () => cloneFile.click());
cloneFile.addEventListener("change", async () => {
  const f = cloneFile.files[0];
  if (!f) return;
  const prev = cloneBtn.textContent;
  cloneBtn.textContent = "…";
  try {
    const fd = new FormData();
    fd.append("file", f);
    const r = await fetch("/api/voice/clone", {method: "POST", body: fd});
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || r.status);
    toolLine("✓ clone", "voiceprint stored — ULTRON speaks in it now");
  } catch (e) {
    toolLine("✗ clone", String(e).slice(0, 90));
    cloneBtn.textContent = prev;
  }
  cloneFile.value = "";
  voiceStatus();
});
voiceStatus();

async function recordToggle() {
  if (recording) { mediaRec.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio: true});
    mediaRec = new MediaRecorder(stream);
    recChunks = [];
    mediaRec.ondataavailable = e => recChunks.push(e.data);
    mediaRec.onstop = async () => {
      recording = false; micBtn.classList.remove("rec"); micBtn.textContent = "MIC";
      stream.getTracks().forEach(t => t.stop());
      const blob = new Blob(recChunks, {type: mediaRec.mimeType || "audio/webm"});
      if (blob.size < 2000) { toolLine("✗ stt", "recording too short"); return; }
      micBtn.textContent = "…";
      try {
        const fd = new FormData();
        fd.append("file", blob, "speech.webm");
        const res = await fetch("/api/stt", {method: "POST", body: fd});
        const d = await res.json();
        if (!res.ok || !d.text) { toolLine("✗ stt", (d.detail || "no speech heard").slice(0, 90)); return; }
        input.value = d.text;
        sendMsg();
      } catch (e) {
        toolLine("✗ stt", String(e).slice(0, 90));
      } finally { micBtn.textContent = "MIC"; }
    };
    mediaRec.start();
    recording = true; micBtn.classList.add("rec"); micBtn.textContent = "REC";
    setTimeout(() => { if (recording) mediaRec.stop(); }, 30000);
  } catch (e) {
    toolLine("✗ mic", "permission denied or unavailable");
  }
}
micBtn.addEventListener("click", recordToggle);

/* ── vitals (unchanged) ─────────────────────────────────────────── */
async function refreshVitals() {
  try {
    const [v, s] = await Promise.all([
      fetch("/api/vitals").then(r => r.json()),
      fetch("/api/status").then(r => r.json()),
    ]);
    document.getElementById("cpuPct").textContent = v.cpu_percent.toFixed(0) + "%";
    document.getElementById("cpuBar").style.width = v.cpu_percent + "%";
    document.getElementById("memPct").textContent = v.mem_percent.toFixed(0) + "%";
    document.getElementById("memBar").style.width = v.mem_percent + "%";
    document.getElementById("dskPct").textContent = v.disk_percent.toFixed(0) + "%";
    document.getElementById("dskBar").style.width = v.disk_percent + "%";
    document.getElementById("osText").textContent = v.platform;
    const dot = document.getElementById("statusDot");
    dot.className = "sysdot " + (s.ready ? "on" : "");
    document.getElementById("provText").textContent = s.ready ? s.provider.toUpperCase() : "NO PROVIDER";
    document.getElementById("modelText").textContent = s.model || "—";
    document.getElementById("memStat").textContent = s.memory.messages + "/" + s.memory.facts;
  } catch (e) { /* cosmetic */ }
}
refreshVitals();
setInterval(refreshVitals, 4000);
