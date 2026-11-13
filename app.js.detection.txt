/* Gesture Smart Camera (web port) – camera, filters and hand gestures run fully in the browser. */
const VISION_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const MODELS = {
  hand: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
  face: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
  seg: "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
};
const W = 960;
let H = 540;
const $ = (id) => document.getElementById(id);
const video = $("cam"), out = $("out"), ctx = out.getContext("2d", { willReadFrequently: true });

/* ------------------------------------------------------------------ settings (same idea as config.py) */
const CFG = {
  pinchOn: 0.30, pinchOff: 0.45, pinchCooldown: 1.0,
  swipeDist: 0.28, swipeTime: 0.6, swipeCooldown: 1.0,
  hold: { "Open Palm": 0.8, Fist: 0.7, Peace: 0.7, "Thumbs Up": 0.7, Heart: 0.5 },
  cooldown: { next: 1, prev: 1, clear: 1.5, shot: 2, rec: 2.5, bg: 1.5, hearts: 2 },
};

/* ------------------------------------------------------------------ filters list */
const FILTERS = [
  { id: "none", name: "Normal", icon: "🎥", desc: "Plain camera" },
  { id: "shades", name: "Sunglasses", icon: "😎", face: true, desc: "Follows every face" },
  { id: "mask", name: "Face Mask", icon: "😷", face: true, desc: "Mask on every face" },
  { id: "hearts", name: "Floating Hearts", icon: "💖", desc: "Heart gesture = burst" },
  { id: "snow", name: "Snowfall", icon: "❄️", desc: "Falling snow" },
  { id: "fire", name: "Fire", icon: "🔥", desc: "Flames from below" },
  { id: "neon", name: "Neon", icon: "💜", desc: "Glowing edges" },
  { id: "rainbow", name: "Rainbow", icon: "🌈", desc: "Animated colours" },
  { id: "vintage", name: "Vintage", icon: "📷", desc: "Sepia + grain" },
  { id: "bw", name: "Black & White", icon: "⚫", desc: "Classic mono" },
  { id: "cartoon", name: "Cartoon", icon: "🎨", desc: "Flat colours + outlines" },
  { id: "bg", name: "Background", icon: "🪄", seg: true, desc: "Blur or replace" },
];
let fi = 0;
const BG_STYLES = ["Blur", "Purple", "Green", "Sunset", "Image"];
let bgImage = null, showFaceBoxes = true;
let bgStyle = 0, gestureLabel = "none";

/* ------------------------------------------------------------------ state */
let running = false, zoom = 1, targetZoom = 1;
let handLM = null, faceLM = null, segmenter = null;
let hands = [], faces = [];
let lastVideoTime = -1, maskCanvas = null, maskReady = false;
const status = { hand: "idle", face: "idle", seg: "idle" };

/* ------------------------------------------------------------------ small helpers */
function setChip(name, st, text) {
  status[name] = st;
  const el = $("chip" + name[0].toUpperCase() + name.slice(1));
  el.className = "chip " + (st === "ok" ? "ok" : st === "load" ? "load" : st === "bad" ? "bad" : "");
  if (text) el.lastChild.textContent = text;
}
let toastTimer;
function toast(msg) {
  const t = $("gtoast");
  t.textContent = msg; t.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), 1400);
}
// landmark (normalised, un-mirrored video coordinates) -> canvas pixels (mirrored + zoomed like the picture)
function P(lm) {
  const c0 = 0.5 - 0.5 / zoom;
  return [(((1 - lm.x) - c0) * zoom) * W, ((lm.y - c0) * zoom) * H];
}
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const d3 = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

/* draw any source (video / mask) with the same mirror + zoom transform */
function drawLike(c, src, w = W, h = H) {
  const sw = src.videoWidth || src.width, sh = src.videoHeight || src.height;
  const cw = sw / zoom, ch = sh / zoom, sx = (sw - cw) / 2, sy = (sh - ch) / 2;
  c.save();
  c.translate(w, 0); c.scale(-1, 1);
  c.drawImage(src, sx, sy, cw, ch, 0, 0, w, h);
  c.restore();
}

/* ------------------------------------------------------------------ MediaPipe loading (all optional) */
let visionMod = null, fileset = null;
async function loadVision() {
  if (fileset) return;
  visionMod = await import(VISION_URL + "/vision_bundle.mjs");
  fileset = await visionMod.FilesetResolver.forVisionTasks(VISION_URL + "/wasm");
}
async function make(Cls, model, extra) {
  for (const delegate of ["GPU", "CPU"]) {
    try {
      return await Cls.createFromOptions(fileset, { baseOptions: { modelAssetPath: model, delegate }, runningMode: "VIDEO", ...extra });
    } catch (e) { console.warn("delegate", delegate, "failed", e); }
  }
  throw new Error("model failed");
}
async function loadHand() {
  setChip("hand", "load", "Hand tracking…");
  try { await loadVision(); handLM = await make(visionMod.HandLandmarker, MODELS.hand, { numHands: 2 }); setChip("hand", "ok", "Hand tracking"); }
  catch (e) { console.warn(e); setChip("hand", "bad", "Hand tracking off"); }
}
async function loadFace() {
  if (faceLM || status.face === "load") return;
  setChip("face", "load", "Face tracking…");
  try { await loadVision(); faceLM = await make(visionMod.FaceLandmarker, MODELS.face, { numFaces: 3 }); setChip("face", "ok", "Face tracking"); }
  catch (e) { console.warn(e); setChip("face", "bad", "Face tracking off"); }
}
async function loadSeg() {
  if (segmenter || status.seg === "load") return;
  setChip("seg", "load", "Background AI…");
  try { await loadVision(); segmenter = await make(visionMod.ImageSegmenter, MODELS.seg, { outputCategoryMask: true, outputConfidenceMasks: false }); setChip("seg", "ok", "Background AI"); }
  catch (e) { console.warn(e); setChip("seg", "bad", "Background AI off"); }
}

/* ------------------------------------------------------------------ gesture engine (port of gestures.py) */
const FINGER_IDS = { index: [8, 6], middle: [12, 10], ring: [16, 14], pinky: [20, 18] };
function palmSize(h) { return d3(h[0], h[9]) || 0.1; }
function fingers(h) {
  const f = {};
  for (const [n, [tip, pip]] of Object.entries(FINGER_IDS)) f[n] = d3(h[tip], h[0]) > d3(h[pip], h[0]) * 1.1;
  f.thumb = d3(h[4], h[17]) > d3(h[3], h[17]) * 1.02 && d3(h[4], h[17]) > 1.05 * palmSize(h) * 0.6;
  return f;
}
function pose(h) {
  const f = fingers(h), palm = palmSize(h);
  const four = [f.index, f.middle, f.ring, f.pinky];
  if (f.thumb && four.every((x) => !x) && h[4].y < h[5].y - 0.2 * palm) return "Thumbs Up";
  if (four.every((x) => x) && f.thumb) return "Open Palm";
  if (f.index && f.middle && !f.ring && !f.pinky) return "Peace";
  if (four.every((x) => !x)) return "Fist";
  return null;
}
let pinchState = [false, false];
function pinch(h, i) {
  const r = d3(h[4], h[8]) / palmSize(h);
  const f = fingers(h);
  const fistLike = !f.middle && !f.ring && !f.pinky && !f.index; // fist also brings tips close
  if (pinchState[i]) pinchState[i] = r < CFG.pinchOff;
  else pinchState[i] = r < CFG.pinchOn && !fistLike;
  return pinchState[i];
}
function isHeart(a, b) {
  const palm = (palmSize(a) + palmSize(b)) / 2;
  const thumbs = d3(a[4], b[4]), idx = d3(a[8], b[8]), wr = d3(a[0], b[0]);
  const idxY = (a[8].y + b[8].y) / 2, thY = (a[4].y + b[4].y) / 2;
  return thumbs < 0.6 * palm && idx < 0.6 * palm && wr > 1.5 * palm && idxY < thY - 0.5 * palm;
}

const lastRun = {};
function act(name, fn, label) {
  const now = performance.now() / 1000;
  if (now - (lastRun[name] || -99) < CFG.cooldown[name]) return;
  lastRun[name] = now;
  fn(); if (label) toast(label);
}
let curPose = null, poseSince = 0, poseFired = false;
let wasPinch = false, lastPinch = -99, swipeHist = [], lastSwipe = -99;
let zoomBase = null;

function runGestures(now) {
  const t = now / 1000;
  if (hands.length === 0) { setGesture("none"); curPose = null; wasPinch = false; pinchState = [false, false]; swipeHist = []; zoomBase = null; targetZoom = Math.max(1, targetZoom); return; }

  if (hands.length === 2) {
    const [a, b] = hands;
    if (isHeart(a, b)) { setGesture("Heart"); act("hearts", () => { setFilter(FILTERS.findIndex((f) => f.id === "hearts")); burstHearts(W / 2, H / 2); }, "🫶 Hearts!"); return; }
    const pa = pinch(a, 0), pb = pinch(b, 1);
    if (pa && pb) {
      setGesture("Zoom");
      const d = Math.hypot(a[0].x - b[0].x, a[0].y - b[0].y);
      if (zoomBase === null) zoomBase = { d, z: targetZoom };
      targetZoom = Math.min(3, Math.max(1, zoomBase.z * (d / zoomBase.d)));
      return;
    }
    zoomBase = null;
  } else zoomBase = null;

  const h = hands[0];
  // pinch -> next filter (rising edge + cooldown)
  const p = pinch(h, 0);
  if (p && !wasPinch && t - lastPinch >= CFG.pinchCooldown) { lastPinch = t; next(); toast("🤏 Next filter"); }
  wasPinch = p;
  if (p) setGesture("Pinch");

  // swipe (fast horizontal move of the wrist, measured in the mirrored picture)
  const x = 1 - h[0].x;
  swipeHist.push([t, x]);
  swipeHist = swipeHist.filter(([tt]) => t - tt <= CFG.swipeTime);
  if (swipeHist.length > 3 && t - lastSwipe > CFG.swipeCooldown && !p) {
    const dx = x - swipeHist[0][1];
    if (Math.abs(dx) > CFG.swipeDist) { lastSwipe = t; swipeHist = []; if (dx > 0) { setGesture("Swipe Right"); next(); toast("👉 Next filter"); } else { setGesture("Swipe Left"); prev(); toast("👈 Previous filter"); } }
  }

  // held poses
  const ps = pose(h);
  if (!p) setGesture(ps || "Hand");
  if (ps !== curPose) { curPose = ps; poseSince = t; poseFired = false; }
  else if (ps && !poseFired && t - poseSince >= (CFG.hold[ps] || 0.7)) {
    poseFired = true;
    if (ps === "Open Palm") act("clear", clearFilter, "🖐 Filter cleared");
    if (ps === "Fist") act("shot", screenshot, "✊ Screenshot saved");
    if (ps === "Peace") act("rec", toggleRecord, "✌ Recording toggled");
    if (ps === "Thumbs Up") act("bg", cycleBackground, "👍 Background: " + BG_STYLES[bgStyle]);
  }
}

/* ------------------------------------------------------------------ actions */
function setFilter(i) {
  fi = (i + FILTERS.length) % FILTERS.length;
  const f = FILTERS[fi];
  const label = f.name + (f.id === "bg" ? " · " + BG_STYLES[bgStyle] : "");
  $("hudFilter").textContent = "Filter: " + label; $("curFilter").textContent = label;
  document.querySelectorAll(".gcard").forEach((b, k) => b.classList.toggle("on", k === fi));
  document.querySelectorAll("#bgSeg button").forEach((b, k) => b.classList.toggle("on", k === bgStyle));
  $("bgNote").style.opacity = f.id === "bg" ? 1 : 0.6;
  document.querySelectorAll(".fbtn").forEach((b, k) => b.classList.toggle("on", k === fi));
  if (f.face) loadFace();
  if (f.seg) loadSeg();
  if (f.id === "snow") initSnow();
}
const next = () => setFilter(fi + 1);
const prev = () => setFilter(fi - 1);
function clearFilter() { setFilter(0); }
function cycleBackground() {
  const bi = FILTERS.findIndex((f) => f.id === "bg");
  if (fi === bi) { do { bgStyle = (bgStyle + 1) % BG_STYLES.length; } while (BG_STYLES[bgStyle] === "Image" && !bgImage); buildBgSeg(); }
  setFilter(bi);
}
function download(url, name) { const a = document.createElement("a"); a.href = url; a.download = name; document.body.append(a); a.click(); a.remove(); }
function screenshot() {
  out.toBlob((b) => { if (b) { const u = URL.createObjectURL(b); download(u, `smart-camera-${Date.now()}.png`); setTimeout(() => URL.revokeObjectURL(u), 4000); } }, "image/png");
  $("screen").animate([{ filter: "brightness(2)" }, { filter: "none" }], 180);
}
let recorder = null, recChunks = [], recTimer = null, recStart = 0;
function toggleRecord() {
  if (recorder && recorder.state === "recording") { recorder.stop(); return; }
  if (typeof MediaRecorder === "undefined") { toast("Recording is not supported in this browser"); return; }
  const stream = out.captureStream(25);
  const mime = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"].find((m) => MediaRecorder.isTypeSupported(m));
  recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  recChunks = [];
  recorder.ondataavailable = (e) => e.data.size && recChunks.push(e.data);
  recorder.onstop = () => {
    clearInterval(recTimer); $("recDot").classList.add("hidden"); $("recBtn").classList.remove("on"); $("recBtn").textContent = "⏺ Record";
    const u = URL.createObjectURL(new Blob(recChunks, { type: "video/webm" }));
    download(u, `smart-camera-${Date.now()}.webm`); setTimeout(() => URL.revokeObjectURL(u), 8000);
  };
  recorder.start(500);
  recStart = Date.now(); $("recDot").classList.remove("hidden"); $("recBtn").classList.add("on"); $("recBtn").textContent = "⏹ Stop";
  recTimer = setInterval(() => { const s = Math.floor((Date.now() - recStart) / 1000); $("recTime").textContent = String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0"); }, 500);
}

/* ------------------------------------------------------------------ particles */
let snow = [], fire = [], hearts = [];
function initSnow() { snow = Array.from({ length: 140 }, () => ({ x: Math.random() * W, y: Math.random() * H, r: 1.5 + Math.random() * 3.5, vy: 40 + Math.random() * 90, vx: -15 + Math.random() * 30 })); }
function burstHearts(x, y) { for (let i = 0; i < 22; i++) hearts.push({ x, y, vx: -140 + Math.random() * 280, vy: -60 - Math.random() * 220, s: 18 + Math.random() * 30, life: 1.6 + Math.random() }); }
function drawSnow(dt) {
  ctx.fillStyle = "rgba(255,255,255,.9)";
  for (const f of snow) {
    f.y += f.vy * dt; f.x += f.vx * dt + Math.sin(f.y / 40) * 0.4;
    if (f.y > H + 5) { f.y = -5; f.x = Math.random() * W; }
    ctx.beginPath(); ctx.arc(f.x, f.y, f.r, 0, 6.3); ctx.fill();
  }
}
function drawFire(dt) {
  for (let i = 0; i < 8; i++) fire.push({ x: Math.random() * W, y: H + 10, vx: -20 + Math.random() * 40, vy: -120 - Math.random() * 200, r: 14 + Math.random() * 26, life: 1 });
  ctx.save(); ctx.globalCompositeOperation = "lighter";
  fire = fire.filter((p) => p.life > 0);
  for (const p of fire) {
    p.x += p.vx * dt; p.y += p.vy * dt; p.life -= dt * 0.9;
    const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r);
    g.addColorStop(0, `rgba(255,${Math.floor(200 * p.life)},60,${0.55 * p.life})`); g.addColorStop(1, "rgba(255,40,0,0)");
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, 6.3); ctx.fill();
  }
  ctx.restore();
}
function drawHearts(dt) {
  if (Math.random() < 0.18) hearts.push({ x: Math.random() * W, y: H + 20, vx: -20 + Math.random() * 40, vy: -90 - Math.random() * 90, s: 16 + Math.random() * 22, life: 5 });
  hearts = hearts.filter((h) => h.life > 0 && h.y > -40);
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  for (const h of hearts) {
    h.x += h.vx * dt; h.y += h.vy * dt; h.life -= dt; h.vx *= 0.995;
    ctx.globalAlpha = Math.min(1, h.life); ctx.font = `${h.s}px sans-serif`; ctx.fillText("❤️", h.x, h.y);
  }
  ctx.globalAlpha = 1;
}

/* ------------------------------------------------------------------ pixel filters */
function pixels(fn) { const img = ctx.getImageData(0, 0, W, H); fn(img.data); ctx.putImageData(img, 0, 0); }
function sobel(d, w, h) {
  const g = new Float32Array(w * h);
  for (let i = 0, j = 0; i < g.length; i++, j += 4) g[i] = 0.299 * d[j] + 0.587 * d[j + 1] + 0.114 * d[j + 2];
  const m = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x;
    const gx = -g[i - w - 1] - 2 * g[i - 1] - g[i + w - 1] + g[i - w + 1] + 2 * g[i + 1] + g[i + w + 1];
    const gy = -g[i - w - 1] - 2 * g[i - w] - g[i - w + 1] + g[i + w - 1] + 2 * g[i + w] + g[i + w + 1];
    m[i] = Math.hypot(gx, gy);
  }
  return m;
}
function fxBW() { pixels((d) => { for (let i = 0; i < d.length; i += 4) { let g = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]; g = Math.max(0, Math.min(255, (g - 128) * 1.15 + 128)); d[i] = d[i + 1] = d[i + 2] = g; } }); }
function fxVintage(t) {
  pixels((d) => { for (let i = 0; i < d.length; i += 4) { const r = d[i], g = d[i + 1], b = d[i + 2]; d[i] = Math.min(255, 0.393 * r + 0.769 * g + 0.189 * b); d[i + 1] = Math.min(255, 0.349 * r + 0.686 * g + 0.168 * b); d[i + 2] = Math.min(255, 0.272 * r + 0.534 * g + 0.131 * b); const n = (Math.random() - 0.5) * 22; d[i] += n; d[i + 1] += n; d[i + 2] += n; } });
  const g = ctx.createRadialGradient(W / 2, H / 2, H * 0.35, W / 2, H / 2, H * 0.95);
  g.addColorStop(0, "rgba(0,0,0,0)"); g.addColorStop(1, "rgba(30,15,0,.65)");
  ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
  if (Math.random() < 0.3) { ctx.fillStyle = "rgba(255,255,255,.25)"; ctx.fillRect(Math.random() * W, 0, 1.5, H); }
}
function fxCartoon() {
  pixels((d) => {
    const m = sobel(d, W, H);
    for (let i = 0, j = 0; i < m.length; i++, j += 4) {
      if (m[i] > 150) { d[j] = d[j + 1] = d[j + 2] = 15; continue; }
      for (let k = 0; k < 3; k++) d[j + k] = Math.round(d[j + k] / 51) * 51;
      // boost saturation a little
      const a = (d[j] + d[j + 1] + d[j + 2]) / 3;
      for (let k = 0; k < 3; k++) d[j + k] = Math.max(0, Math.min(255, a + (d[j + k] - a) * 1.35));
    }
  });
}
function fxNeon(t) {
  pixels((d) => {
    const m = sobel(d, W, H), hue = (t / 20) % 360;
    const c = (h) => { const a = h / 60, x = 1 - Math.abs((a % 2) - 1); const [r, g, b] = [[1, x, 0], [x, 1, 0], [0, 1, x], [0, x, 1], [x, 0, 1], [1, 0, x]][Math.floor(a) % 6]; return [r * 255, g * 255, b * 255]; };
    for (let i = 0, j = 0; i < m.length; i++, j += 4) {
      const e = Math.min(1, m[i] / 120);
      const col = c((hue + (i % W) / W * 120) % 360);
      d[j] = d[j] * 0.18 + col[0] * e * 1.6; d[j + 1] = d[j + 1] * 0.18 + col[1] * e * 1.6; d[j + 2] = d[j + 2] * 0.22 + col[2] * e * 1.6;
    }
  });
}
function fxRainbow(t) {
  const g = ctx.createLinearGradient(0, 0, W, H), off = (t / 25) % 360;
  for (let i = 0; i <= 6; i++) g.addColorStop(i / 6, `hsl(${(off + i * 60) % 360} 95% 55%)`);
  ctx.save(); ctx.globalCompositeOperation = "soft-light"; ctx.globalAlpha = 0.85; ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
  ctx.globalCompositeOperation = "overlay"; ctx.globalAlpha = 0.25; ctx.fillRect(0, 0, W, H); ctx.restore();
}

/* ------------------------------------------------------------------ face filters (landmark based) */
function faceInfo(lm) {
  const pt = (i) => P(lm[i]);
  let a = pt(33), b = pt(263); if (a[0] > b[0]) [a, b] = [b, a];
  const w = dist(pt(234), pt(454));
  return { pt, w, roll: Math.atan2(b[1] - a[1], b[0] - a[0]), eyeL: a, eyeR: b };
}
function ell(x, y, rx, ry, rot, fill) { ctx.save(); ctx.translate(x, y); ctx.rotate(rot); ctx.fillStyle = fill; ctx.beginPath(); ctx.ellipse(0, 0, rx, ry, 0, 0, 6.3); ctx.fill(); ctx.restore(); }
function drawShades(f) {
  const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  const L = mid(f.pt(33), f.pt(133)), R = mid(f.pt(362), f.pt(263));
  const [a, b] = L[0] < R[0] ? [L, R] : [R, L];
  ctx.strokeStyle = "#0b0b0b"; ctx.lineWidth = Math.max(3, f.w * 0.025); ctx.lineCap = "round";
  ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  const e1 = f.pt(234), e2 = f.pt(454); const [le, re] = e1[0] < e2[0] ? [e1, e2] : [e2, e1];
  ctx.beginPath(); ctx.moveTo(a[0] - f.w * 0.15, a[1]); ctx.lineTo(le[0], le[1] - f.w * 0.02); ctx.moveTo(b[0] + f.w * 0.15, b[1]); ctx.lineTo(re[0], re[1] - f.w * 0.02); ctx.stroke();
  for (const e of [a, b]) {
    ell(e[0], e[1], f.w * 0.17, f.w * 0.12, f.roll, "rgba(8,8,12,.92)");
    ell(e[0] - f.w * 0.06, e[1] - f.w * 0.04, f.w * 0.045, f.w * 0.018, f.roll - 0.5, "rgba(255,255,255,.35)");
  }
}

function drawMask(f) {
  // surgical face mask: nose bridge -> cheeks -> chin, with ear loops and pleats
  let t = f.pt(6), c = f.pt(152), l = f.pt(234), r = f.pt(454);
  if (l[0] > r[0]) [l, r] = [r, l];
  const lc = f.pt(93), rc = f.pt(323); const [A, B] = lc[0] < rc[0] ? [lc, rc] : [rc, lc];
  const up = f.w * 0.04; // slightly above the nose tip
  ctx.save();
  ctx.strokeStyle = "rgba(235,240,245,.9)"; ctx.lineWidth = Math.max(2, f.w * 0.012); ctx.lineCap = "round";
  ctx.beginPath(); ctx.moveTo(A[0], A[1]); ctx.quadraticCurveTo(l[0] - f.w * 0.06, l[1] - f.w * 0.02, l[0] - f.w * 0.02, l[1] - f.w * 0.1);
  ctx.moveTo(B[0], B[1]); ctx.quadraticCurveTo(r[0] + f.w * 0.06, r[1] - f.w * 0.02, r[0] + f.w * 0.02, r[1] - f.w * 0.1); ctx.stroke();
  const g = ctx.createLinearGradient(0, t[1], 0, c[1]); g.addColorStop(0, "#9fd3f2"); g.addColorStop(1, "#6fb4e0");
  ctx.fillStyle = g; ctx.strokeStyle = "rgba(255,255,255,.95)"; ctx.lineWidth = Math.max(2, f.w * 0.014);
  ctx.beginPath();
  ctx.moveTo(t[0] - f.w * 0.16, t[1] - up);
  ctx.quadraticCurveTo(t[0], t[1] - up * 2.2, t[0] + f.w * 0.16, t[1] - up);
  ctx.lineTo(B[0], B[1] - f.w * 0.02);
  ctx.quadraticCurveTo(B[0] - f.w * 0.04, c[1] - f.w * 0.04, c[0], c[1] + f.w * 0.04);
  ctx.quadraticCurveTo(A[0] + f.w * 0.04, c[1] - f.w * 0.04, A[0], A[1] - f.w * 0.02);
  ctx.closePath(); ctx.fill(); ctx.stroke();
  ctx.strokeStyle = "rgba(255,255,255,.55)"; ctx.lineWidth = Math.max(1.5, f.w * 0.008);
  for (const k of [0.3, 0.52, 0.74]) {
    const y = t[1] + (c[1] - t[1]) * k, hw = (B[0] - A[0]) * (0.46 - Math.abs(k - 0.4) * 0.3);
    ctx.beginPath(); ctx.moveTo(t[0] - hw, y); ctx.quadraticCurveTo(t[0], y + f.w * 0.025, t[0] + hw, y); ctx.stroke();
  }
  ctx.restore();
}

/* ------------------------------------------------------------------ background effect */
const bgCanvas = document.createElement("canvas"), tmp = document.createElement("canvas"), small = document.createElement("canvas");
function drawBackground(frame) {
  // frame: current picture already on `ctx`; mask = person. If no mask yet, keep the plain picture.
  if (!maskReady) return;
  bgCanvas.width = tmp.width = W; bgCanvas.height = tmp.height = H;
  const b = bgCanvas.getContext("2d");
  const style = BG_STYLES[bgStyle];
  if (style === "Image" && bgImage) {
    const r = Math.max(W / bgImage.width, H / bgImage.height), iw = bgImage.width * r, ih = bgImage.height * r;
    b.drawImage(bgImage, (W - iw) / 2, (H - ih) / 2, iw, ih);
  } else if (style === "Blur" || style === "Image") {
    small.width = W / 10; small.height = H / 10;
    const s = small.getContext("2d"); s.imageSmoothingEnabled = true; s.drawImage(frame, 0, 0, small.width, small.height);
    b.imageSmoothingEnabled = true; b.drawImage(small, 0, 0, W, H);
  } else {
    const g = b.createLinearGradient(0, 0, W, H);
    const cols = { Purple: ["#3b1c8c", "#c04cfd"], Green: ["#0b6b3a", "#9cf06b"], Sunset: ["#ff5e62", "#ffb347"] }[style];
    g.addColorStop(0, cols[0]); g.addColorStop(1, cols[1]); b.fillStyle = g; b.fillRect(0, 0, W, H);
  }
  const t = tmp.getContext("2d"); t.clearRect(0, 0, W, H);
  t.drawImage(frame, 0, 0);
  t.globalCompositeOperation = "destination-in"; t.save(); drawLike(t, maskCanvas); t.restore(); t.globalCompositeOperation = "source-over";
  ctx.drawImage(bgCanvas, 0, 0); ctx.drawImage(tmp, 0, 0);
}
function readMask(res) {
  const m = res.categoryMask; if (!m) return;
  const w = m.width, h = m.height, arr = m.getAsUint8Array();
  if (!maskCanvas || maskCanvas.width !== w || maskCanvas.height !== h) { maskCanvas = document.createElement("canvas"); maskCanvas.width = w; maskCanvas.height = h; }
  // find which value means "person": the person is normally in the middle of the picture, the corners are background
  let center = 0, corner = 0, nc = 0, nk = 0;
  for (let y = Math.floor(h * 0.4); y < h * 0.6; y += 4) for (let x = Math.floor(w * 0.4); x < w * 0.6; x += 4) { center += arr[y * w + x] > 0 ? 1 : 0; nc++; }
  for (const [x, y] of [[2, 2], [w - 3, 2], [2, h - 3], [w - 3, h - 3]]) { corner += arr[y * w + x] > 0 ? 1 : 0; nk++; }
  const personIsOn = center / nc >= corner / nk;
  const mc = maskCanvas.getContext("2d"), img = mc.createImageData(w, h), d = img.data;
  for (let i = 0; i < arr.length; i++) { const on = arr[i] > 0; d[i * 4 + 3] = (on === personIsOn) ? 255 : 0; d[i * 4] = d[i * 4 + 1] = d[i * 4 + 2] = 255; }
  mc.putImageData(img, 0, 0); maskReady = true;
}

/* ------------------------------------------------------------------ face boxes + HUD */
function drawFaceBoxes() {
  ctx.save(); ctx.lineWidth = 3; ctx.strokeStyle = "#ffc400"; ctx.fillStyle = "#ffc400"; ctx.font = "bold 16px sans-serif";
  faces.forEach((lm, i) => {
    let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
    for (const p of lm) { const [x, y] = P(p); if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    const pad = (x1 - x0) * 0.06; x0 -= pad; x1 += pad; y0 -= pad; y1 += pad;
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    ctx.fillText("Face " + (i + 1), x0, Math.max(16, y0 - 8));
  });
  ctx.restore();
}
const GEST_ICON = { none: "❔", Pinch: "🤏", "Open Palm": "🖐", Fist: "✊", Peace: "✌️", "Thumbs Up": "👍", Heart: "🫶", "Swipe Right": "👉", "Swipe Left": "👈", Zoom: "🔍", Hand: "✋" };
let gestHold = 0;
function setGesture(g) {
  const now = performance.now();
  if (g !== "none") gestHold = now;
  else if (now - gestHold < 900) return; // keep the last gesture visible for a moment
  if (g === gestureLabel) return;
  gestureLabel = g;
  $("gestName").textContent = g; $("gestIcon").textContent = GEST_ICON[g] || "✋"; $("hudGesture").textContent = "Gesture: " + g;
}
let hudZoomShown = "";
function updateHud() {
  const z = "Zoom: " + zoom.toFixed(1) + "x"; if (z !== hudZoomShown) { hudZoomShown = z; $("hudZoom").textContent = z; }
  const f = "Faces: " + faces.length; if ($("hudFaces").textContent !== f) $("hudFaces").textContent = f;
}

/* ------------------------------------------------------------------ main loop */
let lastT = performance.now();
function loop(now) {
  if (!running) return;
  requestAnimationFrame(loop);
  const dt = Math.min(0.05, (now - lastT) / 1000); lastT = now;
  if (video.readyState < 2) return;

  zoom += (targetZoom - zoom) * 0.2;
  const newH = Math.round(W * video.videoHeight / video.videoWidth);
  if (newH && newH !== H) { H = newH; out.height = H; }
  drawLike(ctx, video);

  // detection (only when a new video frame arrived)
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    try { if (handLM) hands = handLM.detectForVideo(video, now).landmarks || []; } catch (e) { hands = []; }
    const f = FILTERS[fi];
    try { if (faceLM && (f.face || showFaceBoxes)) faces = faceLM.detectForVideo(video, now).faceLandmarks || []; else if (!faceLM) faces = []; } catch (e) { faces = []; }
    try { if (f.seg && segmenter) segmenter.segmentForVideo(video, now, readMask); } catch (e) { /* ignore */ }
  }
  if (window.__mockFaces) faces = window.__mockFaces;
  runGestures(now);

  const f = FILTERS[fi];
  switch (f.id) {
    case "bw": fxBW(); break;
    case "vintage": fxVintage(now); break;
    case "cartoon": fxCartoon(); break;
    case "neon": fxNeon(now); break;
    case "rainbow": fxRainbow(now); break;
    case "snow": drawSnow(dt); break;
    case "fire": drawFire(dt); break;
    case "hearts": drawHearts(dt); break;
    case "shades": faces.forEach((l) => drawShades(faceInfo(l))); break;
    case "mask": faces.forEach((l) => drawMask(faceInfo(l))); break;
    case "bg": drawBackground(out); break;
  }
  if (showFaceBoxes) drawFaceBoxes();
  updateHud();
  if (f.face && !faceLM && status.face === "bad") hint("Face tracking could not load. Check your internet connection.");
  if (f.seg && !segmenter && status.seg === "bad") hint("Background AI could not load. Check your internet connection.");
}
let hintT = 0;
function hint(msg) { if (performance.now() - hintT > 4000) { hintT = performance.now(); toast(msg); } }

/* ------------------------------------------------------------------ stop / zoom / gallery */
function stopCamera() {
  running = false;
  if (recorder && recorder.state === "recording") recorder.stop();
  video.srcObject?.getTracks().forEach((t) => t.stop()); video.srcObject = null;
  hands = []; faces = []; maskReady = false; targetZoom = zoom = 1; lastVideoTime = -1;
  ctx.fillStyle = "#05060d"; ctx.fillRect(0, 0, W, H);
  document.querySelectorAll(".bar .btn").forEach((b) => (b.disabled = true));
  $("startBox").classList.remove("hidden"); setGesture("none"); $("hudZoom").textContent = "Zoom: 1.0x"; $("hudFaces").textContent = "Faces: 0";
}
const zoomBy = (d) => { targetZoom = Math.min(3, Math.max(1, +(targetZoom + d).toFixed(2))); };
function buildGallery() {
  const g = $("galGrid"); g.replaceChildren();
  FILTERS.forEach((f, i) => {
    const b = document.createElement("button"); b.className = "gcard" + (i === fi ? " on" : "");
    const ic = document.createElement("span"); ic.textContent = f.icon;
    const n = document.createElement("b"); n.textContent = f.name; const d = document.createElement("small"); d.textContent = f.desc;
    b.append(ic, n, d); b.onclick = () => { setFilter(i); $("gallery").classList.add("hidden"); }; g.append(b);
  });
}
function buildBgSeg() {
  const seg = $("bgSeg"); seg.replaceChildren();
  BG_STYLES.forEach((n, i) => {
    const b = document.createElement("button"); b.textContent = n === "Blur" ? "Blur" : n === "Image" ? "My image" : n; if (i === bgStyle) b.className = "on";
    b.onclick = () => {
      if (n === "Image" && !bgImage) { $("bgFile").click(); return; }
      bgStyle = i; setFilter(FILTERS.findIndex((f) => f.id === "bg")); toast("Background: " + n);
    };
    seg.append(b);
  });
}

/* ------------------------------------------------------------------ start */
async function start() {
  const msg = $("startMsg"); msg.textContent = "";
  if (!navigator.mediaDevices?.getUserMedia) { msg.textContent = "This browser has no camera support. Use Chrome or Edge over HTTPS."; return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false });
    video.srcObject = stream; await video.play();
  } catch (e) {
    msg.textContent = e.name === "NotAllowedError" ? "Camera permission was blocked. Allow the camera in the address bar and try again." : "Could not open the camera: " + (e.message || e.name);
    return;
  }
  $("startBox").classList.add("hidden");
  document.querySelectorAll(".bar .btn").forEach((b) => (b.disabled = false));
  running = true; lastT = performance.now(); requestAnimationFrame(loop);
  loadHand().then(loadFace);
}

/* ------------------------------------------------------------------ wire up UI */
const fbox = $("filters");
FILTERS.forEach((f, i) => { const b = document.createElement("button"); b.className = "fbtn" + (i === 0 ? " on" : ""); const s = document.createElement("span"); s.textContent = f.icon; b.append(s, document.createTextNode(f.name)); b.onclick = () => setFilter(i); fbox.append(b); });
$("startBtn").onclick = start;
$("prevBtn").onclick = prev; $("nextBtn").onclick = next; $("clearBtn").onclick = clearFilter;
$("shotBtn").onclick = screenshot; $("recBtn").onclick = toggleRecord;
addEventListener("keydown", (e) => {
  if (!running || e.ctrlKey || e.metaKey) return;
  const k = e.key.toLowerCase();
  if (k === "n" || k === "arrowright") next(); else if (k === "p" || k === "arrowleft") prev();
  else if (k === "c") clearFilter(); else if (k === "+" || k === "=") zoomBy(0.25); else if (k === "-") zoomBy(-0.25); else if (k === "s") screenshot(); else if (k === "r") toggleRecord();
});
$("stopBtn").onclick = stopCamera;
$("zoomIn").onclick = () => zoomBy(0.25); $("zoomOut").onclick = () => zoomBy(-0.25);
$("galBtn").onclick = () => { buildGallery(); $("gallery").classList.remove("hidden"); };
$("galClose").onclick = () => $("gallery").classList.add("hidden");
$("gallery").onclick = (e) => { if (e.target.id === "gallery") $("gallery").classList.add("hidden"); };
$("faceBox").onchange = (e) => { showFaceBoxes = e.target.checked; if (showFaceBoxes && running) loadFace(); };
$("bgFile").onchange = (e) => {
  const file = e.target.files[0]; if (!file) return;
  const img = new Image(); img.onload = () => { bgImage = img; bgStyle = BG_STYLES.indexOf("Image"); buildBgSeg(); setFilter(FILTERS.findIndex((f) => f.id === "bg")); toast("Background image set"); };
  img.src = URL.createObjectURL(file); e.target.value = "";
};
buildBgSeg(); buildGallery(); setFilter(0);
window.__app = { setFilter, FILTERS };
