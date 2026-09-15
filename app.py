<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ASL Live Transcription</title>
<style>
  :root { --paper:#f3f0e7; --panel:#fbfaf6; --rule:#ddd8c8; --ink:#1c1a16;
          --muted:#6d6961; --green:#4a7c4e; --warn:#9c4221; }
  * { box-sizing: border-box; }
  body {
    margin:0; padding:24px 20px 40px; background:var(--paper); color:var(--ink);
    font:15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }
  main { max-width: 940px; margin: 0 auto; }
  h1 { font-family:Georgia,serif; font-size:24px; margin:0 0 16px; }
  h2 { font-family:Georgia,serif; font-size:15px; margin:0 0 10px; }
  .stage { background:#16150f; border-radius:3px; aspect-ratio:4/3; display:grid;
           place-items:center; overflow:hidden; }
  .stage img { width:100%; height:100%; object-fit:contain; display:block; }
  .placeholder { color:#7d7a6e; font-size:14px; }
  .panel { background:var(--panel); border:1px solid var(--rule); border-radius:3px;
           padding:14px 16px 16px; margin-top:16px; }
  #transcript { font-family:Georgia,serif; font-size:22px; min-height:1.6em;
                white-space:pre-wrap; word-break:break-word; }
  #transcript:empty::before { content:"Nothing yet — sign at the camera.";
                              color:var(--muted); font-size:15px; }
  .actions { display:flex; gap:8px; margin-top:12px; flex-wrap:wrap; }
  button { font:inherit; padding:7px 14px; background:#eeebe1;
           border:1px solid #c9c3b2; border-radius:3px; cursor:pointer; color:var(--ink); }
  button:hover:not(:disabled) { background:#e5e1d4; }
  button:focus-visible { outline:2px solid var(--green); outline-offset:2px; }
  button:disabled { opacity:.45; cursor:not-allowed; }
  .primary { background:#dfe7d8; border-color:#a8ba9c; }
  .keys { font-size:13px; color:var(--muted); margin-top:12px; line-height:1.9; }
  .keys b { font-weight:500; color:var(--ink); }
  .letters { display:grid; grid-template-columns:repeat(13, 1fr); gap:5px; margin-top:8px; }
  .letters button { padding:6px 0; font-size:13px; text-align:center; }
  .letters button.wide { grid-column:span 2; }
  .letters button .n { display:block; font-size:10px; color:var(--muted);
                       font-variant-numeric:tabular-nums; }
  .letters button.ready { background:#dfe7d8; border-color:#a8ba9c; }
  .letters button.ready .n { color:var(--green); }
  @media (max-width:640px) { .letters { grid-template-columns:repeat(7,1fr); } }
  .banner { display:none; margin-bottom:14px; padding:9px 13px; border-radius:3px;
            font-size:14px; background:#f6e3da; border:1px solid #d9b3a2; color:var(--warn); }
</style>
</head>
<body>
<main>
  <h1>ASL Live Transcription</h1>
  <div class="banner" id="error"></div>
  <div class="banner" id="busy" style="background:#e3ecf6;border-color:#a9c2dd;color:#274b6d"></div>

  <div class="stage">
    <img id="feed" alt="Camera feed with hand landmarks" hidden>
    <p class="placeholder" id="placeholder">Starting the camera…</p>
  </div>

  <div class="panel">
    <h2>Transcription</h2>
    <div id="transcript"></div>
    <div class="actions">
      <button id="spaceBtn">Add space</button>
      <button id="backBtn">Backspace</button>
      <button id="clearBtn">Clear</button>
      <button id="copyBtn">Copy</button>
      <button id="trainBtn" class="primary">Train model</button>
      <button id="undoBtn" disabled>Undo last recording</button>
    </div>
    <p class="keys">
      <b>Any letter</b> records that sign &nbsp; <b>Space</b> and <b>Backspace</b>
      record those signs &nbsp; <b>Enter</b> trains &nbsp; <b>Esc</b> clears the text.
      J and Z need motion, so they are left out.
    </p>
    <h2 style="margin-top:18px">Teach a letter</h2>
    <p class="keys" style="margin-top:0">
      Click a letter (or press its key), then get your hand into the sign —
      you have three seconds before it starts recording.
    </p>
    <div class="letters" id="letters"></div>
  </div>
</main>

<script>
const $ = (id) => document.getElementById(id);
let lastResult = null;

async function post(path, body) {
  try {
    const r = await fetch(path, {method:"POST",
      headers:{"Content-Type":"application/json"}, body:JSON.stringify(body||{})});
    render(await r.json());
  } catch (e) { showError("Lost contact with the server. Is app.py running?"); }
}
function showError(m) {
  const b = $("error"); b.textContent = m || ""; b.style.display = m ? "block":"none";
}

function render(s) {
  if (!s) return;
  $("transcript").textContent = s.text || "";
  const busyNote = s.pending ? `Get ready: ${s.pending} — ${s.countdown.toFixed(0)}s`
                 : s.recording ? `Recording ${s.recording} — ${s.remaining} frames left`
                 : s.training ? "Training…" : "";
  $("busy").textContent = busyNote;
  $("busy").style.display = busyNote ? "block" : "none";
  $("undoBtn").disabled = !s.can_undo;
  $("trainBtn").disabled = s.training || s.total_samples === 0;
  $("trainBtn").textContent = s.training ? "Training…" : "Train model";
  showError(s.error);

  if (!$("letters").children.length) {
    $("letters").innerHTML = Object.keys(s.counts).map(k =>
      `<button data-letter="${k}" class="${k.length > 1 ? "wide" : ""}">${k}<span class="n">0</span></button>`
    ).join("");
    $("letters").querySelectorAll("button").forEach(b =>
      b.onclick = () => post("/api/record", {letter: b.dataset.letter}));
  }
  const busy = !!(s.recording || s.pending || s.training);
  $("letters").querySelectorAll("button").forEach(b => {
    const n = s.counts[b.dataset.letter] || 0;
    b.querySelector(".n").textContent = n;
    b.classList.toggle("ready", n >= s.recommended);
    b.disabled = busy || !s.running;
  });

  if (s.result && s.result !== lastResult) {
    lastResult = s.result;
    const acc = s.result.accuracy === null ? "trained"
      : Math.round(s.result.accuracy * 100) + "% accurate";
    showError("");
    $("trainBtn").title = `${s.result.classes.length} letters — ${acc}`;
  }

  const feed = $("feed");
  if (s.running && feed.hidden) {
    feed.src = "/video?t=" + Date.now();      // cache-bust so the stream restarts
    feed.hidden = false; $("placeholder").hidden = true;
  } else if (!s.running && !feed.hidden) {
    feed.hidden = true; feed.removeAttribute("src");
    $("placeholder").hidden = false;
    $("placeholder").textContent = "Camera stopped.";
  }
}

$("spaceBtn").onclick = () => post("/api/space");
$("backBtn").onclick  = () => post("/api/backspace");
$("clearBtn").onclick = () => post("/api/clear");
$("trainBtn").onclick = () => post("/api/train");
$("undoBtn").onclick  = () => post("/api/undo");
$("copyBtn").onclick  = async () => {
  await navigator.clipboard.writeText($("transcript").textContent);
  $("copyBtn").textContent = "Copied";
  setTimeout(() => $("copyBtn").textContent = "Copy", 1200);
};

document.addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key;
  // Controls are non-letter keys on purpose: every letter key has to stay
  // free to record itself, including T and C.
  if (k === "Enter")      { e.preventDefault(); return post("/api/train"); }
  if (k === "Escape")     { e.preventDefault(); return post("/api/clear"); }
  if (k === " ")          { e.preventDefault(); return post("/api/record", {letter:"space"}); }
  if (k === "Backspace")  { e.preventDefault(); return post("/api/record", {letter:"del"}); }
  if (!/^[a-zA-Z]$/.test(k)) return;
  const L = k.toUpperCase();
  if (L === "J" || L === "Z") return;   // motion letters, not modelled
  post("/api/record", {letter: L});
});

async function poll() {
  try { render(await (await fetch("/api/state")).json()); } catch (e) {}
}
setInterval(poll, 250);
// The camera starts on its own, so there is nothing to click to begin.
post("/api/start");
</script></body></html>