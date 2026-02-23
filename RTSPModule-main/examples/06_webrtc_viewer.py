#!/usr/bin/env python3
"""
WebRTC Viewer for RTSPModule
============================
Serves a dashboard that streams all cameras via native browser WebRTC.

Architecture:
  - Each camera maps to one `webrtcrs_sink` GStreamer element (running inside the C++ pipeline)
  - ALL sinks share ONE signaling port (single Rust HTTP server via hub singleton)
  - Streams are identified by stream_id (= camera index) in the URL query param
  - This Python script serves ONLY the dashboard HTML — real video frames never touch Python
  - The browser does standard WebRTC SDP offer/answer directly with the Rust signaling server

Usage:
  export GST_PLUGIN_PATH=/home/akhil/gst-webrtcrs/target/release
  python3 examples/06_webrtc_viewer.py [--config configs/config.yaml] [--port 8090] [--signaling-port 9000]

Requirements:
  pip install aiohttp
"""

import os
import sys
import asyncio
import argparse
import json

from aiohttp import web

# Allow importing from the local source tree during development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from rtspmodule import RTSPModule


# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_CONFIG          = "configs/config.yaml"
DEFAULT_WEB_PORT        = 8090   # Dashboard port
DEFAULT_SIGNALING_PORT  = 9000   # Single shared WebRTC signaling port (all cameras)

# ─── Main Server ──────────────────────────────────────────────────────────────

class WebRtcServer:
    def __init__(self, config_path: str, web_port: int, signaling_port: int):
        self.config_path     = config_path
        self.web_port        = web_port
        self.signaling_port  = signaling_port
        self.provider        = RTSPModule()
        self.num_streams     = 0
        # Populated lazily from the hub's /streams endpoint.
        # Maps cam_index → actual stream_id string (e.g. "Camera 3").
        self._stream_ids: list[str] = []

    async def start(self) -> bool:
        if not os.path.exists(self.config_path):
            print(f"[ERROR] Config not found: {self.config_path}")
            return False

        self.provider.start(self.config_path)

        # Give GStreamer a moment to negotiate RTSP and build the pipeline
        await asyncio.sleep(2)

        self.num_streams = self.provider.stream_count()
        print(f"[INFO] {self.num_streams} stream(s) started")

        # Start WebRTC streaming for all cameras.  All sinks share one signaling port.
        # Each call wakes up the tee branch immediately if the pipeline is ready,
        # or queues it for when onPadAdded fires (for streams still negotiating).
        for i in range(self.num_streams):
            ok = self.provider.start_streaming(i)
            print(f"[INFO] Camera {i} → WebRTC stream_id='{i}' on shared port {self.signaling_port}  (queued={not ok})")

        return True

    def stop(self):
        self.provider.stop_streaming_all()
        self.provider.stop()

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _fetch_stream_ids(self) -> list[str]:
        """Query the hub's /streams endpoint and build a cam_index → stream_id mapping.

        The hub registers streams in the order they are started (start_streaming(0),
        start_streaming(1), …), so the order of the returned list matches the camera
        index used by the Python RTSPModule API.

        Returns an empty list if the hub is not reachable (caller must handle fallback)."""
        import aiohttp
        url = f"http://localhost:{self.signaling_port}/streams"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as r:
                    if r.status == 200:
                        data = await r.json()
                        ids = [entry["stream_id"] for entry in data]
                        print(f"[INFO] Hub stream_ids: {ids}")
                        return ids
        except Exception as e:
            print(f"[WARN] Could not fetch stream IDs from hub: {e}")
        return []  # Don't cache a fallback — let JS fetch dynamically

    async def _get_stream_ids(self) -> list[str]:
        """Return cached stream_ids, refreshing if empty."""
        if not self._stream_ids:
            self._stream_ids = await self._fetch_stream_ids()
        return self._stream_ids

    # ── REST: streaming control ────────────────────────────────────────────────

    async def handle_start_stream(self, request):
        cam_id  = int(request.match_info["cam_id"])
        ok      = self.provider.start_streaming(cam_id)
        # Invalidate cache so next /api/status or page load re-fetches
        self._stream_ids = []
        ids     = await self._get_stream_ids()
        real_id = ids[cam_id] if cam_id < len(ids) else str(cam_id)
        return web.json_response({"camera_id": cam_id, "started": ok,
                                   "signaling_port": self.signaling_port,
                                   "stream_id": real_id})

    async def handle_stop_stream(self, request):
        cam_id = int(request.match_info["cam_id"])
        self.provider.stop_streaming(cam_id)
        return web.json_response({"camera_id": cam_id, "stopped": True})

    async def handle_start_all(self, request):
        self.provider.start_streaming_all()
        self._stream_ids = []  # Invalidate cache
        return web.json_response({"started_all": True})

    async def handle_stop_all(self, request):
        self.provider.stop_streaming_all()
        self._stream_ids = []
        return web.json_response({"stopped_all": True})

    # ── REST: status / stats ──────────────────────────────────────────────────

    async def handle_api_status(self, request):
        """Return per-stream state: streaming active, FPS, resolution, signaling_port."""
        # Always fetch fresh stream IDs (hub may have just started)
        self._stream_ids = []
        ids     = await self._get_stream_ids()
        streams = []
        for i in range(self.num_streams):
            s       = self.provider.get_stats(i)
            real_id = ids[i] if i < len(ids) else str(i)
            streams.append({
                "id":             i,
                "stream_id":      real_id,
                "signaling_port": self.signaling_port,
                "streaming":      self.provider.is_webrtc_streaming(i),
                "fps":            round(s.get("current_fps", 0.0), 1),
                "width":          s.get("source_width", 0),
                "height":         s.get("source_height", 0),
                "reconnects":     s.get("reconnect_count", 0),
            })
        sys_info = {"cpu": 0.0, "ram_mb": 0.0}
        try:
            import psutil
            proc = psutil.Process(os.getpid())
            sys_info["cpu"]    = psutil.cpu_percent(interval=None)
            sys_info["ram_mb"] = proc.memory_info().rss / (1024 * 1024)
        except ImportError:
            pass
        return web.json_response({"streams": streams, "system": sys_info})

    # ── HTML Dashboard ────────────────────────────────────────────────────────

    async def handle_index(self, request):
        n              = self.num_streams
        signaling_port = self.signaling_port
        is_gpu         = self.provider.is_gpu_available()
        # Fetch the real stream_ids so the browser can use them in offer URLs.
        # e.g. ["Camera 3", "Camera 4"] instead of ["0", "1"]
        stream_ids     = await self._get_stream_ids()
        # Ensure the list has exactly n entries (pad with numeric fallback)
        while len(stream_ids) < n:
            stream_ids.append(str(len(stream_ids)))

        # Stream cards (one per camera)
        cards_html = ""
        for i in range(n):
            cards_html += f"""
            <div class="card" id="card-{i}">
              <div class="card-header">
                <span class="cam-label">CAM {i}</span>
                <span class="badges">
                  <span class="badge badge-port">:{signaling_port} id={i}</span>
                  <span class="badge badge-state" id="state-{i}">IDLE</span>
                </span>
              </div>
              <div class="video-wrap">
                <video id="video-{i}" autoplay playsinline muted></video>
                <div class="overlay" id="overlay-{i}">
                  <div class="overlay-content">
                    <div class="spinner" id="spinner-{i}" style="display:none"></div>
                    <div class="overlay-msg" id="msg-{i}">WebRTC Idle</div>
                  </div>
                </div>
              </div>
              <div class="card-footer">
                <span id="fps-{i}" class="fps-badge">-- fps</span>
                <span id="res-{i}" class="res-badge">--</span>
                <div class="btn-group">
                  <button class="btn btn-start" onclick="startStream({i})" id="btn-start-{i}">▶ Start</button>
                  <button class="btn btn-stop"  onclick="stopStream({i})"  id="btn-stop-{i}" disabled>■ Stop</button>
                </div>
              </div>
            </div>"""

        gpu_badge = "badge-gpu" if is_gpu else "badge-cpu"
        gpu_label = "GPU" if is_gpu else "CPU"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>RTSPModule — WebRTC Viewer</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
      --bg:       #060810;
      --surface:  #0d1117;
      --border:   #1e2433;
      --accent:   #00e5ff;
      --green:    #00e676;
      --red:      #ff1744;
      --amber:    #ffc107;
      --text:     #e8eaf0;
      --muted:    #4a5568;
      --radius:   10px;
    }}

    body {{
      background: var(--bg);
      color: var(--text);
      font-family: 'Inter', sans-serif;
      min-height: 100vh;
      padding: 20px 24px 40px;
    }}

    /* ── Header ── */
    .topbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 28px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border);
    }}
    .brand {{ display: flex; flex-direction: column; gap: 2px; }}
    .brand h1 {{
      font-size: 22px;
      font-weight: 700;
      background: linear-gradient(90deg, var(--accent), var(--green));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      letter-spacing: -0.5px;
    }}
    .brand sub {{
      font-size: 11px;
      color: var(--muted);
      font-family: 'JetBrains Mono', monospace;
    }}
    .topbar-right {{ display: flex; align-items: center; gap: 12px; }}
    .badge {{ display: inline-block; padding: 3px 9px; border-radius: 20px; font-size: 11px;
              font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; }}
    .badge-gpu  {{ background: linear-gradient(135deg,#76b900,#4a7a00); color:#000; }}
    .badge-cpu  {{ background: linear-gradient(135deg,#1e88e5,#0d47a1); color:#fff; }}
    .badge-webrtc {{ background: rgba(0,229,255,0.12); color: var(--accent);
                    border: 1px solid rgba(0,229,255,0.25); }}
    .badge-port   {{ background: rgba(255,193,7,0.12); color: var(--amber);
                    border: 1px solid rgba(255,193,7,0.25); font-family: 'JetBrains Mono', monospace;
                    font-size: 11px; }}
    .badge-state  {{ font-family: 'JetBrains Mono', monospace; font-size: 10px; }}
    .state-idle     {{ background: rgba(74,85,104,0.25); color: var(--muted);    border: 1px solid var(--border); }}
    .state-connect  {{ background: rgba(255,193,7,0.15); color: var(--amber);   border: 1px solid rgba(255,193,7,0.3); }}
    .state-live     {{ background: rgba(0,230,118,0.15); color: var(--green);   border: 1px solid rgba(0,230,118,0.3); }}
    .state-error    {{ background: rgba(255,23,68,0.15);  color: var(--red);    border: 1px solid rgba(255,23,68,0.3); }}

    /* Global controls */
    .global-controls {{
      display: flex;
      gap: 10px;
      margin-bottom: 24px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .btn {{
      padding: 8px 18px;
      border: none;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s;
      font-family: 'Inter', sans-serif;
    }}
    .btn:disabled {{ opacity: 0.35; cursor: not-allowed; }}
    .btn-start {{ background: linear-gradient(135deg, var(--green), #00c853); color: #000; }}
    .btn-start:hover:not(:disabled) {{ filter: brightness(1.15); transform: translateY(-1px); }}
    .btn-stop  {{ background: linear-gradient(135deg, #e53935, var(--red)); color: #fff; }}
    .btn-stop:hover:not(:disabled)  {{ filter: brightness(1.15); transform: translateY(-1px); }}
    .btn-all-start {{ background: linear-gradient(135deg, var(--accent), #0077ff); color: #000; }}
    .btn-all-stop  {{ background: linear-gradient(135deg, #9c27b0, #6a1b9a); color: #fff; }}
    .btn-all-start:hover {{ filter: brightness(1.1); }}
    .btn-all-stop:hover  {{ filter: brightness(1.1); }}

    /* System telemetry bar */
    #telemetry {{
      font-size: 12px;
      color: var(--muted);
      margin-left: auto;
      font-family: 'JetBrains Mono', monospace;
    }}

    /* ── Grid ── */
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
      gap: 16px;
    }}

    /* ── Camera card ── */
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow: hidden;
      transition: border-color 0.2s, box-shadow 0.2s;
    }}
    .card.is-live {{
      border-color: rgba(0,230,118,0.35);
      box-shadow: 0 0 24px rgba(0,230,118,0.08);
    }}
    .card-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px 14px;
      border-bottom: 1px solid var(--border);
      background: rgba(255,255,255,0.02);
    }}
    .cam-label {{
      font-size: 13px;
      font-weight: 700;
      font-family: 'JetBrains Mono', monospace;
      color: var(--accent);
    }}
    .badges {{ display: flex; gap: 6px; align-items: center; }}

    /* ── Video ── */
    .video-wrap {{
      position: relative;
      aspect-ratio: 16/9;
      background: #000;
      overflow: hidden;
    }}
    video {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }}
    .overlay {{
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(0,0,0,0.78);
      transition: opacity 0.4s;
      pointer-events: none;
    }}
    .overlay.hidden {{ opacity: 0; }}
    .overlay-content {{ text-align: center; }}
    .overlay-msg {{
      font-size: 13px;
      color: var(--muted);
      margin-top: 10px;
      font-family: 'JetBrains Mono', monospace;
    }}
    /* CSS spinner */
    .spinner {{
      width: 32px;
      height: 32px;
      border: 3px solid rgba(0,229,255,0.15);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      margin: 0 auto;
    }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

    /* ── Card footer ── */
    .card-footer {{
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 10px 14px;
      border-top: 1px solid var(--border);
      background: rgba(255,255,255,0.02);
    }}
    .fps-badge {{
      font-size: 12px;
      font-family: 'JetBrains Mono', monospace;
      color: var(--green);
      min-width: 54px;
    }}
    .res-badge {{
      font-size: 11px;
      color: var(--muted);
      font-family: 'JetBrains Mono', monospace;
      flex: 1;
    }}
    .btn-group {{ display: flex; gap: 6px; }}
    .card-footer .btn {{ padding: 5px 12px; font-size: 12px; }}

    /* ── ICE / connection log ── */
    #log {{
      margin-top: 28px;
      padding: 14px 18px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      color: var(--muted);
      max-height: 160px;
      overflow-y: auto;
      line-height: 1.7;
    }}
    #log .log-info  {{ color: var(--accent); }}
    #log .log-ok    {{ color: var(--green); }}
    #log .log-warn  {{ color: var(--amber); }}
    #log .log-error {{ color: var(--red); }}
  </style>
</head>
<body>

  <!-- ── Header ── -->
  <div class="topbar">
    <div class="brand">
      <h1>RTSPModule WebRTC Viewer</h1>
      <sub>native browser WebRTC · zero Python frame copy · {n} stream(s)</sub>
    </div>
    <div class="topbar-right">
      <span class="badge badge-webrtc">WebRTC</span>
      <span class="badge {gpu_badge}">{gpu_label} DECODE</span>
    </div>
  </div>

  <!-- ── Global controls ── -->
  <div class="global-controls">
    <button class="btn btn-all-start" onclick="startAll()">▶▶ Start All</button>
    <button class="btn btn-all-stop"  onclick="stopAll()">■■ Stop All</button>
    <span id="telemetry">cpu --% · ram -- MB</span>
  </div>

  <!-- ── Camera grid ── -->
  <div class="grid">
    {cards_html}
  </div>

  <!-- ── Connection log ── -->
  <div id="log"><span class="log-info">[dashboard]</span> Initializing…</div>

<script>
// ─── Config (injected by Python) ────────────────────────────────────────────
const NUM_STREAMS     = {n};
const SIGNALING_PORT  = {signaling_port};
const HOST            = window.location.hostname;
// STREAM_IDS maps cam_index → actual hub stream_id (e.g. "Camera-3").
// Refreshed dynamically before each WebRTC connection attempt.
let STREAM_IDS      = {json.dumps(stream_ids)};

// ─── Per-camera state ────────────────────────────────────────────────────────
const pcs     = {{}};  // cam_id → RTCPeerConnection
const streams = {{}};  // cam_id → MediaStream
const peerIds = {{}};  // cam_id → peer_id (from SDP answer, for disconnect)
const reconnectTimers = {{}}; // cam_id → timeout ID

// ─── Logging ─────────────────────────────────────────────────────────────────
const logEl = document.getElementById('log');
function log(msg, cls='log-info') {{
  const ts = new Date().toTimeString().slice(0,8);
  const line = document.createElement('div');
  line.innerHTML = `<span class="${{cls}}">${{ts}} ${{msg}}</span>`;
  logEl.appendChild(line);
  logEl.scrollTop = logEl.scrollHeight;
  if (logEl.children.length > 200) logEl.removeChild(logEl.firstChild);
}}

// ─── State helpers ────────────────────────────────────────────────────────────
function setState(camId, state) {{
  // state: 'idle' | 'connecting' | 'live' | 'error'
  const badge  = document.getElementById('state-' + camId);
  const card   = document.getElementById('card-' + camId);
  const overlay = document.getElementById('overlay-' + camId);
  const spinner = document.getElementById('spinner-' + camId);
  const msgEl  = document.getElementById('msg-' + camId);
  const btnStart = document.getElementById('btn-start-' + camId);
  const btnStop  = document.getElementById('btn-stop-' + camId);

  badge.className      = 'badge badge-state';
  card.classList.remove('is-live');

  if (state === 'idle') {{
    badge.textContent    = 'IDLE';
    badge.classList.add('state-idle');
    overlay.classList.remove('hidden');
    spinner.style.display = 'none';
    msgEl.textContent    = 'WebRTC Idle';
    btnStart.disabled    = false;
    btnStop.disabled     = true;
    if (reconnectTimers[camId]) {{ clearTimeout(reconnectTimers[camId]); delete reconnectTimers[camId]; }}
  }} else if (state === 'connecting') {{
    badge.textContent    = 'CONNECTING';
    badge.classList.add('state-connect');
    overlay.classList.remove('hidden');
    spinner.style.display = 'block';
    msgEl.textContent    = 'Connecting…';
    btnStart.disabled    = true;
    btnStop.disabled     = false;
    if (reconnectTimers[camId]) {{ clearTimeout(reconnectTimers[camId]); delete reconnectTimers[camId]; }}
  }} else if (state === 'live') {{
    badge.textContent    = 'LIVE';
    badge.classList.add('state-live');
    overlay.classList.add('hidden');
    spinner.style.display = 'none';
    card.classList.add('is-live');
    btnStart.disabled    = true;
    btnStop.disabled     = false;
    if (reconnectTimers[camId]) {{ clearTimeout(reconnectTimers[camId]); delete reconnectTimers[camId]; }}
  }} else if (state === 'error') {{
    badge.textContent    = 'ERROR';
    badge.classList.add('state-error');
    overlay.classList.remove('hidden');
    spinner.style.display = 'none';
    msgEl.textContent    = 'Error — Reconnecting in 5s…';
    btnStart.disabled    = false;
    btnStop.disabled     = true;
    
    // Auto-reconnect logic
    if (!reconnectTimers[camId]) {{
      reconnectTimers[camId] = setTimeout(() => {{
        log(`[cam ${{camId}}] Auto-reconnecting...`, 'log-warn');
        startStream(camId);
      }}, 5000);
    }}
  }}
}}

// ─── Refresh stream IDs from Python backend ──────────────────────────────────
async function refreshStreamIds() {{
  try {{
    const r = await fetch('/api/status', {{ signal: AbortSignal.timeout(3000) }});
    const d = await r.json();
    if (d.streams && d.streams.length > 0) {{
      STREAM_IDS = d.streams.map(s => s.stream_id);
      log(`[dashboard] Stream IDs refreshed: ${{JSON.stringify(STREAM_IDS)}}`);
    }}
  }} catch(e) {{
    log(`[dashboard] Could not refresh stream IDs: ${{e.message}}`, 'log-warn');
  }}
}}

// ─── WebRTC connection ────────────────────────────────────────────────────────
async function connectWebRTC(camId) {{
  // Close any existing connection cleanly
  if (pcs[camId]) {{
    pcs[camId].close();
    delete pcs[camId];
  }}

  setState(camId, 'connecting');

  // All streams share one signaling port; stream_id differentiates them.
  const signalingUrl = `http://${{HOST}}:${{SIGNALING_PORT}}`;

  // Health check — confirm the shared signaling server is up
  try {{
    await fetch(`${{signalingUrl}}/health`, {{ signal: AbortSignal.timeout(3000) }});
  }} catch(e) {{
    log(`[cam ${{camId}}] Signaling server not ready on :${{SIGNALING_PORT}} — is any stream active?`, 'log-warn');
    setState(camId, 'error');
    return;
  }}

  // Refresh stream IDs to ensure we have real hub names (not stale fallbacks)
  await refreshStreamIds();

  const pc = new RTCPeerConnection({{
    iceServers: [{{ urls: 'stun:stun.l.google.com:19302' }}]
  }});
  pcs[camId] = pc;

  // Receive video track
  pc.ontrack = (evt) => {{
    log(`[cam ${{camId}}] Track received: ${{evt.track.kind}}`, 'log-ok');
    document.getElementById('video-' + camId).srcObject = evt.streams[0];
    setState(camId, 'live');
  }};

  pc.oniceconnectionstatechange = () => {{
    const s = pc.iceConnectionState;
    log(`[cam ${{camId}}] ICE → ${{s}}`);
    if (s === 'failed' || s === 'disconnected') {{
      setState(camId, 'error');
    }} else if (s === 'closed') {{
      setState(camId, 'idle');
    }}
  }};

  pc.onconnectionstatechange = () => {{
    const s = pc.connectionState;
    if (s === 'failed') {{
      log(`[cam ${{camId}}] Connection failed`, 'log-error');
      setState(camId, 'error');
    }}
  }};

  // Create recvonly offer
  pc.addTransceiver('video', {{ direction: 'recvonly' }});

  try {{
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    // Map numeric cam index to the actual stream_id registered with the hub
    const streamId = STREAM_IDS[camId] ?? String(camId);
    log(`[cam ${{camId}}] Sending SDP offer (stream_id='${{streamId}}') to :${{SIGNALING_PORT}}`);

    const resp = await fetch(
      `${{signalingUrl}}/webrtc/offer?stream_id=${{encodeURIComponent(streamId)}}`,
      // stream_id maps this camera's numeric index to its registration name
      {{
        method:  'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body:    JSON.stringify({{ sdp: pc.localDescription.sdp, type: 'offer' }}),
        signal:  AbortSignal.timeout(8000),
      }}
    );

    if (!resp.ok) {{
      const err = await resp.text();
      log(`[cam ${{camId}}] Offer rejected: ${{err}}`, 'log-error');
      setState(camId, 'error');
      return;
    }}

    const answer = await resp.json();
    log(`[cam ${{camId}}] SDP answer received (peer_id=${{answer.peer_id || '?'}})`, 'log-ok');
    peerIds[camId] = answer.peer_id;  // Store for disconnect

    await pc.setRemoteDescription(new RTCSessionDescription({{
      type: 'answer',
      sdp:  answer.sdp,
    }}));

  }} catch(e) {{
    log(`[cam ${{camId}}] Error: ${{e.message}}`, 'log-error');
    setState(camId, 'error');
  }}
}}

async function disconnectWebRTC(camId) {{
  // Notify the Rust signaling server to close this specific peer
  const peerId = peerIds[camId];
  const streamId = STREAM_IDS[camId] ?? String(camId);
  if (peerId && streamId) {{
    try {{
      await fetch(`http://${{HOST}}:${{SIGNALING_PORT}}/webrtc/disconnect`, {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ peer_id: peerId, stream_id: streamId }}),
        signal: AbortSignal.timeout(3000),
      }});
      log(`[cam ${{camId}}] Peer ${{peerId.slice(0,8)}} disconnected`, 'log-warn');
    }} catch(e) {{
      log(`[cam ${{camId}}] Disconnect request failed: ${{e.message}}`, 'log-warn');
    }}
  }}
  // Close local peer connection
  if (pcs[camId]) {{
    pcs[camId].close();
    delete pcs[camId];
  }}
  delete peerIds[camId];
  if (reconnectTimers[camId]) {{
    clearTimeout(reconnectTimers[camId]);
    delete reconnectTimers[camId];
  }}
  const v = document.getElementById('video-' + camId);
  v.srcObject = null;
  setState(camId, 'idle');
}}

// ─── Control API calls (to Python backend) ────────────────────────────────────
async function startStream(camId) {{
  log(`[cam ${{camId}}] Requesting start_streaming from Python…`);
  try {{
    const r = await fetch(`/api/stream/${{camId}}/start`, {{ method: 'POST' }});
    const d = await r.json();
    log(`[cam ${{camId}}] Streaming started (stream_id=${{d.stream_id}}) on port ${{d.signaling_port}}`, 'log-ok');
    // Give GStreamer a brief moment to bring up the branch
    await new Promise(res => setTimeout(res, 600));
    await connectWebRTC(camId);
  }} catch(e) {{
    log(`[cam ${{camId}}] start_streaming failed: ${{e.message}}`, 'log-error');
    setState(camId, 'error');
  }}
}}

async function stopStream(camId) {{
  // Only close this viewer's peer connection — do NOT tear down the GStreamer branch
  await disconnectWebRTC(camId);
}}

async function startAll() {{
  log(`[dashboard] Start all ${{NUM_STREAMS}} streams on shared port ${{SIGNALING_PORT}}…`);
  await fetch('/api/streams/start_all', {{ method: 'POST' }}).catch(() => {{ /* empty */ }});
  await new Promise(res => setTimeout(res, 800));
  for (let i = 0; i < NUM_STREAMS; i++) await connectWebRTC(i);
}}

async function stopAll() {{
  log('[dashboard] Disconnecting all peers…', 'log-warn');
  for (let i = 0; i < NUM_STREAMS; i++) await disconnectWebRTC(i);
}}

// ─── Status polling ─────────────────────────────────────────────────────────
async function pollStatus() {{
  try {{
    const r = await fetch('/api/status');
    const d = await r.json();
    d.streams.forEach(s => {{
      const fpsEl = document.getElementById('fps-' + s.id);
      const resEl = document.getElementById('res-' + s.id);
      if (fpsEl) fpsEl.textContent = s.fps.toFixed(1) + ' fps';
      if (resEl) resEl.textContent = s.width && s.height ? `${{s.width}}×${{s.height}}` : '--';
    }});
    document.getElementById('telemetry').textContent =
      `cpu ${{d.system.cpu.toFixed(1)}}% · ram ${{d.system.ram_mb.toFixed(0)}} MB`;
  }} catch(e) {{ /* empty */ }}
}}

// ─── Init ─────────────────────────────────────────────────────────────────────
(async () => {{
  for (let i = 0; i < NUM_STREAMS; i++) setState(i, 'idle');
  log('[dashboard] Ready — click ▶ Start on a camera or ▶▶ Start All', 'log-ok');
  setInterval(pollStatus, 2000);
}})();
</script>

</body>
</html>"""
        return web.Response(text=html, content_type='text/html')


# ─── App Setup ────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="RTSPModule WebRTC Dashboard")
    parser.add_argument("--config",          default=DEFAULT_CONFIG,         help="Path to config.yaml")
    parser.add_argument("--port",            type=int, default=DEFAULT_WEB_PORT,       help="Dashboard HTTP port")
    parser.add_argument("--signaling-port",  type=int, default=DEFAULT_SIGNALING_PORT, help="Shared WebRTC signaling port (all cameras)")
    # Legacy alias kept for backward compatibility
    parser.add_argument("--base-port",       type=int, default=None,                   help="Alias for --signaling-port (deprecated)")
    args = parser.parse_args()
    # Prefer --signaling-port; fall back to --base-port if only that was given
    if args.base_port is not None and args.signaling_port == DEFAULT_SIGNALING_PORT:
        args.signaling_port = args.base_port

    server = WebRtcServer(args.config, args.port, args.signaling_port)

    if not await server.start():
        print("[ERROR] Failed to start RTSP streams. Exiting.")
        return

    app = web.Application()

    # Dashboard
    app.router.add_get("/", server.handle_index)

    # Per-stream controls
    app.router.add_post("/api/stream/{cam_id}/start", server.handle_start_stream)
    app.router.add_post("/api/stream/{cam_id}/stop",  server.handle_stop_stream)
    app.router.add_post("/api/streams/start_all",     server.handle_start_all)
    app.router.add_post("/api/streams/stop_all",      server.handle_stop_all)

    # Status / telemetry
    app.router.add_get("/api/status", server.handle_api_status)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", args.port)
    await site.start()

    print()
    print(f"  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │  RTSPModule WebRTC Dashboard                            │")
    print(f"  │  Dashboard    → http://localhost:{args.port:<5}                  │")
    print(f"  │  Signaling    → http://localhost:{args.signaling_port:<5} (all cameras)       │")
    print(f"  │  Streams      → GET http://localhost:{args.signaling_port}/streams             │")
    print(f"  └─────────────────────────────────────────────────────────┘")
    print()

    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        print("\n[INFO] Shutting down…")
        server.stop()
        await runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
