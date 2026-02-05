#!/usr/bin/env python3
"""
Ultra-High Performance WebSocket Web Viewer for RTSPModule
Broadcasting at NATIVE FPS.
Uses per-stream acquisition threads and per-client output queues for maximum parallelism.
"""

import os
import sys
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import struct
import collections
from aiohttp import web, WSMsgType
import simplejpeg

# Add src directory to path for lib import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from rtspmodule import RTSPModule

# Global configuration
CONFIG_PATH = "configs/config.yaml"
WEB_PORT = 8080
JPEG_QUALITY = 40 
STREAM_QUEUE_SIZE = 4 # Increased for smoother handoff
CLIENT_QUEUE_SIZE = 10 # Buffer per client to handle network jitter

class StreamServer:
    def __init__(self):
        self.provider = RTSPModule()
        self.running = True
        self.clients = collections.defaultdict(dict) # cam_id -> {ws -> Queue}
        self.num_streams = 0
        self.stream_queues = {} # cam_id -> asyncio.Queue
        self.executor = None
        self.last_frame_ids = {} # cam_id -> last processed frame_id

    async def start(self):
        if not os.path.exists(CONFIG_PATH):
            print(f"[ERROR] Config file not found at {CONFIG_PATH}")
            return False
            
        self.provider.start(CONFIG_PATH)
        await asyncio.sleep(2)
        self.num_streams = self.provider.stream_count()
        print(f"[INFO] Started {self.num_streams} streams at Full Resolution / Native FPS.")
        
        self.executor = ThreadPoolExecutor(max_workers=self.num_streams, thread_name_prefix="AcqThread")
        loop = asyncio.get_running_loop()

        # Start workers per stream
        for i in range(self.num_streams):
            self.stream_queues[i] = asyncio.Queue(maxsize=STREAM_QUEUE_SIZE)
            # 1. Start Acquisition Thread (Producer)
            loop.run_in_executor(self.executor, self.acquisition_worker, i, loop)
            # 2. Start Broadcaster Task (Consumer)
            asyncio.create_task(self.stream_broadcaster(i))
            
        return True

    def _safe_put(self, cam_id, packet):
        """Helper to push to queue without raising QueueFull on the event loop."""
        try:
            self.stream_queues[cam_id].put_nowait(packet)
        except asyncio.QueueFull:
            pass

    def acquisition_worker(self, cam_id, loop):
        """
        DEDICATED THREAD: Continuous frame acquisition and JPEG encoding.
        Runs outside the GIL where possible (simplejpeg).
        """
        print(f"[DEBUG] Started Acquisition Thread for Cam {cam_id}")
        while self.running:
            # Check if any client is watching this specific stream - save CPU if idling
            if not self.clients[cam_id]:
                time.sleep(0.1)
                continue

            # Get frame from C++ backend
            frame_data = self.provider.get_cpu_frame(cam_id, timeout_ms=30)
            
            if not frame_data.get('valid', False):
                time.sleep(0.001)
                continue
            
            frame_id = frame_data.get('frame_id', -1)
            if frame_id == self.last_frame_ids.get(cam_id, -1):
                # Skip duplicate frame to save CPU/Network
                time.sleep(0.001) 
                continue
            
            self.last_frame_ids[cam_id] = frame_id
                
            data = frame_data['data']
            fmt = frame_data.get('format', 'BGR')
            
            # Format Compatibility: simplejpeg expects a 3D array (H, W, C)
            # If we get NV12 (2D array), we must convert it.
            try:
                if fmt == 'NV12':
                    bgr = cv2.cvtColor(data, cv2.COLOR_YUV2BGR_NV12)
                elif data.ndim == 2:
                    # Fallback for any other 2D planar formats
                    bgr = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
                else:
                    bgr = data

                # JPEG Encode FULL RESOLUTION in this thread
                bgr=cv2.resize(bgr, (640,360))
                jpeg_bytes = simplejpeg.encode_jpeg(
                    bgr, 
                    quality=JPEG_QUALITY, 
                    colorspace='BGR', 
                    colorsubsampling='420',
                    fastdct=True
                )
                
                # Packet: [CamID (1B)][FrameID (4B LE)][JPEG...]
                packet = bytes([cam_id]) + struct.pack("<I", frame_id & 0xFFFFFFFF) + jpeg_bytes
                
                # Push to async queue safely via helper to avoid QueueFull tracebacks
                loop.call_soon_threadsafe(self._safe_put, cam_id, packet)
            except Exception as e:
                print(f"[ERROR] Worker {cam_id} ({fmt}) failed: {e}")
                
    async def stream_broadcaster(self, cam_id):
        """
        ASYNC TASK: Pulls encoded frames from its stream queue and fayout to clients.
        """
        queue = self.stream_queues[cam_id]
        while self.running:
            packet = await queue.get()
            
            # Fan out to active clients for this specific camera
            targets = list(self.clients[cam_id].values())
            for client_q in targets:
                try:
                    client_q.put_nowait(packet)
                except asyncio.QueueFull:
                    pass # Slow client - drop frame for them specifically

    async def handle_index(self, request):
        num_streams = self.num_streams
        is_gpu = self.provider.is_gpu_available()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RTSP Module - ParallelStream Dashboard</title>
            <style>
                body {{ background: #080808; color: #f5f5f5; font-family: 'Inter', sans-serif; margin: 0; padding: 20px; }}
                .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 15px; }}
                .container {{ background: #111; border-radius: 8px; overflow: hidden; position: relative; border: 1px solid #1a1a1a; }}
                canvas {{ width: 100%; height: auto; display: block; background: #000; aspect-ratio: 16/9; }}
                .info {{ position: absolute; top: 0; left: 0; right: 0; padding: 12px; background: linear-gradient(rgba(0,0,0,0.85), transparent); font-size: 13px; display: flex; justify-content: space-between; z-index: 10; }}
                .badge {{ padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; text-transform: uppercase; border: 1px solid rgba(255,255,255,0.1); }}
                .badge-gpu {{ background: #76b900; color: #000; }}
                .badge-cpu {{ background: #1e88e5; color: #fff; }}
                .stats-panel {{ margin-top: 25px; padding: 18px; background: #111; border-radius: 8px; border-left: 5px solid #76b900; font-size: 14px; color: #888; }}
                h1 {{ font-weight: 200; letter-spacing: -0.5px; margin: 0; }}
                .status-dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; }}
                .status-live {{ background: #76b900; box-shadow: 0 0 10px #76b900; }}
            </style>
        </head>
        <body>
            <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 25px;">
                <div>
                    <h1>RTSP Module</h1>
                    <div style="font-size: 12px; color: #666; margin-top: 4px;">Python Parallel-Thread Protocol • Multi-Endpoint Streaming</div>
                </div>
                <div style="text-align: right;">
                    <span class="badge {'badge-gpu' if is_gpu else 'badge-cpu'}">{'GPU' if is_gpu else 'CPU'} DECODER</span>
                    <div id="ws-status" style="font-size: 11px; color: #555; margin-top: 6px;"><span class="status-dot"></span>CONNECTING...</div>
                </div>
            </div>
            <div class="grid">
        """
        for i in range(num_streams):
            html += f"""
            <div class="container">
                <div class="info">
                    <span><strong>CAM {i}</strong></span>
                    <span style="font-weight: bold;">
                        <span id="id-{i}" style="color: #f44; margin-right: 12px;">--</span>
                        <span id="fps-{i}" style="color: #76b900;">-- FPS</span>
                    </span>
                </div>
                <canvas id="canvas-{i}"></canvas>
            </div>
            """
        
        html += """
            </div>
            <div class="stats-panel" id="global-stats">Synchronizing telemetry...</div>
            <script>
                const numStreams = """ + str(num_streams) + """;
                const canvases = [];
                const ctxs = [];
                const fpsCounts = new Array(numStreams).fill(0);
                
                for(let i=0; i<numStreams; i++) {
                    const c = document.getElementById('canvas-' + i);
                    canvases.push(c);
                    ctxs.push(c.getContext('2d', {alpha: false, desynchronized: true}));
                }

                function connect(camId) {
                    const ws = new WebSocket((window.location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + window.location.host + '/ws/' + camId);
                    const statusEl = document.getElementById('ws-status');

                    ws.binaryType = 'arraybuffer';
                    ws.onopen = () => { 
                        statusEl.innerHTML = '<span class="status-dot status-live"></span>• BROADCAST LIVE';
                        statusEl.style.color = '#76b900'; 
                    };
                    ws.onclose = () => { 
                        // Only show red if all are closed, but for simplicity:
                        statusEl.innerHTML = '<span class="status-dot"></span>• LINK SEVERED';
                        statusEl.style.color = '#f44'; 
                        setTimeout(() => connect(camId), 1000); 
                    };

                    ws.onmessage = (evt) => {
                        const data = new Uint8Array(evt.data);
                        // Header: [CamID (1B)][FrameID (4B LE)]
                        // We still have the camId in the packet for verification
                        const packetCamId = data[0];
                        if (packetCamId !== camId) return;

                        const view = new DataView(evt.data);
                        const frameId = view.getUint32(1, true);
                        const jpegData = data.subarray(5);

                        const blob = new Blob([jpegData], {type: 'image/jpeg'});
                        // OPTIMIZATION: Use createImageBitmap for off-main-thread decoding
                        createImageBitmap(blob).then(bitmap => {
                            const canvas = canvases[camId];
                            if (canvas.width !== bitmap.width) {
                                canvas.width = bitmap.width;
                                canvas.height = bitmap.height;
                            }
                            ctxs[camId].drawImage(bitmap, 0, 0);
                            bitmap.close(); 
                            fpsCounts[camId]++;
                            
                            const idEl = document.getElementById('id-' + camId);
                            if (idEl) idEl.innerText = frameId;
                        }).catch(e => {
                             // Fallback to Image if bitmap fails (rare)
                            const url = URL.createObjectURL(blob);
                            const img = new Image();
                            img.onload = () => {
                                const canvas = canvases[camId];
                                if (canvas.width !== img.width) {
                                    canvas.width = img.width;
                                    canvas.height = img.height;
                                }
                                ctxs[camId].drawImage(img, 0, 0);
                                URL.revokeObjectURL(url);
                                fpsCounts[camId]++;
                                
                                const idEl = document.getElementById('id-' + camId);
                                if (idEl) idEl.innerText = frameId;
                            };
                            img.src = url;
                        });
                    };
                }

                function refreshUi() {
                    fpsCounts.forEach((count, i) => {
                        const el = document.getElementById('fps-' + i);
                        if (el) el.innerText = count + ' FPS';
                        fpsCounts[i] = 0;
                    });
                    
                    fetch('/api/stats').then(r => r.json()).then(data => {
                        const total = data.streams.reduce((a, b) => a + b.fps, 0);
                        document.getElementById('global-stats').innerHTML = 
                            `⚡ <strong>System Throughput:</strong> ${total.toFixed(0)} FPS | ` +
                            `<strong>CPU:</strong> ${data.system.cpu.toFixed(1)}% | ` +
                            `<strong>Memory:</strong> ${data.system.ram_mb.toFixed(0)} MB`;
                    }).catch(e => {});
                }

                setInterval(refreshUi, 1000);
                for(let i=0; i<numStreams; i++) {
                    connect(i);
                }
            </script>
        </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')

    async def handle_ws(self, request):
        cam_id = int(request.match_info.get('cam_id', 0))
        if cam_id >= self.num_streams:
            return web.Response(status=404, text=f"Camera {cam_id} not found")

        ws = web.WebSocketResponse(heartbeat=30.0)
        await ws.prepare(request)
        
        # Per-client queue to decouple transport from broadcaster
        q = asyncio.Queue(maxsize=CLIENT_QUEUE_SIZE)
        self.clients[cam_id][ws] = q
        
        # Dedicated pusher task for this client
        pusher = asyncio.create_task(self.ws_pusher(ws, q))
        
        print(f"[INFO] New client connected for Cam {cam_id}. Total: {len(self.clients[cam_id])}")
        try:
            async for msg in ws:
                if msg.type == WSMsgType.ERROR: break
        finally:
            self.clients[cam_id].pop(ws, None)
            pusher.cancel()
            try:
                await pusher
            except asyncio.CancelledError:
                pass
            print(f"[INFO] Client disconnected from Cam {cam_id}. Total: {len(self.clients[cam_id])}")
            
        return ws

    async def ws_pusher(self, ws, q):
        """Async task pulling from the client queue and sending over socket."""
        try:
            while True:
                packet = await q.get()
                await ws.send_bytes(packet)
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        except Exception as e:
            # Silence common disconnect errors to keep logs clean
            if "closing transport" in str(e) or "Connection reset" in str(e):
                pass
            else:
                print(f"[DEBUG] Pusher closed: {e}")

    async def handle_stats(self, request):
        stream_stats = []
        for i in range(self.num_streams):
            s = self.provider.get_stats(i)
            stream_stats.append({"id": i, "fps": s.get('current_fps', 0.0)})
        sys_stats = {"cpu": 0, "ram_mb": 0}
        try:
            import psutil
            process = psutil.Process(os.getpid())
            sys_stats["cpu"] = psutil.cpu_percent()
            sys_stats["ram_mb"] = process.memory_info().rss / (1024 * 1024)
        except: pass
        return web.json_response({"streams": stream_stats, "system": sys_stats})

    def stop(self):
        self.running = False
        if self.executor:
            self.executor.shutdown(wait=False)
        self.provider.stop()

async def main():
    server = StreamServer()
    if not await server.start(): return
    app = web.Application()
    app.router.add_get('/', server.handle_index)
    app.router.add_get('/ws/{cam_id}', server.handle_ws)
    app.router.add_get('/api/stats', server.handle_stats)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', WEB_PORT)
    print(f"[SUCCESS] Parallelized Ultra-Viewer: http://localhost:{WEB_PORT}")
    await site.start()
    try:
        while True: await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError): pass
    finally:
        server.stop()
        await runner.cleanup()

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: pass
