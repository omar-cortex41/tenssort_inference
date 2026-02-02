#!/usr/bin/env python3
"""
Multi-Stream Video Handler (SHARED DETECTOR + LOGGING)

Architecture:
- ONE shared detector (cached in GPU memory)
- Each stream has its own tracker (independent track IDs)
- Mutex ensures thread-safe detector access

Usage:
    python multi_stream.py

Configure streams in config/config.yaml under 'streams' section.
"""

import sys
import time
import yaml
import threading
import argparse
from queue import Queue, Empty
from dataclasses import dataclass
from typing import List, Dict, Optional
sys.path.insert(0, 'trt_detector/build')

import cv2
import numpy as np
from trt_detector import DetectorService, ModelConfig


# ============================================================================
# SHARED DETECTOR SERVICE (Dedicated inference thread)
# ============================================================================

class SharedDetector:
    """
    Singleton detector with a dedicated inference thread.

    CRITICAL: Both model loading AND inference happen on the SAME thread.
    This is required because CUDA contexts are thread-specific.

    Benefits:
    - Only ONE model loaded in GPU memory
    - All CUDA operations on same thread (avoids context issues)
    - Thread-safe by design (queue-based)
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._detector = None
        self._config = None
        self._request_queue = Queue(maxsize=100)
        self._inference_thread = None
        self._running = False
        self._ready = threading.Event()  # Signal when model is loaded

    def load(self, config: ModelConfig) -> bool:
        """Start inference thread which will load the model"""
        if self._running:
            return True  # Already running

        self._config = config
        self._running = True

        # Start dedicated inference thread (it will load the model)
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()

        # Wait for model to be loaded (with timeout)
        print("[SharedDetector] Waiting for model to load on inference thread...")
        if not self._ready.wait(timeout=30.0):
            print("[SharedDetector] Timeout waiting for model load!")
            self._running = False
            return False

        return self._detector is not None

    def _inference_loop(self):
        """Dedicated thread that loads model AND processes all inference requests"""
        # =====================================================================
        # LOAD MODEL ON THIS THREAD (same thread that will use CUDA)
        # =====================================================================
        print("[SharedDetector] Loading model on inference thread...")
        self._detector = DetectorService()
        if not self._detector.load_model(self._config):
            print("[SharedDetector] Failed to load model!")
            self._detector = None
            self._ready.set()  # Signal even on failure
            return
        print("[SharedDetector] Model loaded and cached in GPU memory")
        self._ready.set()  # Signal that we're ready

        # =====================================================================
        # INFERENCE LOOP (all CUDA operations on THIS thread)
        # =====================================================================
        while self._running:
            try:
                # Get request from queue (blocking with timeout)
                frame, result_queue = self._request_queue.get(timeout=0.1)

                # Run inference on THIS thread (same CUDA context as load)
                detections = self._detector.detect(frame)

                # Send result back
                result_queue.put(detections)
            except Empty:
                continue
            except Exception as e:
                print(f"[SharedDetector] Inference error: {e}")

    def detect(self, frame) -> List:
        """Submit frame for detection and wait for result"""
        if self._detector is None or not self._running:
            return []

        # Create a queue for this request's result
        result_queue = Queue(maxsize=1)

        # Submit request
        self._request_queue.put((frame, result_queue))

        # Wait for result
        try:
            return result_queue.get(timeout=5.0)
        except Empty:
            print("[SharedDetector] Detection timeout!")
            return []

    def is_loaded(self) -> bool:
        return self._detector is not None and self._running

    def stop(self):
        """Stop the inference thread"""
        self._running = False
        if self._inference_thread:
            self._inference_thread.join(timeout=2.0)

# Load configuration
with open('config/config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# Colors for track IDs
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
]

def get_color(track_id):
    return COLORS[track_id % len(COLORS)]


@dataclass
class FrameResult:
    """Result from processing a single frame"""
    stream_id: int                    # Which camera/stream
    frame: np.ndarray                 # The frame image
    detections: List                  # Raw detections
    fps: float                        # Actual FPS
    timestamp: float                  # When frame was captured


class VideoStream(threading.Thread):
    """
    Handles a single video/camera stream in its own thread.
    Uses SHARED detector (singleton).
    """

    def __init__(self, stream_id: int, source: str, name: str, output_queue: Queue):
        super().__init__(daemon=True)
        self.stream_id = stream_id
        self.source = source
        self.name = name
        self.output_queue = output_queue  # Queue to send results to main thread

        self.cap = None
        self.video_fps = 30
        self.width = 0
        self.height = 0
        self.running = False
        self.frame_count = 0

        # For accurate FPS calculation
        self.last_fps_time = time.time()
        self.fps_frame_count = 0
        self.actual_fps = 0.0

    def _setup_video(self) -> bool:
        """Open video source"""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            print(f"[Stream {self.stream_id}] Failed to open: {self.source}")
            return False

        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[Stream {self.stream_id}] Ready: {self.name} ({self.width}x{self.height} @ {self.video_fps:.1f} fps)")
        return True

    def run(self):
        """Thread main loop - processes frames continuously"""
        # Get shared detector (singleton - already loaded)
        shared_detector = SharedDetector()
        if not shared_detector.is_loaded():
            print(f"[Stream {self.stream_id}] Shared detector not loaded!")
            return

        if not self._setup_video():
            return

        self.running = True
        self.last_fps_time = time.time()
        print(f"[Stream {self.stream_id}] Using shared detector")

        while self.running:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                # Loop video for simulation
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    break

            # Detect (thread-safe via queue in SharedDetector)
            detections = shared_detector.detect(frame)

            # Update FPS
            self.frame_count += 1
            self.fps_frame_count += 1
            now = time.time()
            elapsed = now - self.last_fps_time
            if elapsed >= 1.0:
                self.actual_fps = self.fps_frame_count / elapsed
                self.fps_frame_count = 0
                self.last_fps_time = now

            # Send result to main thread (non-blocking, drop if queue full)
            result = FrameResult(
                stream_id=self.stream_id,
                frame=frame,
                detections=detections,
                fps=self.actual_fps,
                timestamp=now
            )

            try:
                # Replace old frame if queue is full (keep latest)
                if self.output_queue.full():
                    try:
                        self.output_queue.get_nowait()
                    except Empty:
                        pass
                self.output_queue.put_nowait(result)
            except:
                pass

        # Cleanup
        if self.cap:
            self.cap.release()

    def stop(self):
        """Stop the stream"""
        self.running = False


class MultiStreamManager:
    """Manages multiple video streams running in parallel threads"""

    def __init__(self):
        self.streams: Dict[int, VideoStream] = {}
        self.queues: Dict[int, Queue] = {}
        self.latest_results: Dict[int, FrameResult] = {}

    def add_stream(self, stream_id: int, source: str, name: str = None) -> bool:
        """Add a video stream (starts its own thread with dedicated detector)"""
        if name is None:
            name = f"Camera {stream_id}"

        # Each stream gets a queue for its results (size 1 = keep only latest)
        queue = Queue(maxsize=2)
        self.queues[stream_id] = queue

        # Create stream thread
        stream = VideoStream(stream_id, source, name, queue)
        self.streams[stream_id] = stream
        return True

    def start_all(self):
        """Start all stream threads"""
        print(f"\nStarting {len(self.streams)} stream(s) in parallel...\n")
        for stream in self.streams.values():
            stream.start()

    def get_latest_results(self) -> Dict[int, FrameResult]:
        """Get latest frame from each stream (non-blocking)"""
        for stream_id, queue in self.queues.items():
            try:
                # Get latest result without blocking
                result = queue.get_nowait()
                self.latest_results[stream_id] = result
            except Empty:
                pass  # Keep previous result
        return self.latest_results

    def stop_all(self):
        """Stop all streams"""
        for stream in self.streams.values():
            stream.stop()
        # Wait for threads to finish
        for stream in self.streams.values():
            stream.join(timeout=2.0)


def draw_results(frame: np.ndarray, result: FrameResult, stream_name: str) -> np.ndarray:
    """Draw detections on frame"""
    # Draw all detections
    for det in result.detections:
        x, y, w, h = det.x, det.y, det.width, det.height
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{det.label} {det.confidence:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y - lh - 6), (x + lw, y), (0, 255, 0), -1)
        cv2.putText(frame, label, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw stream info
    cv2.putText(frame, f"{stream_name}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {result.fps:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Detections: {len(result.detections)}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame


# ============================================================================
# MAIN
# ============================================================================

def run_logging_mode(mgr):
    """Logging mode - output detections and FPS to console"""
    print("\n" + "=" * 80)
    print("Starting inference... Press Ctrl+C to stop")
    print("=" * 80 + "\n")

    # Track stats per stream
    stream_stats = {sid: {'frames': 0, 'total_time': 0} for sid in mgr.streams}
    last_log_time = {sid: time.time() for sid in mgr.streams}

    try:
        while True:
            results = mgr.get_latest_results()

            if not results:
                time.sleep(0.01)
                continue

            for stream_id, result in results.items():
                stream = mgr.streams[stream_id]
                stats = stream_stats[stream_id]
                stats['frames'] += 1

                # Log every frame
                det_summary = ", ".join([
                    f"{cfg['class_names'][d.class_id] if d.class_id < len(cfg['class_names']) else d.class_id}:{d.confidence:.2f}"
                    for d in result.detections[:5]
                ])
                if len(result.detections) > 5:
                    det_summary += f", ... (+{len(result.detections)-5} more)"

                print(f"[{stream.name}] Frame {stats['frames']:5d} | "
                      f"FPS: {result.fps:5.1f} | "
                      f"Detections: {len(result.detections):3d} | {det_summary}")

    except KeyboardInterrupt:
        print("\n\nStopping...")

    # Final stats
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    for stream_id, stream in mgr.streams.items():
        stats = stream_stats[stream_id]
        print(f"[{stream.name}] Processed {stats['frames']} frames")
    print("=" * 80)

    mgr.stop_all()
    SharedDetector().stop()


def main():
    print("=" * 60)
    print("Multi-Stream TensorRT Detector")
    print("=" * 60)

    # Get streams from config
    streams_cfg = cfg.get('streams', [])
    if not streams_cfg:
        streams_cfg = [
            {'id': 0, 'source': cfg['video']['path'], 'name': 'Camera 1'},
        ]
        print("No 'streams' in config, using single video source")

    # =========================================================================
    # Initialize SHARED DETECTOR (ONE model for all streams)
    # =========================================================================
    shared_detector = SharedDetector()
    det_config = ModelConfig(
        cfg['model']['engine_path'],
        cfg['class_names'],
        conf_threshold=cfg['model']['conf_threshold'],
        nms_threshold=cfg['model']['nms_threshold']
    )
    if not shared_detector.load(det_config):
        print("Failed to load shared detector!")
        return

    print(f"\n{'='*50}")
    print(f"  SHARED DETECTOR: 1 model for {len(streams_cfg)} stream(s)")
    print(f"  VRAM savings: ~{(len(streams_cfg)-1) * 80}MB (vs separate detectors)")
    print(f"{'='*50}\n")

    # Setup manager
    mgr = MultiStreamManager()

    # Add streams
    for s in streams_cfg:
        mgr.add_stream(s['id'], s['source'], s.get('name', f"Camera {s['id']}"))

    if not mgr.streams:
        print("No streams available!")
        return

    # Start all stream threads
    mgr.start_all()

    print(f"Processing {len(mgr.streams)} stream(s) with SHARED DETECTOR...")

    run_logging_mode(mgr)

    print("\nDone!")


if __name__ == "__main__":
    main()
