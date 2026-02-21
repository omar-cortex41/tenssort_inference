
import sys
import os
import time
import threading
import queue
import cv2
import numpy as np
import yaml
import hashlib
from collections import deque

# Add TRT detector path
sys.path.insert(0, 'trt_detector/build')
from trt_detector import DetectorService, ModelConfig

# Try to import RTSPModule
RTSP_MODULE_AVAILABLE = False
try:
    sys.path.insert(0, 'RTSPModule-main/src')
    import rtspmodule
    RTSP_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: RTSPModule not available: {e}")
    print("         Build RTSPModule first or use multi_stream_cpp.py instead.")
    print("         See docstring for GStreamer installation instructions.\n")


def calculate_grid_layout(num_streams, max_width=1920, max_height=1080):
    """Calculate optimal grid layout for any number of streams"""
    import math

    if num_streams == 0:
        return 1, 1, 640, 360

    # Calculate grid dimensions (prefer wider layouts)
    cols = math.ceil(math.sqrt(num_streams))
    rows = math.ceil(num_streams / cols)

    # Calculate cell size to fit within max dimensions
    cell_w = max_width // cols
    cell_h = max_height // rows

    # Maintain 16:9 aspect ratio
    if cell_w / cell_h > 16/9:
        cell_w = int(cell_h * 16 / 9)
    else:
        cell_h = int(cell_w * 9 / 16)

    return rows, cols, cell_w, cell_h


class DisplayThread:
    """Separate thread for display - handles resize and drawing (Fix #2)"""
    def __init__(self, num_streams):
        self.queue = queue.Queue(maxsize=2)
        self.running = True
        self.num_streams = num_streams
        self.rows, self.cols, self.cell_w, self.cell_h = calculate_grid_layout(num_streams)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            try:
                frame_data, stats = self.queue.get(timeout=0.1)

                # Create grid
                grid_h = self.cell_h * self.rows
                grid_w = self.cell_w * self.cols
                grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

                for i, data in enumerate(frame_data):
                    if data is None:
                        continue

                    frame, detections = data
                    if frame is None:
                        continue

                    # OPTIMIZATION: Use INTER_LINEAR instead of INTER_NEAREST for better quality
                    # INTER_LINEAR is faster than INTER_CUBIC and looks better than INTER_NEAREST
                    disp = cv2.resize(frame, (self.cell_w, self.cell_h),
                                     interpolation=cv2.INTER_LINEAR)

                    # Draw detections with scaling
                    scale_x = self.cell_w / frame.shape[1]
                    scale_y = self.cell_h / frame.shape[0]
                    for det in detections:
                        x1 = int(det.x * scale_x)
                        y1 = int(det.y * scale_y)
                        x2 = int((det.x + det.width) * scale_x)
                        y2 = int((det.y + det.height) * scale_y)
                        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 165, 255), 1)
                        label = f"{det.label} {det.confidence:.2f}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                        cv2.rectangle(disp, (x1, y1 - th - 4), (x1 + tw + 4, y1), (0, 255, 0), -1)
                        cv2.putText(disp, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX,
                                   0.35, (0, 0, 0), 1, cv2.LINE_AA)

                    # Add stream label
                    cv2.putText(disp, f"Stream {i}", (5, 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                    r, c = i // self.cols, i % self.cols
                    y1, y2 = r * self.cell_h, (r + 1) * self.cell_h
                    x1, x2 = c * self.cell_w, (c + 1) * self.cell_w
                    grid[y1:y2, x1:x2] = disp

                # Add stats bar at bottom
                cv2.putText(grid, stats, (10, grid_h - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.imshow("Multi-Stream Detection", grid)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
            except queue.Empty:
                continue

    def show(self, frame_data, stats):
        """frame_data is list of (frame, detections) tuples"""
        try:
            self.queue.put_nowait((frame_data, stats))
        except queue.Full:
            pass

    def stop(self):
        self.running = False
        cv2.destroyAllWindows()


def draw_detections_scaled(frame, detections, scale_x, scale_y):
    """Draw detection boxes with scaling"""
    for det in detections:
        x1 = int(det.x * scale_x)
        y1 = int(det.y * scale_y)
        x2 = int((det.x + det.width) * scale_x)
        y2 = int((det.y + det.height) * scale_y)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 1)
        label = f"{det.label} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.35, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def generate_rtsp_config(streams, output_path):
    """Generate RTSPModule config from main config streams."""
    rtsp_config = {
        'settings': {
            'buffer_size': 8,  # OPTIMIZATION: Increased from 3 to 8 for better buffering
            'retry_max_attempts': -1,  # -1 = infinite retries (loop videos)
            'backoff_multiplier': 2,
            'gpu_id': 0,
            'log_base_path': './logs',
            'cpu_buffer_enabled': True,
            'cpu_buffer_duration_sec': 3.0,  # OPTIMIZATION: Increased from 2.0 to 3.0
            'output_format': 'NV12',
            'decoder_preference': 'auto'
        },
        'streams': []
    }

    for stream in streams:
        # Convert relative paths to absolute
        source = stream['source']
        if not source.startswith(('rtsp://', 'http://', 'file://', '/')):
            source = os.path.abspath(source)

        rtsp_config['streams'].append({
            'name': stream.get('name', f"Stream {stream.get('id', len(rtsp_config['streams']))}"),
            'url': source
        })

    with open(output_path, 'w') as f:
        yaml.dump(rtsp_config, f, default_flow_style=False)

    return output_path


def main():
    if not RTSP_MODULE_AVAILABLE:
        print("ERROR: RTSPModule is required but not available.")
        print("\nTo build RTSPModule:")
        print("  1. Install GStreamer dev packages:")
        print("     sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \\")
        print("                      gstreamer1.0-plugins-good gstreamer1.0-plugins-bad")
        print("  2. Build RTSPModule:")
        print("     cd RTSPModule-main && mkdir build && cd build && cmake .. && make -j")
        print("\nAlternatively, use multi_stream_cpp.py for local video files.")
        return

    # Load detector config
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Get streams from main config
    streams = config.get('streams', [])
    if not streams:
        print("ERROR: No streams defined in config/config.yaml")
        print("Add streams like:")
        print("  streams:")
        print("    - id: 0")
        print("      source: videos/vid1.mp4")
        print("      name: Camera 1")
        return

    # Generate RTSPModule config from main config streams
    rtsp_config_path = os.path.abspath("RTSPModule-main/configs/generated_config.yaml")
    generate_rtsp_config(streams, rtsp_config_path)

    print("=" * 60)
    print("Multi-Stream Detection (RTSPModule + TensorRT)")
    print("=" * 60)
    print(f"\nStreams from config/config.yaml:")
    for s in streams:
        print(f"  [{s.get('id', '?')}] {s.get('name', 'Unknown')}: {s['source']}")

    # Initialize RTSPModule
    print("\n[1] Initializing RTSPModule...")
    rtsp = rtspmodule.RTSPModule()

    if not rtsp.is_gpu_available():
        print("WARNING: GPU not available, falling back to CPU buffer mode")

    rtsp.start(rtsp_config_path)
    time.sleep(2)  # Warm up
    
    num_streams = rtsp.stream_count()
    print(f"    Connected to {num_streams} streams")
    
    # Check buffer mode
    cpu_buffer_mode = rtsp.is_cpu_buffer_enabled()
    print(f"    CPU Buffer Mode: {cpu_buffer_mode}")
    
    # Initialize TensorRT detector
    print("\n[2] Loading TensorRT model...")
    model_config = ModelConfig(
        config['model']['engine_path'],
        config['class_names'],
        config['model']['conf_threshold'],
        config['model']['nms_threshold'],
        config['model']['input_width'],
        config['model']['input_height']
    )
    
    detector = DetectorService()
    if not detector.load_model(model_config):
        print("Failed to load model!")
        rtsp.stop()
        return
    
    max_batch = detector.get_max_batch_size()
    print(f"    Model loaded. Max batch size: {max_batch}")
    
    # Determine which detection path to use
    use_zero_copy = not cpu_buffer_mode and rtsp.is_gpu_available()
    print(f"\n[3] Detection mode: {'ZERO-COPY GPU' if use_zero_copy else 'CPU BUFFER'}")
    print("\nPress 'q' to quit")
    print("=" * 60)

    # OPTIMIZATION: 3-stage async pipeline with queues
    # Stage 1: Capture thread → capture_queue
    # Stage 2: Inference thread (main) → inference_queue
    # Stage 3: Display thread (existing) reads from last_frame_data

    capture_queue = queue.Queue(maxsize=2)  # Small queue to prevent memory buildup
    inference_queue = queue.Queue(maxsize=2)

    # Timing stats
    inference_times = deque(maxlen=60)
    capture_times = deque(maxlen=60)
    last_detections = [[] for _ in range(num_streams)]
    frame_counter = 0
    fps_start_time = time.time()
    current_fps = 0.0

    # Per-stream latency tracking for RTSP production
    stream_latencies = [deque(maxlen=30) for _ in range(num_streams)]
    stream_frame_times = [time.time()] * num_streams

    # Display frame skip (from config)
    display_config = config.get('display', {})
    frame_skip = display_config.get('frame_skip', 1)
    display_frame_counter = 0
    print(f"    Display frame skip: {frame_skip} (show every {frame_skip} frame(s))")

    # Camera IDs to fetch (all streams, up to max batch size)
    camera_ids = list(range(min(num_streams, max_batch)))

    # Start display thread with dynamic grid for all streams
    display_thread = DisplayThread(num_streams)

    # Keep track of last valid frames + detections for display
    # Format: (frame, detections) tuples - display thread handles resize/draw
    last_frame_data = [None] * num_streams

    # Capture thread - runs independently
    capture_running = threading.Event()
    capture_running.set()

    def capture_worker():
        """Dedicated capture thread - always grabbing frames"""
        while capture_running.is_set():
            try:
                t_cap = time.time()

                if use_zero_copy:
                    # GPU zero-copy path
                    gpu_ptrs = []
                    widths = []
                    heights = []
                    valid_indices = []

                    for cam_id in camera_ids:
                        gpu_ptr, width, height, is_valid = rtsp.get_gpu_frame_ptr(cam_id)
                        if is_valid:
                            gpu_ptrs.append(gpu_ptr)
                            widths.append(width)
                            heights.append(height)
                            valid_indices.append(cam_id)

                    if gpu_ptrs:
                        cap_time = (time.time() - t_cap) * 1000
                        capture_queue.put(('gpu', gpu_ptrs, widths, heights, valid_indices, cap_time), timeout=0.1)
                else:
                    # CPU buffer path with dynamic batching
                    # Read batch sizes from config
                    batching_config = config.get('batching', {})
                    MIN_BATCH_SIZE = batching_config.get('min_batch_size', 4)
                    MAX_BATCH_SIZE = batching_config.get('max_batch_size', 12)
                    BATCH_TIMEOUT_MS = batching_config.get('timeout_ms', 2)

                    batch_result = rtsp.get_batch(camera_ids, timeout_ms=BATCH_TIMEOUT_MS)
                    valid_count = batch_result['valid_count']

                    if valid_count == 0:
                        time.sleep(0.001)
                        continue

                    # Try to batch up more frames if we have too few
                    # Skip extra batching if min_batch_size=1 (low-latency mode)
                    if valid_count < MIN_BATCH_SIZE and MIN_BATCH_SIZE > 1:
                        extra_result = rtsp.get_batch(camera_ids, timeout_ms=max(1, BATCH_TIMEOUT_MS // 2))
                        if extra_result['valid_count'] > 0:
                            for i in range(len(camera_ids)):
                                if extra_result['valid_mask'][i] and not batch_result['valid_mask'][i]:
                                    batch_result['data'][i] = extra_result['data'][i]
                                    batch_result['valid_mask'][i] = True
                                    valid_count += 1
                                    if valid_count >= MAX_BATCH_SIZE:
                                        break

                    if valid_count < MIN_BATCH_SIZE:
                        time.sleep(0.001)
                        continue

                    cap_time = (time.time() - t_cap) * 1000
                    capture_queue.put(('cpu', batch_result, cap_time), timeout=0.1)

            except queue.Full:
                # Queue is full, skip this frame (inference is bottleneck)
                continue
            except Exception as e:
                if capture_running.is_set():
                    print(f"[CAPTURE] Error: {e}")
                time.sleep(0.01)

    capture_thread = threading.Thread(target=capture_worker, daemon=True)
    capture_thread.start()
    print(f"    Started async capture pipeline")

    try:
        while display_thread.running:
            # === INFERENCE STAGE - Consume from capture queue ===
            try:
                capture_data = capture_queue.get(timeout=0.01)
            except queue.Empty:
                continue

            capture_mode = capture_data[0]

            if capture_mode == 'gpu':
                # GPU zero-copy path
                _, gpu_ptrs, widths, heights, valid_indices, cap_time = capture_data
                capture_times.append(cap_time)

                # ZERO-COPY INFERENCE
                t0 = time.time()
                batch_results = detector.detect_batch_gpu_nv12(gpu_ptrs, widths, heights)
                inference_times.append((time.time() - t0) * 1000)

                # Store results (no display frames in zero-copy mode)
                for idx, detections in zip(valid_indices, batch_results):
                    last_detections[idx] = detections
                    # Track per-stream latency
                    now = time.time()
                    stream_latency = (now - stream_frame_times[idx]) * 1000
                    stream_latencies[idx].append(stream_latency)
                    stream_frame_times[idx] = now
                    # Create placeholder for display
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, f"Cam {idx} (GPU Zero-Copy)", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    last_frame_data[idx] = (placeholder, detections)

                frame_counter += len(gpu_ptrs)

            else:  # capture_mode == 'cpu'
                # CPU buffer path
                _, batch_result, cap_time = capture_data
                capture_times.append(cap_time)

                batch_data = batch_result['data']
                valid_mask = batch_result['valid_mask']
                frame_width = batch_result['width']
                frame_height = batch_result['height']
                is_nv12 = batch_result['format'] == 'NV12'

                # Collect valid frames
                nv12_frames = []
                bgr_frames = []
                valid_indices = []

                for i, is_valid in enumerate(valid_mask):
                    if is_valid:
                        frame_data = batch_data[i]
                        # Check actual frame format by shape
                        if len(frame_data.shape) == 3 and frame_data.shape[2] == 3:
                            # Already BGR - use old path
                            bgr_frames.append(frame_data)
                            valid_indices.append(camera_ids[i])
                        elif is_nv12 or len(frame_data.shape) == 2:
                            # NV12 format - use new direct path
                            nv12_frames.append(frame_data)
                            valid_indices.append(camera_ids[i])

                if not nv12_frames and not bgr_frames:
                    continue

                # BATCHED INFERENCE
                t0 = time.time()

                if nv12_frames:
                    # Direct NV12 to GPU - skip CPU color conversion!
                    batch_results = detector.detect_batch_nv12(nv12_frames, frame_width, frame_height)

                    # Convert NV12 to BGR for display (off the critical path)
                    display_bgr_frames = []
                    for nv12_frame in nv12_frames:
                        bgr = cv2.cvtColor(nv12_frame, cv2.COLOR_YUV2BGR_NV12)
                        display_bgr_frames.append(bgr)
                else:
                    # Fallback to BGR path
                    batch_results = detector.detect_batch(bgr_frames)
                    display_bgr_frames = bgr_frames

                inference_times.append((time.time() - t0) * 1000)

                # Store results
                for i, (idx, detections) in enumerate(zip(valid_indices, batch_results)):
                    last_detections[idx] = detections
                    # Track per-stream latency
                    now = time.time()
                    stream_latency = (now - stream_frame_times[idx]) * 1000
                    stream_latencies[idx].append(stream_latency)
                    stream_frame_times[idx] = now
                    last_frame_data[idx] = (display_bgr_frames[i], detections)

                frame_counter += len(valid_indices)

            # === STATS ===
            elapsed_total = time.time() - fps_start_time
            avg_inference = sum(inference_times) / len(inference_times) if inference_times else 0
            avg_capture = sum(capture_times) / len(capture_times) if capture_times else 0
            per_stream_fps = current_fps / len(camera_ids) if camera_ids else 0

            # Calculate per-stream latency stats
            stream_latency_avgs = []
            for latencies in stream_latencies:
                if latencies:
                    stream_latency_avgs.append(sum(latencies) / len(latencies))
            avg_stream_latency = sum(stream_latency_avgs) / len(stream_latency_avgs) if stream_latency_avgs else 0
            max_stream_latency = max(stream_latency_avgs) if stream_latency_avgs else 0
            min_stream_latency = min(stream_latency_avgs) if stream_latency_avgs else 0

            if elapsed_total >= 1.0:
                current_fps = frame_counter / elapsed_total
                frame_counter = 0
                fps_start_time = time.time()
                per_stream_fps = current_fps / len(camera_ids) if camera_ids else 0

                # OPTIMIZATION: Enhanced performance stats with per-stream latency
                mode = 'GPU Zero-Copy' if use_zero_copy else 'NV12 Direct'
                total_latency = avg_capture + avg_inference
                gpu_util_estimate = (avg_inference / 16.6) * 100  # % of 60Hz frame time
                print(f"\r[PERF] FPS: {current_fps:.1f} total | {per_stream_fps:.1f}/stream | "
                      f"Inf: {avg_inference:.1f}ms | Cap: {avg_capture:.1f}ms | "
                      f"Latency: {total_latency:.1f}ms | Stream Lat: {avg_stream_latency:.1f}ms "
                      f"(min:{min_stream_latency:.1f} max:{max_stream_latency:.1f}) | "
                      f"GPU~{gpu_util_estimate:.0f}% | Mode: {mode}   ", end='', flush=True)

            # === DISPLAY (with frame skip) ===
            display_frame_counter += 1
            if display_frame_counter >= frame_skip:
                display_frame_counter = 0
                # Send frame data to display thread (handles resize + drawing)
                stats = (f"FPS: {current_fps:.1f} total | {per_stream_fps:.1f}/stream | "
                        f"Inf: {avg_inference:.1f}ms | Cap: {avg_capture:.1f}ms | "
                        f"Mode: {'GPU' if use_zero_copy else 'NV12'}")
                display_thread.show(last_frame_data, stats)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Stop capture thread
        capture_running.clear()
        if capture_thread.is_alive():
            capture_thread.join(timeout=1.0)

        display_thread.stop()
        rtsp.stop()
        print("Done.")


if __name__ == "__main__":
    main()

