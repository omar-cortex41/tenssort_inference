"""
Concurrent RTSP Client - Dual Buffer Mode Support

Automatically detects buffer mode from config and uses appropriate frame retrieval:
- GPU Mode (cpu_buffer_enabled=false): Uses get_cuda_frame() for zero-copy GPU access
- CPU Mode (cpu_buffer_enabled=true): Uses get_cpu_frame() to get oldest available frame

Note: CPU mode is auto-enabled if GPU hardware (nvdec/cudaconvert) is unavailable.

Each stream is received in its own thread with GIL-released C++ bindings.
"""
import time
import threading
from collections import defaultdict
from queue import Queue, Empty
import csv
import datetime
import os
import numpy as np
import subprocess
import sys

# CuPy is optional for CPU buffer mode
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

try:
    import psutil
except ImportError:
    psutil = None
try:
    import pynvml
except ImportError:
    pynvml = None
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
import rtspmodule


class SystemMonitor:
    """Monitor system resources (CPU, RAM, GPU, VRAM) for this process."""
    
    def __init__(self, pid=None):
        self.gpu_handle = None
        self.use_nvidia_smi = False
        self.pid = pid or os.getpid()
        self.process = None
        if psutil:
            self.process = psutil.Process(self.pid)
        self._init_gpu()

    def _init_gpu(self):
        if pynvml:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self.use_nvidia_smi = True
        else:
            self.use_nvidia_smi = True

    def get_stats(self):
        """Get current resource usage for THIS PROCESS."""
        # CPU & RAM for this process
        cpu_pct = 0.0
        ram_mb = 0.0
        if self.process:
            try:
                raw_cpu = self.process.cpu_percent(interval=None)
                # Normalize to 100% max (divide by core count)
                cpu_count = psutil.cpu_count() or 1
                cpu_pct = raw_cpu / cpu_count
                ram_mb = self.process.memory_info().rss / (1024**2)
            except:
                pass

        # GPU & VRAM (system-wide, as per-process GPU stats require CUDA context)
        gpu_pct = 0.0
        vram_mb = 0.0
        
        if self.gpu_handle:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_pct = util.gpu
                vram_mb = mem.used / (1024**2)
            except:
                pass
        elif self.use_nvidia_smi:
            try:
                cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits"
                out = subprocess.check_output(cmd.split(), encoding='utf-8').strip()
                p_gpu, p_mem = out.split(',')
                gpu_pct = float(p_gpu)
                vram_mb = float(p_mem)
            except:
                pass
                
        return {
            "cpu": cpu_pct,
            "ram_mb": ram_mb,
            "gpu": gpu_pct,
            "vram_mb": vram_mb
        }


class StreamReceiver(threading.Thread):
    """Worker thread for receiving frames from a single RTSP stream."""
    
    def __init__(self, provider, camera_id, output_queue, use_cpu_buffer=False):
        super().__init__(daemon=True)
        self.provider = provider
        self.camera_id = camera_id
        self.output_queue = output_queue
        self.use_cpu_buffer = use_cpu_buffer
        self.running = True
        
        # Frame tracking
        self.last_frame_id = -1
        self.frame_count = 0
    
    def stop(self):
        self.running = False
        
    def run(self):
        """Main receiving loop - runs in parallel with other receivers."""
        while self.running and self.provider.is_running():
            try:
                if self.use_cpu_buffer:
                    # CPU Buffer Mode: FIFO - get oldest unread frame
                    frame = self.provider.get_cpu_frame(self.camera_id, timeout_ms=10)
                    
                    if not frame.get("valid"):
                        continue
                    
                    current_id = frame["frame_id"]
                    self.last_frame_id = current_id
                    self.frame_count += 1
                    
                    # Store CPU frame data
                    frame_data = {
                        "data": frame["data"],  # numpy array
                        "camera_id": self.camera_id,
                        "frame_id": current_id,
                        "width": frame["width"],
                        "height": frame["height"],
                        "format": frame["format"],
                        "timestamp_ns": frame.get("timestamp_ns", 0),
                        "source": "cpu_buffer",
                    }
                else:
                    # GPU Mode: Get CUDA frame
                    cuda_info = self.provider.get_cuda_frame(self.camera_id, 10)
                    
                    if not cuda_info.get("valid"):
                        continue
                    
                    current_id = cuda_info["frame_id"]
                    if current_id == self.last_frame_id:
                        time.sleep(0.001)
                        continue
                    
                    self.last_frame_id = current_id
                    self.frame_count += 1
                    
                    # Store raw GPU pointer info
                    frame_data = {
                        "ptr": cuda_info['ptr'],
                        "size": cuda_info['size'],
                        "camera_id": self.camera_id,
                        "frame_id": current_id,
                        "width": cuda_info["width"],
                        "height": cuda_info["height"],
                        "shape": cuda_info["shape"],
                        "format": cuda_info.get("format", "NV12"),
                        "source": "gpu_queue",
                    }
                
                # Put frame in queue (drop old if full)
                try:
                    if self.output_queue.full():
                        try:
                            self.output_queue.get_nowait()
                        except Empty:
                            pass
                    self.output_queue.put_nowait(frame_data)
                except:
                    pass
                    
            except Exception as e:
                print(f"[ERROR] Receiver {self.camera_id}: {e}")
                time.sleep(0.01)


def get_gpu_array(frame_data):
    """Create a CuPy array on-demand from raw GPU frame data."""
    if not HAS_CUPY:
        raise RuntimeError("CuPy not available for GPU array access")
    mem = cp.cuda.UnownedMemory(frame_data['ptr'], frame_data['size'], None)
    memptr = cp.cuda.MemoryPointer(mem, 0)
    return cp.ndarray(frame_data['shape'], dtype=cp.uint8, memptr=memptr)


def get_cpu_from_gpu(frame_data):
    """Copy GPU frame to CPU memory as a NumPy array."""
    if not HAS_CUPY:
        raise RuntimeError("CuPy not available for GPU to CPU transfer")
    gpu_array = get_gpu_array(frame_data)
    return cp.asnumpy(gpu_array)


class ConcurrentFrameReceiver:
    """Concurrent frame receiver supporting both GPU and CPU buffer modes."""
    
    def __init__(self, config_file="configs/config.yaml"):
        self.provider = rtspmodule.RTSPModule()
        self.config_file = config_file
        self.workers = []
        self.output_queues = {}
        self.start_time = None
        self.sys_monitor = SystemMonitor()
        self.use_cpu_buffer = False  # Will be set after start
        self.gpu_available = True    # Will be set after start
        self.buffer_mode = "unknown"
        
    def start(self):
        self.provider.start(self.config_file)
        time.sleep(0.1)
        
        # Detect buffer mode and GPU availability
        self.use_cpu_buffer = self.provider.is_cpu_buffer_enabled()
        self.gpu_available = self.provider.is_gpu_available()
        
        if self.gpu_available:
            self.buffer_mode = "GPU Queue (CUDA)" if not self.use_cpu_buffer else "CPU Ring Buffer (config)"
        else:
            self.buffer_mode = "CPU Ring Buffer (GPU fallback)"
        
        num_streams = self.provider.stream_count()
        print(f"[INFO] GPU Available: {self.gpu_available}")
        print(f"[INFO] Buffer Mode: {self.buffer_mode}")
        print(f"[INFO] Started {num_streams} streams with CONCURRENT reception.")
        
        # Create output queues and worker threads
        self.output_queues = {i: Queue(maxsize=2) for i in range(num_streams)}
        
        for cam_id in range(num_streams):
            worker = StreamReceiver(
                self.provider, cam_id, 
                self.output_queues[cam_id],
                use_cpu_buffer=self.use_cpu_buffer
            )
            self.workers.append(worker)
            worker.start()
        
        print(f"[INFO] Spawned {len(self.workers)} concurrent receiver threads.")
        self.start_time = time.time()
        
    def stop(self):
        print("[INFO] Stopping receiver threads...")
        for worker in self.workers:
            worker.stop()
        for worker in self.workers:
            worker.join(timeout=1.0)
        self.provider.stop()
        
    def get_frame(self, camera_id):
        """Get latest frame from a specific camera (non-blocking)."""
        try:
            return self.output_queues[camera_id].get_nowait()
        except Empty:
            return None
    
    def get_all_frames(self):
        """Get latest frames from all cameras (non-blocking)."""
        frames = []
        for cam_id in range(self.provider.stream_count()):
            frame = self.get_frame(cam_id)
            if frame:
                frames.append(frame)
        return frames
    
    def get_stats(self):
        """Get current stats from C++ side (includes accurate FPS)."""
        stats = {}
        for worker in self.workers:
            cpp_stats = self.provider.get_stats(worker.camera_id)
            stats[worker.camera_id] = {
                "frame_count": worker.frame_count,
                "fps": cpp_stats.get("current_fps", 0.0),
                "instant_fps": cpp_stats.get("instant_fps", 0.0),
                "source_fps": cpp_stats.get("source_fps", 0.0),
            }
        return stats
    
    def run(self, duration_sec=10):
        self.start()
        
        print(f"\n[INFO] Receiving frames for {duration_sec} seconds...")
        print("-" * 90)
        
        start = time.time()
        last_print = start
        last_stats_print = start
        last_counts = defaultdict(int)
        STATS_PRINT_INTERVAL = 5.0
        
        try:
            while time.time() - start < duration_sec:
                # Process frames from all cameras
                frames = self.get_all_frames()
                for frame_data in frames:
                    if frame_data.get("source") == "cpu_buffer":
                        # CPU buffer mode - data is already numpy array
                        cpu_frame = frame_data["data"]
                    else:
                        # GPU mode - copy to CPU if needed
                        if HAS_CUPY:
                            cpu_frame = get_cpu_from_gpu(frame_data)
                        else:
                            cpu_frame = None
                    
                    # cpu_frame is now a numpy array, do processing here
                    if cpu_frame is not None:
                        del cpu_frame  # Explicit cleanup
                
                now = time.time()
                elapsed_total = now - start
                
                # Brief FPS update every second
                if now - last_print >= 1.0:
                    fps_parts = []
                    stats = self.get_stats()
                    
                    for cam_id in range(self.provider.stream_count()):
                        cam_stats = stats.get(cam_id, {"frame_count": 0, "fps": 0.0})
                        sliding_fps = cam_stats["fps"]
                        fps_parts.append(f"Cam{cam_id}: {sliding_fps:5.1f}")
                        last_counts[cam_id] = cam_stats["frame_count"]
                    
                    fps_str = " | ".join(fps_parts)
                    total_frames = sum(s["frame_count"] for s in stats.values())
                    total_fps = sum(s["fps"] for s in stats.values())
                    
                    # System Stats
                    sys_stats = self.sys_monitor.get_stats()
                    sys_msg = f"[CPU:{sys_stats['cpu']:3.0f}% RAM:{sys_stats['ram_mb']:5.0f}M GPU:{sys_stats['gpu']:3.0f}% VRAM:{sys_stats['vram_mb']:4.0f}M]"
                    
                    print(f"[{int(elapsed_total):4d}s] {sys_msg} {fps_str} | Total FPS: {total_fps:.1f} | Frames: {total_frames}")
                    last_print = now
                
                # Detailed queue stats every 5 seconds
                if now - last_stats_print >= STATS_PRINT_INTERVAL:
                    last_stats_print = now
                    num_streams = self.provider.stream_count()
                    sys_stats = self.sys_monitor.get_stats()
                    print("\n" + "="*120)
                    print(f"[STATS] @ {elapsed_total:.1f}s | Mode: {self.buffer_mode} | {len(self.workers)} threads | "
                          f"CPU: {sys_stats['cpu']:.1f}% RAM: {sys_stats['ram_mb']:.0f}MB GPU: {sys_stats['gpu']:.1f}% VRAM: {sys_stats['vram_mb']:.0f}MB")
                    print("-"*120)
                    print(f"{'Cam':<5} {'Recv':<8} {'Dec':<8} {'Dup':<6} {'Cons':<8} {'QDrop':<8} {'QDrop%':<8} {'QDepth':<8} {'FPS':<8}")
                    print("-"*120)
                    for cam_id in range(num_streams):
                        s = self.provider.get_stats(cam_id)
                        print(f"{cam_id:<5} {s['frames_received']:<8} {s['frames_decoded']:<8} "
                              f"{s.get('frames_duplicate', 0):<6} {s['frames_consumed']:<8} {s['frames_dropped_queue']:<8} "
                              f"{s['queue_drop_rate']:<8.1f} {s['queue_depth']:<8} {s.get('current_fps', 0):<8.1f}")
                    print("="*110 + "\n")
                
                time.sleep(0.001)

        except KeyboardInterrupt:
            print("\n[INFO] Stopping.")
        finally:
            end_time = time.time()
            final_stats = self.get_stats()
            final_sys_stats = self.sys_monitor.get_stats()
            
            # Print final queue statistics
            num_streams = self.provider.stream_count()
            print("\n" + "="*110)
            print("[INFO] Final Frame Queue Statistics:")
            print("="*110)
            for cam_id in range(num_streams):
                s = self.provider.get_stats(cam_id)
                print(f"Cam {cam_id}: received={s['frames_received']}, decoded={s['frames_decoded']}, "
                      f"duplicate={s.get('frames_duplicate', 0)}, consumed={s['frames_consumed']}, "
                      f"queue_dropped={s['frames_dropped_queue']} ({s['queue_drop_rate']:.1f}%)")
            print("="*110)
            
            self.stop()
        
        print("-" * 90)
        
        # Final Summary
        duration = end_time - start
        fps_per_camera = {}
        for cam_id, cam_stats in final_stats.items():
            # Use accurate C++ sliding window FPS instead of frame_count/duration
            fps_per_camera[cam_id] = cam_stats["fps"]
        
        return {
            "fps_per_camera": fps_per_camera,
            "total_frames": sum(s["frame_count"] for s in final_stats.values()),
            "duration": duration,
            "stats": final_stats,
            "sys_stats": final_sys_stats,
            "buffer_mode": self.buffer_mode,
        }


def log_to_csv(results, filename="benchmark_results.csv"):
    file_exists = os.path.isfile(filename)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    fps_data = results["fps_per_camera"]
    stream_count = len(fps_data)
    if stream_count == 0:
        return

    avg_fps = sum(fps_data.values()) / stream_count
    min_fps = min(fps_data.values())
    total_frames = results["total_frames"]
    duration = results["duration"]
    buffer_mode = results.get("buffer_mode", "unknown")

    # System Stats
    sys_stats = results.get("sys_stats", {"cpu": 0, "ram_mb": 0, "gpu": 0, "vram_mb": 0})
    
    sorted_cam_ids = sorted(fps_data.keys())
    per_cam_fps = [f"{fps_data[cam_id]:.2f}" for cam_id in sorted_cam_ids]
    
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["Timestamp", "Buffer_Mode", "Stream_Count", "Duration_Sec", "Total_Frames", "Avg_FPS_All", "Min_FPS"]
            header += ["CPU_Pct", "RAM_MB", "GPU_Pct", "VRAM_MB"]
            header += [f"Cam_{cam_id}_FPS" for cam_id in sorted_cam_ids]
            writer.writerow(header)
        
        row = [timestamp, buffer_mode, stream_count, f"{duration:.2f}", total_frames, f"{avg_fps:.2f}", f"{min_fps:.2f}"]
        row += [f"{sys_stats['cpu']:.1f}", f"{sys_stats['ram_mb']:.0f}", f"{sys_stats['gpu']:.1f}", f"{sys_stats['vram_mb']:.0f}"]
        row += per_cam_fps
        writer.writerow(row)
    
    print(f"\n[INFO] Results logged to {filename}")


def main():
    print("="*80)
    print("[INFO] Concurrent RTSP Client - Dual Buffer Mode (with GPU Fallback)")
    print("="*80)
    
    receiver = ConcurrentFrameReceiver("configs/config.yaml")
    results = receiver.run(duration_sec=3600)
    
    # Final Summary
    print(f"\n{'='*80}")
    print(f"[FINAL SUMMARY]")
    print(f"{'='*80}")
    print(f"Buffer Mode: {results['buffer_mode']}")
    print(f"GPU Available: {receiver.gpu_available}")
    print(f"Duration: {results['duration']:.1f}s")
    print(f"Total Frames Consumed: {results['total_frames']:,}")
    
    # Resource Usage
    sys_stats = results["sys_stats"]
    print(f"\n[Resource Usage (this process)]")
    print(f"  CPU: {sys_stats['cpu']:.1f}%")
    print(f"  RAM: {sys_stats['ram_mb']:.0f} MB")
    print(f"  GPU: {sys_stats['gpu']:.1f}%")
    print(f"  VRAM: {sys_stats['vram_mb']:.0f} MB")
    
    print(f"\n{'Camera':<10} {'Avg FPS':<12} {'Source FPS':<12} {'Total Frames':<15}")
    print("-" * 55)
    
    fps_data = results["fps_per_camera"]
    stats = results["stats"]
    
    for cam_id in sorted(fps_data.keys()):
        fps = fps_data[cam_id]
        source_fps = stats[cam_id].get("source_fps", 0.0)
        count = stats[cam_id]["frame_count"]
        print(f"Cam {cam_id:<6} {fps:<12.2f} {source_fps:<12.2f} {count:<15}")
    
    print("-" * 55)
    
    log_to_csv(results)


if __name__ == "__main__":
    main()
