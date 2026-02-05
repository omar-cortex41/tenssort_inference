"""
Minimal RTSP Client - No CuPy/PyTorch

This client tests raw GPU frame reception without any Python GPU libraries.
Used to measure baseline VRAM usage of GStreamer alone.
"""
import time
import threading
from collections import defaultdict
from queue import Queue, Empty
import csv
import datetime
import os
import subprocess

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
    def __init__(self):
        self.gpu_handle = None
        self.use_nvidia_smi = False
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
        cpu_pct = 0.0
        ram_gb = 0.0
        if psutil:
            cpu_pct = psutil.cpu_percent(interval=None)
            ram_gb = psutil.Process().memory_info().rss / (1024**3)

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
                
        return {"cpu": cpu_pct, "ram": ram_gb, "gpu": gpu_pct, "vram": vram_mb}


class StreamReceiver(threading.Thread):
    """Worker thread for receiving frames from a single RTSP stream."""
    
    def __init__(self, provider, camera_id, output_queue, use_cpu_buffer=False):
        super().__init__(daemon=True)
        self.provider = provider
        self.camera_id = camera_id
        self.output_queue = output_queue
        self.use_cpu_buffer = use_cpu_buffer
        self.running = True
        self.last_frame_id = -1
        self.frame_count = 0
    
    def stop(self):
        self.running = False
        
    def run(self):
        """Main receiving loop - no GPU library wrapping."""
        while self.running and self.provider.is_running():
            # Check for runtime fallback (mode change)
            current_mode_cpu = self.provider.is_cpu_buffer_enabled()
            if current_mode_cpu != self.use_cpu_buffer:
                if current_mode_cpu:
                    print(f"[WARN] Receiver {self.camera_id}: Runtime fallback detected - switching to CPU buffer mode")
                self.use_cpu_buffer = current_mode_cpu

            try:
                frame_data = {}
                current_id = -1
                
                if self.use_cpu_buffer:
                    # CPU Mode: Get frame from CPU ring buffer
                    cpu_frame = self.provider.get_cpu_frame(self.camera_id, 10)
                    if not cpu_frame.get("valid"):
                        continue
                        
                    current_id = cpu_frame.get("frame_id", -1)
                    if current_id == self.last_frame_id:
                        time.sleep(0.001) # Avoid spin loop on same frame
                        continue
                        
                    frame_data = {
                        "ptr": 0, # No GPU pointer
                        "camera_id": self.camera_id,
                        "frame_id": current_id,
                        "width": cpu_frame["width"],
                        "height": cpu_frame["height"],
                        "shape": cpu_frame["data"].shape,
                        "size": cpu_frame["data"].nbytes,
                        "format": cpu_frame.get("format", "unknown")
                    }
                    
                else:
                    # GPU Mode: Get frame from GPU queue
                    # GIL is RELEASED during this call
                    cuda_info = self.provider.get_cuda_frame(self.camera_id, 10)
                    
                    if not cuda_info.get("valid"):
                        continue
                    
                    current_id = cuda_info["frame_id"]
                    if current_id == self.last_frame_id:
                        time.sleep(0.001)
                        continue
                    
                    frame_data = {
                        "ptr": cuda_info["ptr"],  # Raw GPU pointer
                        "camera_id": self.camera_id,
                        "frame_id": current_id,
                        "width": cuda_info["width"],
                        "height": cuda_info["height"],
                        "shape": cuda_info["shape"],
                        "size": cuda_info["size"],
                        "format": "CUDA"
                    }
                
                self.last_frame_id = current_id
                self.frame_count += 1
                
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


class ConcurrentFrameReceiver:
    """Concurrent frame receiver - minimal version without GPU libraries."""
    
    def __init__(self, config_file="configs/config.yaml"):
        self.provider = rtspmodule.RTSPModule()
        self.config_file = config_file
        self.workers = []
        self.output_queues = {}
        self.start_time = None
        self.sys_monitor = SystemMonitor()
        
    def start(self):
        self.provider.start(self.config_file)
        time.sleep(2.0)
        
        num_streams = self.provider.stream_count()
        use_cpu_buffer = self.provider.is_cpu_buffer_enabled()
        mode_str = "CPU Ring Buffer" if use_cpu_buffer else "GPU Queue"
        
        print(f"[INFO] Started {num_streams} streams (no CuPy/PyTorch)")
        print(f"[INFO] Initial Buffer Mode: {mode_str}")
        
        self.output_queues = {i: Queue(maxsize=2) for i in range(num_streams)}
        
        for cam_id in range(num_streams):
            worker = StreamReceiver(self.provider, cam_id, self.output_queues[cam_id], use_cpu_buffer)
            self.workers.append(worker)
            worker.start()
        
        print(f"[INFO] Spawned {len(self.workers)} receiver threads")
        self.start_time = time.time()
        
    def stop(self):
        print("[INFO] Stopping receiver threads...")
        for worker in self.workers:
            worker.stop()
        for worker in self.workers:
            worker.join(timeout=1.0)
        self.provider.stop()
        
    def get_frame(self, camera_id):
        try:
            return self.output_queues[camera_id].get_nowait()
        except Empty:
            return None
    
    def get_all_frames(self):
        frames = []
        for cam_id in range(self.provider.stream_count()):
            frame = self.get_frame(cam_id)
            if frame:
                frames.append(frame)
        return frames
    
    def get_stats(self):
        stats = {}
        for worker in self.workers:
            cpp_stats = self.provider.get_stats(worker.camera_id)
            stats[worker.camera_id] = {
                "frame_count": worker.frame_count,
                "fps": cpp_stats.get("current_fps", 0.0),
            }
        return stats
    
    def run(self, duration_sec=10):
        self.start()
        
        print(f"\n[INFO] Receiving frames for {duration_sec} seconds...")
        print("-" * 90)
        
        start = time.time()
        last_print = start
        
        try:
            while time.time() - start < duration_sec:
                _ = self.get_all_frames()
                
                now = time.time()
                elapsed_total = now - start
                
                if now - last_print >= 1.0:
                    fps_parts = []
                    stats = self.get_stats()
                    
                    for cam_id in range(self.provider.stream_count()):
                        cam_stats = stats.get(cam_id, {"fps": 0.0})
                        fps_parts.append(f"Cam{cam_id}: {cam_stats['fps']:5.1f}")
                    
                    fps_str = " | ".join(fps_parts)
                    total_frames = sum(s["frame_count"] for s in stats.values())
                    total_fps = sum(s["fps"] for s in stats.values())
                    
                    sys_stats = self.sys_monitor.get_stats()
                    sys_msg = f"[CPU:{sys_stats['cpu']:3.0f}% MEM:{sys_stats['ram']:4.1f}G GPU:{sys_stats['gpu']:3.0f}% VRAM:{sys_stats['vram']:4.0f}M]"
                    
                    print(f"[{int(elapsed_total):4d}s] {sys_msg} {fps_str} | Total FPS: {total_fps:.1f} | Frames: {total_frames}")
                    last_print = now
                
                time.sleep(0.001)

        except KeyboardInterrupt:
            print("\n[INFO] Stopping.")
        finally:
            end_time = time.time()
            final_stats = self.get_stats()
            self.stop()
        
        print("-" * 90)
        
        duration = end_time - start
        return {
            "fps_per_camera": {k: v["frame_count"] / duration for k, v in final_stats.items()},
            "total_frames": sum(s["frame_count"] for s in final_stats.values()),
            "duration": duration,
            "stats": final_stats,
            "sys_stats": self.sys_monitor.get_stats(),
        }


def main():
    print("[INFO] Minimal RTSP Client (No CuPy/PyTorch)")
    print("[INFO] Testing baseline VRAM usage of GStreamer alone")
    
    receiver = ConcurrentFrameReceiver("configs/config.yaml")
    results = receiver.run(duration_sec=3600)
    
    print(f"\n{'Camera':<10} {'Avg FPS':<12} {'Total Frames':<15}")
    print("-" * 40)
    
    fps_data = results["fps_per_camera"]
    stats = results["stats"]
    
    for cam_id in sorted(fps_data.keys()):
        fps = fps_data[cam_id]
        count = stats[cam_id]["frame_count"]
        print(f"Cam {cam_id:<6} {fps:<12.2f} {count:<15}")
    
    print("-" * 40)
    print(f"Total Duration: {results['duration']:.1f}s")
    print(f"Total Frames: {results['total_frames']}")


if __name__ == "__main__":
    main()
