"""
Load Test RTSP Client - Simulates AI Workload Latency

This script mimics a slow consumer (like a heavy AI inference loop) to test
the RTSP module's buffer switching and overwrite/drop behavior.

Behaviors:
- Starts all streams defined in config.
- Does NOT consume frames at full speed.
- Instead, consumes one frame every `simulate_processing_time` seconds.
- Prints detailed statistics to show buffer filling and frame dropping.
"""
import time
import threading
from collections import defaultdict
from queue import Queue, Empty
import csv
import datetime
import os
import subprocess

# Latency settings
SIMULATE_AI_Latency = True     # If True, introduces delay
PROCESSING_TIME_SEC = 2.0      # Time to sleep between frame requests (simulated inference time)
                               # Set to a large value (e.g. 100.0) to simulate effectively NO consumption.

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
                cpu_count = psutil.cpu_count() or 1
                cpu_pct = raw_cpu / cpu_count
                ram_mb = self.process.memory_info().rss / (1024**2)
            except:
                pass

        # GPU & VRAM (system-wide)
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


class LazyStreamReceiver(threading.Thread):
    """Worker thread that can toggle between slow (AI sim) and fast (burst) consumption."""
    
    def __init__(self, provider, camera_id, use_cpu_buffer=False):
        super().__init__(daemon=True)
        self.provider = provider
        self.camera_id = camera_id
        self.use_cpu_buffer = use_cpu_buffer
        self.running = True
        self.frame_count = 0
        self.simulate_latency = True  # Default to slow mode
    
    def set_latency_mode(self, enabled):
        self.simulate_latency = enabled
    
    def stop(self):
        self.running = False
        
    def run(self):
        """Main loop."""
        while self.running and self.provider.is_running():
            try:
                # Fetch frame (short timeout to allow checking running/latency flags)
                valid = False
                frame_data = None
                
                if self.use_cpu_buffer:
                    frame = self.provider.get_cpu_frame(self.camera_id, timeout_ms=10)
                    valid = frame.get("valid", False)
                else:
                    info = self.provider.get_cuda_frame(self.camera_id, timeout_ms=10)
                    valid = info.get("valid", False)
                
                if valid:
                    self.frame_count += 1
                    # If in latency mode, simulate heavy processing
                    # In burst mode, simply loop immediately (consume as fast as possible)
                    if self.simulate_latency:
                        time.sleep(PROCESSING_TIME_SEC)
                else:
                    # If no frame, sleep briefly
                    time.sleep(0.001)

            except Exception as e:
                print(f"[ERROR] Receiver {self.camera_id}: {e}")
                time.sleep(1.0)


class LoadTestReceiver:
    """Manager for load testing."""
    
    def __init__(self, config_file="configs/config.yaml"):
        self.provider = rtspmodule.RTSPModule()
        self.config_file = config_file
        self.workers = []
        self.start_time = None
        self.sys_monitor = SystemMonitor()
        self.use_cpu_buffer = False
        self.buffer_mode = "unknown"
        
    def start(self):
        print(f"[INFO] starting RTSPModule with {self.config_file}")
        self.provider.start(self.config_file)
        time.sleep(5.0) # Warmup
        
        self.use_cpu_buffer = self.provider.is_cpu_buffer_enabled()
        self.buffer_mode = "CPU Ring Buffer" if self.use_cpu_buffer else "GPU Queue (CUDA)"
        
        num_streams = self.provider.stream_count()
        print(f"[INFO] Buffer Mode: {self.buffer_mode}")
        print(f"[INFO] Started {num_streams} streams.")
        print(f"[INFO] AI Workload Latency: {PROCESSING_TIME_SEC}s per frame")
        
        # Create workers (start in latency mode)
        for cam_id in range(num_streams):
            worker = LazyStreamReceiver(
                self.provider, cam_id, 
                use_cpu_buffer=self.use_cpu_buffer
            )
            # Default is already latency=True
            self.workers.append(worker)
            worker.start()
        
        print(f"[INFO] Spawned {len(self.workers)} receiver threads.")
        self.start_time = time.time()
        
    def stop(self):
        print("[INFO] Stopping threads...")
        for worker in self.workers:
            worker.stop()
        for worker in self.workers:
            worker.join(timeout=1.0)
        self.provider.stop()
    
    def get_stats(self):
        stats = {}
        for worker in self.workers:
            cpp_stats = self.provider.get_stats(worker.camera_id)
            stats[worker.camera_id] = {
                "consumed_by_worker": worker.frame_count,
                "cpp_consumed": cpp_stats.get("frames_consumed", 0),
                "fps": cpp_stats.get("current_fps", 0.0),
                "overwritten": cpp_stats.get("frames_overwritten", 0),
                "dropped_queue": cpp_stats.get("frames_dropped_queue", 0),
                "queue_depth": cpp_stats.get("queue_depth", 0),
                "received": cpp_stats.get("frames_received", 0),
                "decoded": cpp_stats.get("frames_decoded", 0),
            }
        return stats
    
    def print_current_stats(self, header_msg="STATS"):
        stats = self.get_stats()
        sys_stats = self.sys_monitor.get_stats()
        elapsed = time.time() - self.start_time
        
        print("\n" + "="*110)
        sys_info = f"Sys: CPU={sys_stats['cpu']:.1f}% RAM={sys_stats['ram_mb']:.0f}M GPU={sys_stats['gpu']:.1f}% VRAM={sys_stats['vram_mb']:.0f}M"
        print(f"[{header_msg}] @ {elapsed:.1f}s | {sys_info}")
        print(f"{'Cam':<5} {'FPS':<6} {'Recv':<8} {'Dec':<8} {'Depth':<6} {'PyCons':<8} {'CppCons':<8} {'Overwr':<8} {'QDrop':<8}")
        print("-" * 105)
        
        for cam_id in sorted(stats.keys()):
            s = stats[cam_id]
            print(f"{cam_id:<5} {s['fps']:<6.1f} {s['received']:<8} {s['decoded']:<8} {s['queue_depth']:<6} "
                  f"{s['consumed_by_worker']:<8} {s['cpp_consumed']:<8} {s['overwritten']:<8} {s['dropped_queue']:<8}")
        print("="*110)

    def set_all_latency(self, enabled):
        for worker in self.workers:
            worker.set_latency_mode(enabled)

    def run(self, duration_sec=60):
        self.start()
        
        # Cycle configuration
        LATENCY_PHASE_SEC = 20
        BURST_PHASE_SEC = 20
        
        start_time = time.time()
        phase_start = start_time
        last_print = start_time
        in_latency_mode = True
        
        print(f"\n[INFO] Starting cycle: {LATENCY_PHASE_SEC}s Latency <-> {BURST_PHASE_SEC}s Burst")
        print("-" * 90)
        
        try:
            while time.time() - start_time < duration_sec:
                now = time.time()
                phase_elapsed = now - phase_start
                
                # Periodic stats printing (every 1.0s)
                if now - last_print >= 1.0:
                    mode_str = "LATENCY" if in_latency_mode else "BURST"
                    self.print_current_stats(f"STATS - {mode_str} MODE")
                    last_print = now
                
                # Check for phase switch
                if in_latency_mode and phase_elapsed >= LATENCY_PHASE_SEC:
                    # Switch to BURST
                    print(f"\n[INFO] END OF LATENCY PHASE - SWITCHING TO BURST (No delay) for {BURST_PHASE_SEC}s...")
                    self.set_all_latency(False)
                    in_latency_mode = False
                    phase_start = now
                
                elif not in_latency_mode and phase_elapsed >= BURST_PHASE_SEC:
                    # Switch to LATENCY
                    print(f"\n[INFO] END OF BURST PHASE - SWITCHING TO LATENCY ({PROCESSING_TIME_SEC}s delay) for {LATENCY_PHASE_SEC}s...")
                    self.set_all_latency(True)
                    in_latency_mode = True
                    phase_start = now
                
                # Sleep briefly
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted.")
        finally:
            self.stop()
            print("[INFO] Use Client Load Test Finished")


def main():
    receiver = LoadTestReceiver("configs/config.yaml")
    receiver.run(duration_sec=3600)  # Run for an hour or until Ctrl+C

if __name__ == "__main__":
    main()
