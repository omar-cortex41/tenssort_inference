"""
Concurrent RTSP Viewer - Dual Buffer Mode Support

Automatically detects buffer mode from config and uses appropriate frame retrieval:
- GPU Mode (cpu_buffer_enabled=false): Uses get_cuda_frame() with GPU color conversion
- CPU Mode (cpu_buffer_enabled=true): Uses get_cpu_frame() for CPU-based frames

Note: CPU mode is auto-enabled if GPU hardware (nvdec/cudaconvert) is unavailable.

Each stream is processed in its own thread with GIL-released C++ bindings.
"""
import time
import math
import os
import subprocess
import threading
from queue import Queue, Empty
import cv2
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
import rtspmodule

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


# GPU kernel for NV12->BGR conversion (only used in GPU mode)
if HAS_CUPY:
    NV12_TO_BGR_RESIZE_KERNEL = cp.RawKernel(r'''
    extern "C" __global__
    void nv12_to_bgr_resize(const unsigned char* __restrict__ y_plane,
                            const unsigned char* __restrict__ uv_plane,
                            unsigned char* __restrict__ bgr_out,
                            int src_width, int src_height, int src_stride,
                            int dst_width, int dst_height,
                            float scale_x, float scale_y) {
        
        int dx = blockIdx.x * blockDim.x + threadIdx.x;
        int dy = blockIdx.y * blockDim.y + threadIdx.y;

        if (dx >= dst_width || dy >= dst_height) return;

        int sx = (int)(dx * scale_x);
        int sy = (int)(dy * scale_y);

        if (sx >= src_width) sx = src_width - 1;
        if (sy >= src_height) sy = src_height - 1;

        int y_idx = sy * src_stride + sx;
        int uv_idx = (sy / 2) * src_stride + (sx / 2) * 2;

        unsigned char Y = y_plane[y_idx];
        unsigned char U = uv_plane[uv_idx];
        unsigned char V = uv_plane[uv_idx + 1];

        int C = Y - 16;
        int D = U - 128;
        int E = V - 128;

        int R = (298 * C + 409 * E + 128) >> 8;
        int G = (298 * C - 100 * D - 208 * E + 128) >> 8;
        int B = (298 * C + 516 * D + 128) >> 8;

        int out_idx = (dy * dst_width + dx) * 3;
        bgr_out[out_idx]     = (unsigned char)((B < 0) ? 0 : ((B > 255) ? 255 : B));
        bgr_out[out_idx + 1] = (unsigned char)((G < 0) ? 0 : ((G > 255) ? 255 : G));
        bgr_out[out_idx + 2] = (unsigned char)((R < 0) ? 0 : ((R > 255) ? 255 : R));
    }
    ''', 'nv12_to_bgr_resize')
else:
    NV12_TO_BGR_RESIZE_KERNEL = None


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


class StreamWorker(threading.Thread):
    """Worker thread for processing a single RTSP stream concurrently."""
    
    def __init__(self, provider, camera_id, target_w, target_h, output_queue, use_cpu_buffer=False):
        super().__init__(daemon=True)
        self.provider = provider
        self.camera_id = camera_id
        self.target_w = target_w
        self.target_h = target_h
        self.output_queue = output_queue
        self.use_cpu_buffer = use_cpu_buffer
        self.running = True
        self.cuda_stream = None
        self.gpu_buffer = None
        self.pinned_buffer = None
        self.frame_count = 0
        
    def init_resources(self):
        """Initialize GPU resources (must be called from thread context)."""
        if not self.use_cpu_buffer and HAS_CUPY:
            self.cuda_stream = cp.cuda.Stream(non_blocking=True)
            self.gpu_buffer = cp.ndarray((self.target_h, self.target_w, 3), dtype=cp.uint8)
            self.pinned_buffer = np.empty((self.target_h, self.target_w, 3), dtype=np.uint8)
        else:
            self.pinned_buffer = np.empty((self.target_h, self.target_w, 3), dtype=np.uint8)
        
    def stop(self):
        self.running = False
        
    def run(self):
        """Main processing loop - runs in parallel with other workers."""
        self.init_resources()
        
        while self.running and self.provider.is_running():
            # Check for runtime fallback (mode change)
            current_mode_cpu = self.provider.is_cpu_buffer_enabled()
            if current_mode_cpu != self.use_cpu_buffer:
                if current_mode_cpu:
                    print(f"[WARN] Worker {self.camera_id}: Runtime fallback detected - switching to CPU buffer mode")
                self.use_cpu_buffer = current_mode_cpu

            try:
                if self.use_cpu_buffer:
                    # CPU Buffer Mode: FIFO - get oldest unread frame
                    frame = self.provider.get_cpu_frame(self.camera_id, timeout_ms=10)
                    
                    if not frame.get("valid"):
                        continue
                    
                    # Get frame data as numpy array
                    frame_data = frame["data"]
                    src_h = frame["height"]
                    src_w = frame["width"]
                    fmt = frame.get("format", "NV12")
                    
                    if src_w == 0 or src_h == 0:
                        continue
                    
                    # Convert to BGR for display
                    if fmt == "BGR":
                        bgr_frame = frame_data.reshape((src_h, src_w, 3))
                    elif fmt == "RGB":
                        rgb_frame = frame_data.reshape((src_h, src_w, 3))
                        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                    elif fmt == "NV12":
                        nv12_frame = frame_data.reshape((int(src_h * 1.5), src_w))
                        bgr_frame = cv2.cvtColor(nv12_frame, cv2.COLOR_YUV2BGR_NV12)
                    else:
                        # Assume raw RGB/BGR
                        bgr_frame = frame_data.reshape((src_h, src_w, 3))
                    
                    # Resize for display
                    display_frame = cv2.resize(bgr_frame, (self.target_w, self.target_h))
                    self.frame_count += 1
                    
                else:
                    # GPU Mode: Get CUDA frame and convert on GPU
                    if not HAS_CUPY:
                        time.sleep(0.1)
                        continue
                    
                    info = self.provider.get_cuda_frame(self.camera_id, 10)
                    
                    if not info.get('valid', False):
                        continue
                        
                    src_w, src_h = info['width'], info['height']
                    stride = info.get('stride', src_w)
                    
                    if src_w == 0 or src_h == 0:
                        continue
                    
                    actual_gpu_size = int(stride * src_h * 1.5)
                    
                    # GPU processing (GIL released by CuPy)
                    with self.cuda_stream:
                        mem = cp.cuda.UnownedMemory(info['ptr'], actual_gpu_size, None)
                        memptr = cp.cuda.MemoryPointer(mem, 0)
                        
                        threads = (32, 32)
                        blocks_x = (self.target_w + 31) // 32
                        blocks_y = (self.target_h + 31) // 32
                        scale_x = src_w / self.target_w
                        scale_y = src_h / self.target_h
                        
                        NV12_TO_BGR_RESIZE_KERNEL(
                            (blocks_x, blocks_y), threads,
                            (memptr, memptr + stride * src_h, self.gpu_buffer,
                             src_w, src_h, stride,
                             self.target_w, self.target_h,
                             cp.float32(scale_x), cp.float32(scale_y))
                        )
                        
                        self.gpu_buffer.get(out=self.pinned_buffer)
                    
                    self.cuda_stream.synchronize()
                    display_frame = self.pinned_buffer.copy()
                    self.frame_count += 1
                
                # Put frame in output queue
                try:
                    if self.output_queue.full():
                        try:
                            self.output_queue.get_nowait()
                        except Empty:
                            pass
                    self.output_queue.put_nowait({
                        'camera_id': self.camera_id,
                        'frame': display_frame,
                    })
                except:
                    pass
                    
            except Exception as e:
                print(f"[ERROR] Worker {self.camera_id}: {e}")
                time.sleep(0.01)


def main():
    print("="*80)
    print("[INFO] Concurrent RTSP Viewer - Dual Buffer Mode (with GPU Fallback)")
    print("="*80)
    
    provider = rtspmodule.RTSPModule()
    provider.start("configs/config.yaml")
    
    time.sleep(1)
    
    # Detect buffer mode and GPU availability
    use_cpu_buffer = provider.is_cpu_buffer_enabled()
    gpu_available = provider.is_gpu_available()
    
    if gpu_available:
        buffer_mode = "GPU Queue (CUDA)" if not use_cpu_buffer else "CPU Ring Buffer (config)"
    else:
        buffer_mode = "CPU Ring Buffer (GPU fallback)"
    
    num_streams = provider.stream_count()
    print(f"[INFO] GPU Available: {gpu_available}")
    print(f"[INFO] Buffer Mode: {buffer_mode}")
    print(f"[INFO] Started {num_streams} streams with CONCURRENT processing.")
    print("[INFO] Each stream runs in its own thread with GIL released.")
    print("[INFO] Press 'q' to quit.")
    
    # Initialize system monitor
    sys_monitor = SystemMonitor()
    
    TARGET_W = 640
    TARGET_H = 360
    
    cols = int(math.ceil(math.sqrt(num_streams)))
    rows = int(math.ceil(num_streams / cols))
    output_queues = {i: Queue(maxsize=2) for i in range(num_streams)}
    workers = []
    for cam_id in range(num_streams):
        worker = StreamWorker(provider, cam_id, TARGET_W, TARGET_H, 
                              output_queues[cam_id], use_cpu_buffer=use_cpu_buffer)
        workers.append(worker)
        worker.start()
    
    print(f"[INFO] Spawned {len(workers)} concurrent worker threads.")
    display_frames = {}
    stream_stats = {i: {'fps': 0.0} for i in range(num_streams)}
    
    global_frame_count = 0
    start_time = time.time()
    last_stats_print = time.time()
    STATS_PRINT_INTERVAL = 5.0
    
    try:
        while provider.is_running():
            for cam_id in range(num_streams):
                try:
                    data = output_queues[cam_id].get_nowait()
                    display_frames[cam_id] = data['frame']
                except Empty:
                    pass
            display_list = []
            for cam_id in range(num_streams):
                if cam_id in display_frames:
                    frame = display_frames[cam_id].copy()
                else:
                    frame = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
                    cv2.putText(frame, "Waiting...", (TARGET_W//2 - 50, TARGET_H//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                drop_stats = provider.get_stats(cam_id)
                fps_val = drop_stats.get('current_fps', 0.0)
                stream_stats[cam_id]['fps'] = fps_val 
                queue_drop_rate = drop_stats.get('queue_drop_rate', 0.0)
                queue_depth = drop_stats.get('queue_depth', 0)
                
                if queue_drop_rate == 0:
                    color = (0, 255, 0)
                elif queue_drop_rate < 5:
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)
                # Line 1: Camera ID and FPS (bold red)
                cv2.putText(frame, f"Cam {cam_id} | FPS: {fps_val:.1f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                consumed = drop_stats.get('frames_consumed', 0)
                cv2.putText(frame, f"Q:{queue_depth} | Cons:{consumed} | Drop:{queue_drop_rate:.1f}%", (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                display_list.append(frame)
            
            total_slots = rows * cols
            while len(display_list) < total_slots:
                display_list.append(np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8))
            
            grid_rows = []
            for r in range(rows):
                row_imgs = display_list[r * cols : (r + 1) * cols]
                grid_rows.append(np.hstack(row_imgs))
            
            final_grid = np.vstack(grid_rows)
            
            global_frame_count += 1
            elapsed = time.time() - start_time
            total_fps = sum(s['fps'] for s in stream_stats.values())
            
            # Show buffer mode in display
            mode_str = "CPU" if use_cpu_buffer else "GPU"
            cv2.putText(final_grid, f"Global FPS: {total_fps:.1f} | Mode: {mode_str}", 
                       (20, final_grid.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            final_grid = cv2.resize(final_grid, (1920, 1080))
            cv2.imshow("RTSP Concurrent Viewer", final_grid)
            
            # Check for mode change in main loop for UI update
            use_cpu_buffer = provider.is_cpu_buffer_enabled()
            if use_cpu_buffer and "GPU" in buffer_mode:
                 buffer_mode = "CPU Ring Buffer (Runtime Fallback)"
            
            # Print stats periodically
            now = time.time()
            if now - last_stats_print >= STATS_PRINT_INTERVAL:
                last_stats_print = now
                sys_stats = sys_monitor.get_stats()
                print("\n" + "="*120)
                print(f"[STATS] @ {elapsed:.1f}s | Mode: {buffer_mode} | {len(workers)} threads | "
                      f"CPU: {sys_stats['cpu']:.1f}% RAM: {sys_stats['ram_mb']:.0f}MB GPU: {sys_stats['gpu']:.1f}% VRAM: {sys_stats['vram_mb']:.0f}MB")
                print("-"*120)
                print(f"{'Cam':<5} {'Recv':<8} {'Dec':<8} {'Dup':<6} {'Cons':<8} {'QDrop':<8} {'QDrop%':<8} {'QDepth':<8} {'FPS':<8}")
                print("-"*120)
                for cam_id in range(num_streams):
                    s = provider.get_stats(cam_id)
                    print(f"{cam_id:<5} {s['frames_received']:<8} {s['frames_decoded']:<8} "
                          f"{s.get('frames_duplicate', 0):<6} {s['frames_consumed']:<8} {s['frames_dropped_queue']:<8} "
                          f"{s['queue_drop_rate']:<8.1f} {s['queue_depth']:<8} {s.get('current_fps', 0):<8.1f}")
                print("="*120)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    
    finally:
        end_time = time.time()
        duration = end_time - start_time
        final_sys_stats = sys_monitor.get_stats()
        
        print("\n[INFO] Stopping worker threads...")
        for worker in workers:
            worker.stop()
        
        # Wait for threads to finish
        for worker in workers:
            worker.join(timeout=1.0)
        
        # Collect final stats BEFORE stopping provider (FPS resets to 0 after stop)
        total_frames_consumed = 0
        final_cam_stats = {}
        
        print("\n" + "="*110)
        print("[INFO] Final Frame Queue Statistics:")
        print("="*110)
        for cam_id in range(num_streams):
            s = provider.get_stats(cam_id)
            final_cam_stats[cam_id] = s  # Save for final summary
            total_frames_consumed += s['frames_consumed']
            print(f"Cam {cam_id}: received={s['frames_received']}, decoded={s['frames_decoded']}, "
                  f"duplicate={s.get('frames_duplicate', 0)}, consumed={s['frames_consumed']}, "
                  f"queue_dropped={s['frames_dropped_queue']} ({s['queue_drop_rate']:.1f}%)")
        print("="*110)
        
        cv2.destroyAllWindows()
        provider.stop()
        
        # Final Summary (using saved stats)
        print(f"\n{'='*80}")
        print(f"[FINAL SUMMARY]")
        print(f"{'='*80}")
        print(f"Buffer Mode: {buffer_mode}")
        print(f"GPU Available: {gpu_available}")
        print(f"Duration: {duration:.1f}s")
        print(f"Total Display Frames: {global_frame_count}")
        print(f"Total Frames Consumed: {total_frames_consumed:,}")
        
        # Resource Usage
        print(f"\n[Resource Usage (this process)]")
        print(f"  CPU: {final_sys_stats['cpu']:.1f}%")
        print(f"  RAM: {final_sys_stats['ram_mb']:.0f} MB")
        print(f"  GPU: {final_sys_stats['gpu']:.1f}%")
        print(f"  VRAM: {final_sys_stats['vram_mb']:.0f} MB")
        
        print(f"\n{'Camera':<10} {'Avg FPS':<12} {'Source FPS':<12} {'Frames Consumed':<15}")
        print("-" * 55)
        
        for cam_id in range(num_streams):
            s = final_cam_stats[cam_id]  # Use saved stats
            fps = s.get('current_fps', 0.0)
            source_fps = s.get('source_fps', 0.0)
            consumed = s['frames_consumed']
            print(f"Cam {cam_id:<6} {fps:<12.2f} {source_fps:<12.2f} {consumed:<15}")
        
        print("-" * 55)
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
