"""
CPU Buffer Viewer - Displays frames from CPU ring buffer (FIFO)

Uses get_cpu_frame() to retrieve frames from the true ring buffer.
Frames are consumed in FIFO order (oldest first).
Supports NV12, RGB, BGR format conversion for display.
"""
import time
import math
import signal
import sys
import threading
from queue import Queue, Empty
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
import rtspmodule

# Global provider for signal handler
_provider = None

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print("\n[INTERRUPT] Received Ctrl+C, stopping...")
    if _provider:
        try:
            _provider.stop()
        except:
            pass
    cv2.destroyAllWindows()
    sys.exit(0)


def nv12_to_bgr(nv12_frame, width, height):
    """Convert NV12 frame to BGR using OpenCV."""
    # NV12 frame shape is (height * 1.5, width)
    # Reshape for cv2.cvtColor
    yuv = nv12_frame.reshape((int(height * 1.5), width))
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    return bgr


class CpuStreamWorker(threading.Thread):
    """Worker thread for processing frames from CPU buffer (FIFO)."""
    
    def __init__(self, provider, camera_id, target_w, target_h, output_queue):
        super().__init__(daemon=True)
        self.provider = provider
        self.camera_id = camera_id
        self.target_w = target_w
        self.target_h = target_h
        self.output_queue = output_queue
        self.running = True
        
    def stop(self):
        self.running = False
        
    def run(self):
        """Main processing loop."""
        while self.running and self.provider.is_running():
            try:
                # Get frame from CPU buffer (FIFO - oldest unread)
                frame_data = self.provider.get_cpu_frame(self.camera_id, timeout_ms=50)
                
                if not frame_data.get('valid', False):
                    time.sleep(0.01)
                    continue
                
                raw_frame = frame_data['data']
                width = frame_data['width']
                height = frame_data['height']
                fmt = frame_data['format']
                frame_id = frame_data['frame_id']
                
                # Convert to BGR for display
                if fmt == 'NV12':
                    bgr = nv12_to_bgr(raw_frame, width, height)
                elif fmt == 'BGR':
                    bgr = raw_frame
                elif fmt == 'RGB':
                    bgr = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
                elif fmt == 'RGBA':
                    bgr = cv2.cvtColor(raw_frame, cv2.COLOR_RGBA2BGR)
                elif fmt == 'BGRA':
                    bgr = cv2.cvtColor(raw_frame, cv2.COLOR_BGRA2BGR)
                else:
                    # Unknown format, try to display as-is
                    bgr = raw_frame
                
                # Resize to target
                resized = cv2.resize(bgr, (self.target_w, self.target_h))
                
                # Put in output queue
                try:
                    if self.output_queue.full():
                        try:
                            self.output_queue.get_nowait()
                        except Empty:
                            pass
                    self.output_queue.put_nowait({
                        'camera_id': self.camera_id,
                        'frame': resized,
                        'frame_id': frame_id,
                        'format': fmt,
                    })
                except:
                    pass
                
                # Small sleep to not overwhelm CPU
                time.sleep(0.01)
                    
            except Exception as e:
                print(f"[ERROR] Worker {self.camera_id}: {e}")
                time.sleep(0.1)


def main():
    global _provider
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("[INFO] Initializing CPU Buffer Viewer...")
    print("[INFO] This viewer uses get_cpu_frame() to read from CPU RAM buffer.")
    
    provider = rtspmodule.RTSPModule()
    _provider = provider
    
    provider.start("configs/config.yaml")
    
    # Wait for streams to initialize and buffer to fill
    print("[INFO] Waiting for CPU buffer to fill...")
    time.sleep(3.0)
    
    num_streams = provider.stream_count()
    print(f"[INFO] Started {num_streams} streams.")
    
    # Check if CPU buffer is enabled
    info = provider.get_cpu_buffer_info(0)
    if info['buffer_capacity'] == 0:
        print("[WARNING] CPU buffer is not enabled! Set 'cpu_buffer_enabled: true' in config.yaml")
        print("[INFO] Continuing anyway, but no frames will be available...")
    else:
        print(f"[INFO] CPU buffer: {info['buffer_count']}/{info['buffer_capacity']} frames, "
              f"{info['memory_usage_mb']:.1f}MB, format={info['format']}")
    
    print("[INFO] Press 'q' to quit.")
    
    TARGET_W = 640
    TARGET_H = 360
    
    cols = int(math.ceil(math.sqrt(num_streams)))
    rows = int(math.ceil(num_streams / cols))
    
    output_queues = {i: Queue(maxsize=2) for i in range(num_streams)}
    workers = []
    
    # Start worker threads
    for cam_id in range(num_streams):
        worker = CpuStreamWorker(provider, cam_id, TARGET_W, TARGET_H, output_queues[cam_id])
        workers.append(worker)
        worker.start()
    
    print(f"[INFO] Spawned {len(workers)} worker threads.")
    
    display_frames = {}
    
    start_time = time.time()
    last_stats_print = time.time()
    STATS_PRINT_INTERVAL = 5.0
    
    try:
        while provider.is_running():
            # Collect frames from workers
            for cam_id in range(num_streams):
                try:
                    data = output_queues[cam_id].get_nowait()
                    display_frames[cam_id] = data
                except Empty:
                    pass
            
            # Build display grid
            display_list = []
            for cam_id in range(num_streams):
                if cam_id in display_frames:
                    data = display_frames[cam_id]
                    frame = data['frame'].copy()
                    frame_id = data.get('frame_id', 0)
                    fmt = data.get('format', '?')
                else:
                    frame = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
                    frame_id = 0
                    fmt = '?'
                    cv2.putText(frame, "Waiting...", (TARGET_W//2 - 50, TARGET_H//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                
                # Get buffer info
                buf_info = provider.get_cpu_buffer_info(cam_id)
                stats = provider.get_stats(cam_id)
                
                # Overlay info
                mode_text = "FIFO"
                cv2.putText(frame, f"Cam {cam_id} [{mode_text}] | FPS: {stats['current_fps']:.1f}", 
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                cv2.putText(frame, f"Frame: {frame_id} | Buf: {buf_info['buffer_count']}/{buf_info['buffer_capacity']}", 
                           (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                cv2.putText(frame, f"Fmt: {fmt} | Mem: {buf_info['memory_usage_mb']:.1f}MB", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                display_list.append(frame)
            
            # Pad to fill grid
            total_slots = rows * cols
            while len(display_list) < total_slots:
                display_list.append(np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8))
            
            # Create grid
            grid_rows = []
            for r in range(rows):
                row_imgs = display_list[r * cols : (r + 1) * cols]
                grid_rows.append(np.hstack(row_imgs))
            
            final_grid = np.vstack(grid_rows)
            
            # Global stats
            total_fps = sum(provider.get_stats(i)['current_fps'] for i in range(num_streams))
            total_mem = sum(provider.get_cpu_buffer_info(i)['memory_usage_mb'] for i in range(num_streams))
            
            cv2.putText(final_grid, 
                       f"CPU BUFFER (FIFO) | Total FPS: {total_fps:.1f} | RAM: {total_mem:.1f}MB | Press 'q' quit", 
                       (20, final_grid.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Display
            final_grid = cv2.resize(final_grid, (1920, 1080))
            cv2.imshow("CPU Buffer Viewer", final_grid)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            
            # Print stats periodically
            now = time.time()
            if now - last_stats_print >= STATS_PRINT_INTERVAL:
                last_stats_print = now
                elapsed = now - start_time
                print(f"\n[CPU BUFFER STATS] @ {elapsed:.1f}s | RAM: {total_mem:.1f}MB")
                print("-" * 70)
                for cam_id in range(min(num_streams, 5)):  # Only first 5
                    info = provider.get_cpu_buffer_info(cam_id)
                    stats = provider.get_stats(cam_id)
                    print(f"  Cam {cam_id}: buf={info['buffer_count']}/{info['buffer_capacity']}, "
                          f"mem={info['memory_usage_mb']:.1f}MB, fps={stats['current_fps']:.1f}")
                if num_streams > 5:
                    print(f"  ... and {num_streams - 5} more streams")
                    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    
    finally:
        print("\n[INFO] Stopping worker threads...")
        for worker in workers:
            worker.stop()
        
        for worker in workers:
            worker.join(timeout=1.0)
        
        # Print final stats
        print("\n[INFO] Final CPU Buffer Statistics:")
        print("=" * 70)
        total_mem = 0
        for cam_id in range(num_streams):
            info = provider.get_cpu_buffer_info(cam_id)
            stats = provider.get_stats(cam_id)
            total_mem += info['memory_usage_mb']
            print(f"Cam {cam_id}: buffer={info['buffer_count']}/{info['buffer_capacity']}, "
                  f"mem={info['memory_usage_mb']:.1f}MB, decoded={stats['frames_decoded']}")
        print(f"\nTotal RAM usage: {total_mem:.1f}MB")
        print("=" * 70)
        
        cv2.destroyAllWindows()
        provider.stop()
        print(f"[INFO] Duration: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
