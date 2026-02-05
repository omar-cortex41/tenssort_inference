import time
import math
import cv2
import cupy as cp
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import rtspmodule
from tools.minimal_client.frame_sync import FrameSynchronizer

def main():
    print("[INFO] Initializing RTSP Provider...")
    
    provider = rtspmodule.RTSPModule()
    provider.start("config.yaml")
    
    syncer = FrameSynchronizer(provider)
    time.sleep(2.0)
    
    num_streams = provider.stream_count()
    TARGET_W = 640
    TARGET_H = 360
    
    cols = int(math.ceil(math.sqrt(num_streams)))
    rows = int(math.ceil(num_streams / cols))
    GRID_WIDTH = cols * TARGET_W
    GRID_HEIGHT = rows * TARGET_H
    output_filename = "./output/recording.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    target_fps = 30.0
    
    print(f"[INFO] Initializing VideoWriter: {GRID_WIDTH}x{GRID_HEIGHT} @ {target_fps} fps")
    writer = cv2.VideoWriter(output_filename, fourcc, target_fps, (GRID_WIDTH, GRID_HEIGHT))

    last_frames_cpu = {}
    
    stream_stats = {
        i: {'count': 0, 'fps': 0.0, 'last_time': time.time()} 
        for i in range(num_streams)
    }
    
    global_frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            batch = syncer.get_batch(timeout_sec=0.04) 

            for cam_id in range(num_streams):
                info = batch.get(cam_id)
                
                if info is not None:
                    try:
                        w, h = info['width'], info['height']
                        stride = info.get('stride', w)
                        
                        actual_gpu_size = int(stride * h * 1.5)
                        mem = cp.cuda.UnownedMemory(info['ptr'], actual_gpu_size, None)
                        memptr = cp.cuda.MemoryPointer(mem, 0)
                        
                        h_nv12 = int(h * 1.5)
                        gpu_nv12 = cp.ndarray((h_nv12, stride), dtype=cp.uint8, memptr=memptr)
                        
                        cpu_nv12 = gpu_nv12.get()
                        
                        bgr = cv2.cvtColor(cpu_nv12, cv2.COLOR_YUV2BGR_NV12)
                        resized_frame = cv2.resize(bgr, (TARGET_W, TARGET_H))
                        
                        last_frames_cpu[cam_id] = resized_frame

                        stats = stream_stats[cam_id]
                        stats['count'] += 1
                        now = time.time()
                        if now - stats['last_time'] >= 1.0:
                            stats['fps'] = stats['count'] / (now - stats['last_time'])
                            stats['count'] = 0
                            stats['last_time'] = now
                            
                    except Exception as e:
                        print(f"[ERROR] Cam {cam_id} processing failed: {e}")

            display_list = []
            
            for cam_id in range(num_streams):
                if cam_id in last_frames_cpu:
                    frame = last_frames_cpu[cam_id].copy()
                else:
                    frame = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
                    cv2.putText(frame, "Waiting...", (TARGET_W//2 - 50, TARGET_H//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                
                fps_val = stream_stats[cam_id]['fps']
                color = (0, 255, 0) if fps_val > 0 else (0, 0, 255)
                cv2.putText(frame, f"Cam {cam_id} | FPS: {fps_val:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
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
            avg_fps = global_frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(final_grid, f"REC | Global FPS: {avg_fps:.1f}", (20, final_grid.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            writer.write(final_grid)
            
            if global_frame_count % 30 == 0:
                print(f"[RECORDING] Frames: {global_frame_count} | Avg FPS: {avg_fps:.2f}")

    except KeyboardInterrupt:
        print("\n[INFO] Stop signal received.")
    
    finally:
        print("[INFO] Cleaning up...")
        if 'writer' in locals() and writer.isOpened():
            writer.release()
            print(f"[INFO] Video saved to {output_filename}")
            
        provider.stop()
        print(f"[INFO] Total Duration: {time.time() - start_time:.2f}s | Total Frames: {global_frame_count}")

if __name__ == "__main__":
    main()