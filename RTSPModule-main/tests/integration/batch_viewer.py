import cv2
import numpy as np
import time
import math
import sys
import os

# Add lib to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
import rtspmodule

def create_grid(images, valid_mask, rows, cols, cell_w, cell_h):
    """
    Arrange a batch of images into a single grid image.
    images: numpy array (N, H, W, C)
    valid_mask: boolean array (N,)
    """
    batch_size = images.shape[0]
    grid_h = rows * cell_w 
    frame_h, frame_w = images.shape[1], images.shape[2]
    
    # Create canvas
    canvas = np.zeros((rows * frame_h, cols * frame_w, 3), dtype=np.uint8)
    
    for i in range(batch_size):
        r, c = divmod(i, cols)
        
        y_start = r * frame_h
        x_start = c * frame_w
        
        if valid_mask[i]:
            # Direct copy from batch buffer to canvas
            canvas[y_start:y_start+frame_h, x_start:x_start+frame_w] = images[i]
        else:
            # Draw placeholder for invalid
            cv2.putText(canvas, f"CAM {i} (NC)", (x_start+50, y_start+frame_h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
    return canvas

def main():
    config_path = "configs/config.yaml"
    
    print(f"[INFO] Initializing RTSPModule...")
    provider = rtspmodule.RTSPModule()
    
    provider.start(config_path)
    
    # Wait a moment for connections
    print("[INFO] Waiting for streams to connect...")
    time.sleep(2)
    
    stream_count = provider.stream_count()
    print(f"[INFO] Detected {stream_count} streams")
    
    if stream_count == 0:
        print("[ERROR] No streams found in config")
        provider.stop()
        return

    # Calculate grid layout
    cols = math.ceil(math.sqrt(stream_count))
    rows = math.ceil(stream_count / cols)
    
    # Scale down for display if too large (1920x1080 * 22 is huge)
    target_w = 640
    target_h = 360
    
    # We request full resolution from batch, then resize for display
    # (Or we could configure batch to scale, but currently batch uses source res)
    
    camera_ids = list(range(stream_count))
    
    print(f"[INFO] Starting batch viewer loop. Grid: {rows}x{cols}")
    print("[INFO] Press 'q' to exit")
    
    fps_history = []
    

    while True:
        t0 = time.perf_counter()
        
        # 1. Get Batch
        # Use small timeout to approximate non-blocking poll
        batch = provider.get_batch(camera_ids, timeout_ms=40)
        
        batch_time = time.perf_counter() - t0
        
        # 2. Process for Display
        data = batch['data']
        valid_mask = batch['valid_mask']
        fmt = batch.get('format', 'BGR')
        
        if data is None or data.size == 0:
            continue

        # Handle NV12 format (N, H*1.5, W) -> Convert to BGR for display
        if fmt == "NV12" or (data.ndim == 3 and batch['height'] > 0 and data.shape[1] == int(batch['height'] * 1.5)):
            # Warning: Converting NV12 on CPU in Python is slow. 
            # Ideally use GPU or request BGR from C++.
            # For visualization, we'll convert a subset if possible, or just the first/Y plane (grayscale)
            
            # Fast path: Just show Y plane (grayscale) to avoid expensive conversion
            H = batch['height']
            W = batch['width']
            
            # Extract Y plane only: data[:, :H, :]
            display_frames = data[:, :H, :]
            
            # Convert to 3-channel grayscale for consistency
            # display_frames = np.stack([display_frames]*3, axis=-1)  # Takes memory
            
            # Decimate
            display_frames = display_frames[:, ::3, ::3]
            
            # Add channel dim for grid logic
            display_frames = display_frames[..., np.newaxis] # (N, h, w, 1)
            display_frames = np.repeat(display_frames, 3, axis=3) # (N, h, w, 3)

        else:
            # BGR/RGB path (N, H, W, C)
            try:
                display_frames = data[:, ::3, ::3, :] # 1/9th size
            except IndexError:
                print(f"[ERROR] Unexpected shape: {data.shape} for format {fmt}")
                break
        
        # Arrange in grid
        N, H, W, C = display_frames.shape
        
        grid_img = np.zeros((rows * H, cols * W, 3), dtype=np.uint8)
        
        # Draw performance stats of batch fetch
        cv2.putText(grid_img, f"Batch Latency: {batch_time*1000:.1f} ms", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(grid_img, f"Format: {fmt}", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        for i in range(N):
            r, c = divmod(i, cols)
            y = r * H
            x = c * W
            
            if valid_mask[i]:
                grid_img[y:y+H, x:x+W] = display_frames[i]
                
                # Overlay info
                meta = batch['metadata'][i]
                info_txt = f"CAM {i} | {meta['width']}x{meta['height']}"
                cv2.putText(grid_img, info_txt, (x+10, y+H-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                # Offline placeholder
                cv2.rectangle(grid_img, (x, y), (x+W, y+H), (20, 20, 20), -1)
                cv2.putText(grid_img, f"CAM {i} OFFLINE", (x+10, y+H//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        total_time = time.perf_counter() - t0
        fps = 1.0 / total_time if total_time > 0 else 0
        fps_history.append(fps)
        if len(fps_history) > 30: fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)
        
        cv2.putText(grid_img, f"Display FPS: {avg_fps:.1f}", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Batch Viewer", cv2.resize(grid_img,(1920,1080)))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
                
        
    print("[INFO] Stopping...")
    provider.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
