import sys
import os
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import rtspmodule

def main():
    print("=== RTSPModule Batch Capture Example ===")
    
    rtsp = rtspmodule.RTSPModule()
    
    # Batch requires CPU buffer mode
    if not rtsp.is_cpu_buffer_enabled():
        # If GPU was preferred but failed, it might be enabled.
        # If explicitly disabled in config, we can't use batch.
        print("Note: get_batch() requires 'cpu_buffer_enabled: true'.")
        # Proceeding assuming it might be enabled or user will fix config.
        
    rtsp.start(os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs/config.yaml")))
    time.sleep(2)
    
    num_streams = rtsp.stream_count()
    if num_streams == 0:
        return
        
    print(f"Batch processing all {num_streams} streams...")
    
    # Create list of all camera IDs
    camera_ids = list(range(num_streams))
    
    try:
        start_time = time.time()
        frames_processed = 0
        
        for i in range(50):
            # Fetch batch of frames
            # timeout_ms=10 ensures we don't wait long if some cams are slow
            batch_result = rtsp.get_batch(camera_ids, timeout_ms=10)
            
            # Unpack results
            batch_data = batch_result['data']      # Numpy array (N, H, W, C)
            valid_mask = batch_result['valid_mask'] # Boolean mask of valid frames
            metadata = batch_result['metadata']     # List of dicts
            
            if batch_data is not None:
                valid_count = batch_result['valid_count']
                total_count = batch_result['count']
                
                # batch_data is a single contiguous numpy array
                # Use it directly for inference (e.g., passed to TensorRT or PyTorch)
                
                print(f"Batch {i}: {valid_count}/{total_count} valid frames. "
                      f"Shape: {batch_data.shape} Type: {batch_data.dtype}")
                
                # Check metadata for specific frames
                for idx, valid in enumerate(valid_mask):
                    if valid:
                        meta = metadata[idx]
                        # e.g., print timestamp delta for the first camera
                        if idx == 0:
                             print(f"   Cam 0 Timestamp: {meta['timestamp_ns']} ns")
            
            frames_processed += 1
            # Simulate processing time
            time.sleep(0.01)

        elapsed = time.time() - start_time
        print(f"\nCompleted {frames_processed} batches in {elapsed:.2f}s")
        print(f"Effective Batch FPS: {frames_processed/elapsed:.1f}")
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rtsp.stop()

if __name__ == "__main__":
    main()
