import sys
import os
import time
import pprint

# Add the library path (../src)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

try:
    import rtspmodule
except ImportError:
    print("Error: Could not import rtspmodule. Make sure the package is in ../src")
    sys.exit(1)

def main():
    print("=== RTSPModule Basic Lifecycle Example ===")
    
    # Initialize the module
    rtsp = rtspmodule.RTSPModule()
    
    # Configure logging
    log_path = os.path.abspath("./logs")
    print(f"Setting log path to: {log_path}")
    rtsp.set_log_path(log_path)
    
    # Check GPU availability
    if rtsp.is_gpu_available():
        print("Hardware Acceleration: GPU (NVDEC/CUDA) IS available.")
    else:
        print("Hardware Acceleration: GPU unavailable. Using CPU fallback.")

    # Start streams
    config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs/config.yaml"))
    print(f"Loading config from: {config_file}")
    
    try:
        rtsp.start(config_file)
        print("Streams started successfully.")
    except RuntimeError as e:
        print(f"Failed to start streams: {e}")
        return

    # Check status
    if rtsp.is_running():
        num_streams = rtsp.stream_count()
        print(f"RTSP Client is running with {num_streams} streams configured.")
    
    # Monitor for a few seconds
    print("\nMonitoring stats for 5 seconds...")
    try:
        for i in range(5):
            time.sleep(1.0)
            print(f"\n--- Update {i+1}/5 ---")
            
            # Print stats for the first few active streams
            active_streams = rtsp.stream_count()
            limit = min(3, active_streams) # Just show first 3 to avoid spam
            
            for cam_id in range(limit):
                stats = rtsp.get_stats(cam_id)
                print(f"Camera {cam_id}: "
                      f"FPS={stats['current_fps']:.1f}/{stats['source_fps']:.1f}, "
                      f"Decoded={stats['frames_decoded']}, "
                      f"Dropped={stats['frames_dropped_queue']}")
                
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        # cleanup
        print("\nStopping streams...")
        rtsp.stop()
        print("Stopped.")
        
        if not rtsp.is_running():
            print("Verified: Client is no longer running.")

if __name__ == "__main__":
    main()
