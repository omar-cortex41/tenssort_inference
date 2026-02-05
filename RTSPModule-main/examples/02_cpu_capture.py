import sys
import os
import time
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None

# Add the library path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import rtspmodule

def main():
    print("=== RTSPModule CPU Frame Capture Example ===")
    
    rtsp = rtspmodule.RTSPModule()
    
    # Check if we can use CPU functions
    if not rtsp.is_cpu_buffer_enabled() and rtsp.is_gpu_available():
        print("WARNING: CPU buffer is NOT enabled in config.")
        print("get_cpu_frame() will throw an error unless 'cpu_buffer_enabled: true' is set in config.yaml")
        # In a real app, you'd check this before starting.
        # For this example, we assume the user might have set it or GPU is missing (auto-fallback).
    
    config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs/config.yaml"))
    rtsp.start(config_file)
    
    print("Waiting for streams to initialize...")
    time.sleep(2)
    
    try:
        num_streams = rtsp.stream_count()
        if num_streams == 0:
            print("No streams configured.")
            return

        print(f"Reading frames from {num_streams} cameras...")
        
        frame_count = 0
        start_time = time.time()
        
        while frame_count < 100:
            # We'll just grab from the first camera for demonstration
            camera_id = 0 
            
            # Retrieve frame (blocking wait up to 100ms)
            frame_info = rtsp.get_cpu_frame(camera_id, timeout_ms=100)
            
            if frame_info['valid']:
                frame_data = frame_info['data'] # This is a numpy array
                width = frame_info['width']
                height = frame_info['height']
                fmt = frame_info['format']
                
                # Get buffer info (how full is the ring buffer?)
                buf_info = rtsp.get_cpu_buffer_info(camera_id)
                
                print(f"Frame {frame_info['frame_id']}: {width}x{height} {fmt} | "
                      f"Buffer: {buf_info['buffer_count']}/{buf_info['buffer_capacity']} frames", end='\r')
                
                # Visualization (optional)
                if cv2 is not None:
                    # Handle formats
                    if fmt == "NV12" or fmt == "I420":
                        # Convert YUV to BGR for display
                        # data shape is (h*1.5, w) for NV12
                        bgr = cv2.cvtColor(frame_data, cv2.COLOR_YUV2BGR_NV12)
                        cv2.imshow(f"Camera {camera_id}", bgr)
                    elif fmt in ["RGB", "BGR", "RGBA", "BGRA"]:
                         # OpenCV expects BGR. If RGB, convert.
                         if fmt == "RGB":
                             display_img = cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR)
                         elif fmt == "RGBA":
                             display_img = cv2.cvtColor(frame_data, cv2.COLOR_RGBA2BGR)
                         else:
                             display_img = frame_data
                             
                         cv2.imshow(f"Camera {camera_id}", display_img)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
            else:
                # No frame available yet
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        rtsp.stop()
        if cv2 is not None:
            cv2.destroyAllWindows()
            
    print(f"\nProcessed {frame_count} frames.")

if __name__ == "__main__":
    main()
