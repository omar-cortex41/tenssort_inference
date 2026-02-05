import sys
import os
import time
import ctypes

# Add the library path
# Add the library path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import rtspmodule

# Extract potential CUDA dependencies
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

def main():
    print("=== RTSPModule GPU Frame Capture Example ===")
    
    rtsp = rtspmodule.RTSPModule()
    
    # 1. Check Hardware Availability
    if not rtsp.is_gpu_available():
        print("ERROR: GPU acceleration is not available on this system.")
        print("This example requires a working NVIDIA GPU and generic GStreamer NVDEC plugins.")
        return

    # 2. Start
    config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs/config.yaml"))
    rtsp.start(config_file)
    time.sleep(2) # Warm up
    
    try:
        num_streams = rtsp.stream_count()
        if num_streams == 0:
            print("No streams.")
            return

        print(f"Acquiring CUDA frames from Camera 0 (requires 'cpu_buffer_enabled: false' in config)...")
        
        # NOTE: get_cuda_frame ONLY works if cpu_buffer_enabled is False.
        # If it's True, the frame is downloaded to CPU automatically, so direct GPU access might be disabled
        # depending on internal implementation. The binding docs say:
        # "unavailable when cpu_buffer_enabled=true".
        
        if rtsp.is_cpu_buffer_enabled():
            print("ERROR: Config 'cpu_buffer_enabled' is set to True.")
            print("Zero-copy GPU access requires 'cpu_buffer_enabled: false'.")
            print("Please edit configs/config.yaml to run this example.")
            return

        for i in range(20):
            # Get frame pointer info
            # timeout_ms=1000 to wait for a frame
            frame_info = rtsp.get_cuda_frame(camera_id=0, timeout_ms=1000)
            
            if frame_info['valid']:
                ptr_val = frame_info['ptr']
                width = frame_info['width']
                height = frame_info['height']
                size = frame_info['size']
                fmt = frame_info['format']
                
                print(f"Frame {frame_info['frame_id']}: {fmt} {width}x{height}, GPU Ptr: 0x{ptr_val:x}")
                
                if HAS_CUPY and ptr_val != 0:
                    # Create a CuPy array from the raw pointer
                    # Unowned memory wrapper (we don't own the underlying buffer, C++ does)
                    
                    # Create a memory pointer interface
                    mem = cp.cuda.UnownedMemory(ptr_val, size, None)
                    mptr = cp.cuda.MemoryPointer(mem, 0)
                    
                    # Create array based on format
                    if fmt in ["NV12", "I420"]:
                        # NV12 is H*1.5 height (luma + chroma)
                        h_yuv = int(height * 1.5)
                        shape = (h_yuv, width)
                        # uint8
                        gpu_array = cp.ndarray(shape, dtype=cp.uint8, memptr=mptr)
                        
                        # Example: Calculate a simple mean on GPU
                        mean_val = float(cp.mean(gpu_array))
                        print(f"   -> Average pixel intensity (GPU computed): {mean_val:.2f}")
                        
                    elif fmt in ["RGB", "BGR"]:
                        shape = (height, width, 3)
                        gpu_array = cp.ndarray(shape, dtype=cp.uint8, memptr=mptr)
                        print(f"   -> GPU Array shape: {gpu_array.shape}")

            else:
                print("No GPU frame available (timeout)")
                
    except RuntimeError as e:
        print(f"Runtime Error: {e}")
    except KeyboardInterrupt:
        pass
    finally:
        rtsp.stop()

if __name__ == "__main__":
    main()
