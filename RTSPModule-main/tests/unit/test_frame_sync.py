import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))) # Root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))) # Src

from tools.minimal_client.frame_sync import FrameSynchronizer
import rtspmodule


def main():
    provider = rtspmodule.RTSPModule()
    provider.start("config.yaml")
    time.sleep(2)
    # Create a FrameSynchronizer instance
    synchronizer = FrameSynchronizer(provider)
    
    try:
        while True:
            # Get a batch of frames with a timeout of 40ms
            batch_result = synchronizer.get_batch(timeout_sec=0.04)
            
            # Print the batch result
            print("\nBatch Result:")
            for cam_id, frame_info in batch_result.items():
                if frame_info is not None:
                    print(f"  Camera {cam_id}:")
                    print(f"    - Valid: {frame_info['valid']}")
                    print(f"    - Frame ID: {frame_info['frame_id']}")
                    print(f"    - Resolution: {frame_info['width']}x{frame_info['height']}")
                    print(f"    - CUDA Pointer: {hex(frame_info['ptr'])}")
                    print(f"    - Size: {frame_info['size']} bytes")
                else:
                    print(f"  Camera {cam_id}: No frame received (timeout)")
    
    except KeyboardInterrupt:
        print("\nStopping frame synchronization example.")

if __name__ == "__main__":
    main()