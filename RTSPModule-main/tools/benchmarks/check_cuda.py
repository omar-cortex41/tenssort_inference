import ctypes
import sys

def check_cuda():
    print("Checking CUDA availability via ctypes...")
    try:
        # Load libcuda.so
        cuda = ctypes.CDLL("libcuda.so.1")
        
        # Call cuInit(0)
        res = cuda.cuInit(0)
        print(f"cuInit(0) returned: {res}")
        
        if res == 0:
            count = ctypes.c_int()
            cuda.cuDeviceGetCount(ctypes.byref(count))
            print(f"CUDA Device Count: {count.value}")
            
            for i in range(count.value):
                name = ctypes.create_string_buffer(256)
                dev = ctypes.c_int()
                cuda.cuDeviceGet(ctypes.byref(dev), i)
                cuda.cuDeviceGetName(name, 256, dev)
                print(f"Device {i}: {name.value.decode('utf-8')}")
            return True
        else:
            print("Failed to initialize CUDA")
            return False
            
    except OSError as e:
        print(f"Could not load libcuda.so.1: {e}")
        return False
    except Exception as e:
        print(f"Error checking CUDA: {e}")
        return False

if __name__ == "__main__":
    if check_cuda():
        print("CUDA check PASSED")
        sys.exit(0)
    else:
        print("CUDA check FAILED")
        sys.exit(1)
