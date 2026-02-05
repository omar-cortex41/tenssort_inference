"""
VRAM Profiler - Step-by-step GPU memory tracking

Measures VRAM at each stage to identify where memory is consumed.
"""
import subprocess
import time

def get_vram_mb():
    """Get current GPU VRAM usage via nvidia-smi."""
    try:
        cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
        out = subprocess.check_output(cmd.split(), encoding='utf-8').strip()
        return float(out)
    except:
        return 0.0

def print_vram(label, baseline=0):
    vram = get_vram_mb()
    delta = vram - baseline if baseline else 0
    if baseline:
        print(f"  [{label:40}] VRAM: {vram:6.0f} MB  (+{delta:6.0f} MB)")
    else:
        print(f"  [{label:40}] VRAM: {vram:6.0f} MB")
    return vram

def main():
    print("=" * 70)
    print("VRAM PROFILER - Step-by-step memory tracking")
    print("=" * 70)
    
    # Baseline before anything
    baseline = print_vram("Baseline (before imports)")
    
    # Step 1: Import pynvml
    print("\n[STEP 1] Import pynvml...")
    try:
        import pynvml
        pynvml.nvmlInit()
        print_vram("After pynvml init", baseline)
    except ImportError:
        print("  pynvml not available")
    
    # Step 2: Import CuPy (triggers CUDA init)
    print("\n[STEP 2] Import CuPy...")
    import cupy as cp
    after_cupy = print_vram("After cupy import", baseline)
    
    # Step 3: First CuPy operation (may trigger lazy init)
    print("\n[STEP 3] First CuPy operation (triggers CUDA context)...")
    _ = cp.zeros(1)
    cp.cuda.Stream.null.synchronize()
    after_cupy_op = print_vram("After cp.zeros(1)", baseline)
    
    # Step 4: Import RTSPModule
    print("\n[STEP 4] Import RTSPModule...")
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
    import rtspmodule
    after_import = print_vram("After RTSPModule import", baseline)
    
    # Step 5: Create RTSPModule instance
    print("\n[STEP 5] Create RTSPModule instance...")
    provider = rtspmodule.RTSPModule()
    after_create = print_vram("After RTSPModule()", baseline)
    
    # Step 6: Start (this runs initCudaContext + decoders)
    print("\n[STEP 6] Start RTSPModule (probe + decoder init)...")
    provider.start("configs/config.yaml")
    time.sleep(3)  # Wait for pipeline to stabilize
    after_start = print_vram("After provider.start()", baseline)
    
    # Step 7: Get first frame
    print("\n[STEP 7] Get first GPU frame...")
    for _ in range(10):
        info = provider.get_cuda_frame(0, 100)
        if info.get("valid"):
            break
        time.sleep(0.1)
    after_frame = print_vram("After first frame", baseline)
    
    # Step 8: Wrap in CuPy
    print("\n[STEP 8] Wrap frame in CuPy array...")
    if info.get("valid"):
        mem = cp.cuda.UnownedMemory(info['ptr'], info['size'], None)
        memptr = cp.cuda.MemoryPointer(mem, 0)
        gpu_array = cp.ndarray(info['shape'], dtype=cp.uint8, memptr=memptr)
        _ = gpu_array.sum()  # Force access
        cp.cuda.Stream.null.synchronize()
    after_wrap = print_vram("After CuPy wrap", baseline)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - Memory allocations by stage:")
    print("=" * 70)
    print(f"  CuPy context:          {after_cupy_op - baseline:6.0f} MB")
    print(f"  RTSPModule import:     {after_import - after_cupy_op:6.0f} MB")
    print(f"  RTSPModule create:     {after_create - after_import:6.0f} MB") 
    print(f"  Probe + Decoder init:  {after_start - after_create:6.0f} MB")
    print(f"  First frame:           {after_frame - after_start:6.0f} MB")
    print(f"  CuPy wrap:             {after_wrap - after_frame:6.0f} MB")
    print("-" * 70)
    print(f"  TOTAL DELTA:           {after_wrap - baseline:6.0f} MB")
    print("=" * 70)
    
    # Cleanup
    provider.stop()
    time.sleep(1)
    print_vram("After cleanup", baseline)

if __name__ == "__main__":
    main()
