#!/usr/bin/env python3
"""
Batch Frame Retrieval FPS Benchmark.
Calculates throughput, latency, and delivery performance for multi-stream batching.
"""
import time
import sys
import os
import numpy as np

# Add lib directory to path
# Add src directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir))) # tests/unit -> tests -> root
# Or simpler:
project_dir = os.path.abspath(os.path.join(script_dir, "../../"))
src_dir = os.path.join(project_dir, 'src')
sys.path.insert(0, src_dir)

try:
    import rtspmodule
except ImportError as e:
    print(f"[ERROR] rtspmodule import failed: {e}")
    print("[TIP] Ensure the module is built and in src/rtspmodule.")
    sys.exit(1)

def run_benchmark(iterations=500, warm_up=50):
    print("=" * 80)
    print("BATCH API PERFORMANCE BENCHMARK")
    print("=" * 80)
    
    provider = rtspmodule.RTSPModule()
    
    config_path = os.path.join(project_dir, "configs/config.yaml")
    try:
        provider.start(config_path)
    except Exception as e:
        print(f"[ERROR] Failed to start provider with {config_path}: {e}")
        return
    
    # Wait for streams to connect and stabilize
    print(f"[INFO] Initializing streams from {config_path}...")
    time.sleep(5)
    
    stream_count = provider.stream_count()
    if stream_count == 0:
        print("[ERROR] No streams active. Check your RTMPS/RTSP sources.")
        provider.stop()
        return

    print(f"[INFO] Active Streams: {stream_count}")
    print(f"[INFO] CPU Buffer:   {'Enabled' if provider.is_cpu_buffer_enabled() else 'Disabled'}")
    print(f"[INFO] GPU Available: {'Yes' if provider.is_gpu_available() else 'No'}")
    
    if not provider.is_cpu_buffer_enabled():
        print("[ERROR] get_batch() requires cpu_buffer_enabled: true in config.yaml")
        provider.stop()
        return

    camera_ids = list(range(stream_count))
    
    # --- Warm-up Phase ---
    print(f"[INFO] Warming up ({warm_up} iterations)...")
    for _ in range(warm_up):
        _ = provider.get_batch(camera_ids, timeout_ms=30)

    # --- Benchmark Phase ---
    print(f"[INFO] Benchmarking ({iterations} iterations)...")
    
    total_valid_frames = 0
    total_bytes = 0
    latencies = []
    
    start_time = time.perf_counter()
    
    for i in range(iterations):
        batch_start = time.perf_counter()
        
        batch = provider.get_batch(camera_ids, timeout_ms=33)
        
        batch_end = time.perf_counter()
        latencies.append(batch_end - batch_start)
        
        total_valid_frames += batch['valid_count']
        if batch['data'] is not None:
            total_bytes += batch['data'].nbytes
            
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{iterations} iterations...")

    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # --- Results Calculation ---
    avg_latency_ms = (sum(latencies) / len(latencies)) * 1000
    fps = total_valid_frames / total_time
    throughput_mb = total_bytes / (1024 * 1024)
    throughput_rate = throughput_mb / total_time
    
    # Efficiency: ratio of frames received vs expected
    expected_frames = iterations * stream_count
    efficiency = (total_valid_frames / expected_frames) * 100 if expected_frames > 0 else 0
    
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("-" * 80)
    print(f"  Total Iterations:      {iterations}")
    print(f"  Total Time:            {total_time:.3f} s")
    print(f"  Total Frames (Valid):  {total_valid_frames}")
    print(f"  Stream Count:          {stream_count}")
    print("-" * 80)
    print(f"  AVERAGE FPS (TOTAL):   {fps:.2f} fps")
    print(f"  FPS PER STREAM:        {fps/stream_count:.2f} fps")
    print(f"  AVG BATCH LATENCY:     {avg_latency_ms:.3f} ms")
    print(f"  THROUGHPUT:            {throughput_rate:.2f} MB/s")
    print(f"  DATA TRANSFERRED:      {throughput_mb:.2f} MB")
    print(f"  DELIVERY EFFICIENCY:   {efficiency:.1f}%")
    print("=" * 80)
    
    provider.stop()

if __name__ == "__main__":
    # Allow simple CLI overrides
    it = 500
    if len(sys.argv) > 1:
        try:
            it = int(sys.argv[1])
        except ValueError:
            pass
            
    run_benchmark(iterations=it)
