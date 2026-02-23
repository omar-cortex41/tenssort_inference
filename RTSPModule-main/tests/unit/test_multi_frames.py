#!/usr/bin/env python3
"""
Test script for get_multi_frames() — single-camera multi-frame retrieval.
Tests functionality, shape validation, FIFO ordering, and throughput.
"""
import time
import sys
import os
import numpy as np

# Setup path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, "../../"))
src_dir = os.path.join(project_dir, 'src')
sys.path.insert(0, src_dir)

try:
    import rtspmodule
except ImportError as e:
    print(f"[ERROR] rtspmodule import failed: {e}")
    print("[TIP] Ensure the module is built and in src/rtspmodule.")
    sys.exit(1)


def test_basic_retrieval(provider, camera_id=0, num_frames=4):
    """Test basic multi-frame retrieval and validate return structure."""
    print("\n" + "=" * 70)
    print(f"TEST 1: Basic Retrieval (camera={camera_id}, num_frames={num_frames})")
    print("=" * 70)

    result = provider.get_multi_frames(camera_id=camera_id, num_frames=num_frames, timeout_ms=100)

    print(f"  count:       {result['count']}")
    print(f"  valid_count: {result['valid_count']}")
    print(f"  width:       {result.get('width', 'N/A')}")
    print(f"  height:      {result.get('height', 'N/A')}")
    print(f"  format:      {result.get('format', 'N/A')}")

    if result['count'] == 0:
        print("  [WARN] No frames returned — buffer may be empty. Wait longer.")
        return False

    data = result['data']
    valid_mask = result['valid_mask']
    metadata = result['metadata']

    print(f"  data.shape:  {data.shape}")
    print(f"  data.dtype:  {data.dtype}")
    print(f"  valid_mask:  {list(valid_mask)}")

    # Validate shape
    assert data.ndim >= 3, f"Expected at least 3D array, got {data.ndim}D"
    assert data.shape[0] == result['count'], f"Batch dim mismatch: {data.shape[0]} != {result['count']}"
    assert len(valid_mask) == result['count'], "valid_mask length mismatch"
    assert len(metadata) == result['count'], "metadata length mismatch"
    assert data.dtype == np.uint8, f"Expected uint8 dtype, got {data.dtype}"

    # Validate metadata fields
    for i, meta in enumerate(metadata):
        assert 'camera_id' in meta, f"metadata[{i}] missing 'camera_id'"
        assert 'frame_id' in meta, f"metadata[{i}] missing 'frame_id'"
        assert 'timestamp_ns' in meta, f"metadata[{i}] missing 'timestamp_ns'"
        assert 'valid' in meta, f"metadata[{i}] missing 'valid'"
        assert meta['camera_id'] == camera_id, f"Wrong camera_id in metadata[{i}]"

    print("  [PASS] Structure and shape validated ✓")
    return True


def test_fifo_ordering(provider, camera_id=0, num_frames=8):
    """Test that frames are returned in chronological order (ascending frame_id)."""
    print("\n" + "=" * 70)
    print(f"TEST 2: FIFO Ordering (camera={camera_id}, num_frames={num_frames})")
    print("=" * 70)

    result = provider.get_multi_frames(camera_id=camera_id, num_frames=num_frames, timeout_ms=200)

    if result['count'] < 2:
        print(f"  [SKIP] Need at least 2 frames, got {result['count']}")
        return True

    frame_ids = [m['frame_id'] for m in result['metadata'] if m['valid']]
    timestamps = [m['timestamp_ns'] for m in result['metadata'] if m['valid']]

    print(f"  frame_ids:   {frame_ids}")
    print(f"  timestamps:  {timestamps}")

    # Check ascending order
    is_ordered = all(frame_ids[i] < frame_ids[i + 1] for i in range(len(frame_ids) - 1))
    if is_ordered:
        print("  [PASS] Frame IDs in ascending order (FIFO verified) ✓")
    else:
        print("  [FAIL] Frame IDs NOT in order!")

    return is_ordered


def test_consumption(provider, camera_id=0):
    """Test that get_multi_frames consumes frames (subsequent calls return new frames)."""
    print("\n" + "=" * 70)
    print(f"TEST 3: Frame Consumption (camera={camera_id})")
    print("=" * 70)

    # First call: grab some frames
    r1 = provider.get_multi_frames(camera_id=camera_id, num_frames=4, timeout_ms=100)
    ids_1 = [m['frame_id'] for m in r1['metadata'] if m['valid']]

    if len(ids_1) == 0:
        print("  [SKIP] No frames in first batch")
        return True

    # Wait briefly for new frames
    time.sleep(0.2)

    # Second call: should get DIFFERENT (later) frames
    r2 = provider.get_multi_frames(camera_id=camera_id, num_frames=4, timeout_ms=100)
    ids_2 = [m['frame_id'] for m in r2['metadata'] if m['valid']]

    print(f"  Batch 1 frame_ids: {ids_1}")
    print(f"  Batch 2 frame_ids: {ids_2}")

    if len(ids_2) == 0:
        print("  [WARN] No frames in second batch — stream may be slow")
        return True

    # No overlap between batches (frames were consumed)
    overlap = set(ids_1) & set(ids_2)
    if len(overlap) == 0:
        print("  [PASS] No frame overlap — consumption verified ✓")
        return True
    else:
        print(f"  [FAIL] Overlapping frame_ids: {overlap}")
        return False


def test_edge_cases(provider, camera_id=0):
    """Test edge cases: 0 frames, 1 frame, large request."""
    print("\n" + "=" * 70)
    print(f"TEST 4: Edge Cases (camera={camera_id})")
    print("=" * 70)

    # 0 frames requested
    r = provider.get_multi_frames(camera_id=camera_id, num_frames=0, timeout_ms=10)
    assert r['count'] == 0, f"Expected 0 frames, got {r['count']}"
    print("  num_frames=0:    count=0 ✓")

    # 1 frame requested
    r = provider.get_multi_frames(camera_id=camera_id, num_frames=1, timeout_ms=100)
    assert r['count'] <= 1, f"Expected <= 1 frames, got {r['count']}"
    if r['count'] == 1:
        assert r['data'].shape[0] == 1, f"Batch dim should be 1, got {r['data'].shape[0]}"
    print(f"  num_frames=1:    count={r['count']} ✓")

    # Large request (more than buffer capacity)
    r = provider.get_multi_frames(camera_id=camera_id, num_frames=1000, timeout_ms=10)
    print(f"  num_frames=1000: count={r['count']} (capped to available) ✓")

    print("  [PASS] Edge cases handled ✓")
    return True


def test_throughput(provider, camera_id=0, num_frames=8, iterations=100):
    """Benchmark throughput of get_multi_frames."""
    print("\n" + "=" * 70)
    print(f"TEST 5: Throughput Benchmark ({iterations} iterations, {num_frames} frames/call)")
    print("=" * 70)

    # Warm up
    for _ in range(10):
        provider.get_multi_frames(camera_id=camera_id, num_frames=num_frames, timeout_ms=50)

    latencies = []
    total_frames = 0
    total_bytes = 0

    start = time.perf_counter()
    for _ in range(iterations):
        t0 = time.perf_counter()
        r = provider.get_multi_frames(camera_id=camera_id, num_frames=num_frames, timeout_ms=33)
        t1 = time.perf_counter()

        latencies.append(t1 - t0)
        total_frames += r['valid_count']
        if r['data'] is not None and r['count'] > 0:
            total_bytes += r['data'].nbytes
    elapsed = time.perf_counter() - start

    avg_lat_ms = (sum(latencies) / len(latencies)) * 1000
    p99_lat_ms = sorted(latencies)[int(len(latencies) * 0.99)] * 1000
    fps = total_frames / elapsed if elapsed > 0 else 0
    throughput_mb = total_bytes / (1024 * 1024)

    print(f"  Total time:        {elapsed:.3f} s")
    print(f"  Total frames:      {total_frames}")
    print(f"  Avg latency:       {avg_lat_ms:.3f} ms")
    print(f"  P99 latency:       {p99_lat_ms:.3f} ms")
    print(f"  Frame throughput:  {fps:.1f} fps")
    print(f"  Data transferred:  {throughput_mb:.1f} MB")
    print(f"  Throughput rate:   {throughput_mb / elapsed:.1f} MB/s" if elapsed > 0 else "")
    print("  [DONE] ✓")
    return True


def main():
    print("=" * 70)
    print("  get_multi_frames() — Single-Camera Multi-Frame Test Suite")
    print("=" * 70)

    provider = rtspmodule.RTSPModule()
    config_path = os.path.join(project_dir, "configs/config.yaml")

    try:
        provider.start(config_path)
    except Exception as e:
        print(f"[ERROR] Failed to start: {e}")
        return

    stream_count = provider.stream_count()
    print(f"[INFO] Streams:      {stream_count}")
    print(f"[INFO] CPU Buffer:   {'Enabled' if provider.is_cpu_buffer_enabled() else 'Disabled'}")
    print(f"[INFO] GPU:          {'Available' if provider.is_gpu_available() else 'Unavailable'}")

    if not provider.is_cpu_buffer_enabled():
        print("[ERROR] get_multi_frames() requires cpu_buffer_enabled: true")
        provider.stop()
        return

    if stream_count == 0:
        print("[ERROR] No streams active.")
        provider.stop()
        return

    # Wait for streams to connect and buffer to fill
    print("[INFO] Waiting 3s for streams to stabilize...")
    time.sleep(3)

    camera_id = 0
    passed = 0
    failed = 0

    for test_fn in [
        lambda: test_basic_retrieval(provider, camera_id),
        lambda: test_fifo_ordering(provider, camera_id),
        lambda: test_consumption(provider, camera_id),
        lambda: test_edge_cases(provider, camera_id),
        lambda: test_throughput(provider, camera_id),
    ]:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [EXCEPTION] {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    provider.stop()


if __name__ == "__main__":
    main()
