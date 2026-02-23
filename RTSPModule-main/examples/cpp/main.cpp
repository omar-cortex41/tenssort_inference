/**
 * @file main.cpp
 * @brief Example demonstrating RTSPCore library usage in C++
 *
 * This example shows how to:
 *   1. Load configuration from YAML
 *   2. Start RTSP streams with automatic reconnection
 *   3. Retrieve frames (CPU mode via CpuFrame)
 *   4. Access stream statistics
 *   5. Use batch frame retrieval for multiple cameras
 *
 * Build Instructions:
 *   cmake -B build -S .
 *   cmake --build build
 *
 * Run:
 *   ./build/rtsp_example ../../configs/config.yaml
 * 
 *
 */

#include <rtspmodule/rtsp_client.h>
#include <rtspmodule/rtsp_structs.h>
#include <rtspmodule/batch_types.h>

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <thread>

// -----------------------------------------------------------------------------
// Global signal handler for graceful shutdown
// -----------------------------------------------------------------------------
namespace {
    volatile std::sig_atomic_t g_running = 1;
    
    void signal_handler(int signal) {
        std::cout << "\n[INFO] Caught signal " << signal << ", shutting down...\n";
        g_running = 0;
    }
}

// -----------------------------------------------------------------------------
// Helper: Print frame statistics
// -----------------------------------------------------------------------------
void print_stats(const RtspClient& client, int camera_id) {
    const auto stats = client.getStats(camera_id);
    
    std::cout << "  [Camera " << camera_id << "] "
              << "FPS: " << std::fixed << std::setprecision(1) << stats.current_fps
              << " | Frames: " << stats.frames_decoded
              << " | Drops: " << stats.frames_dropped_queue
              << " | Queue: " << stats.queue_depth << "/" << stats.queue_max_depth
              << " | Resolution: " << stats.source_width << "x" << stats.source_height
              << "\n";
}
 
// -----------------------------------------------------------------------------
// Example 1: Single-frame retrieval (CPU buffer mode)
// -----------------------------------------------------------------------------
void example_single_frame(RtspClient& client) {
    std::cout << "\n=== Example 1: Single Frame Retrieval ===\n";
    
    const int stream_count = client.getStreamCount();
    if (stream_count == 0) {
        std::cerr << "[WARN] No streams configured\n";
        return;
    }
    
    constexpr int TIMEOUT_MS = 1000;  // 1 second timeout
    
    // Attempt to get a frame from each camera
    for (int cam = 0; cam < stream_count && g_running; ++cam) {
        const CpuFrame frame = client.getCpuFrame(cam, TIMEOUT_MS);
        
        if (frame.valid) {
            std::cout << "  [Camera " << cam << "] "
                      << "Frame #" << frame.frame_id
                      << " | " << frame.width << "x" << frame.height
                      << " | Format: " << frame.format
                      << " | Size: " << frame.data_size << " bytes\n";
        } else {
            std::cout << "  [Camera " << cam << "] No frame available (timeout)\n";
        }
    }
}

// -----------------------------------------------------------------------------
// Example 2: Batch frame retrieval (efficient multi-camera access)
// -----------------------------------------------------------------------------
void example_batch_retrieval(RtspClient& client) {
    std::cout << "\n=== Example 2: Batch Frame Retrieval ===\n";
    
    const int stream_count = client.getStreamCount();
    if (stream_count == 0) {
        std::cerr << "[WARN] No streams configured\n";
        return;
    }
    
    // Build list of all camera IDs
    BatchConfig config;
    for (int i = 0; i < stream_count; ++i) {
        config.camera_ids.push_back(i);
    }
    config.timeout_ms = 100;  // Short timeout for demonstration
    
    const FrameBatch batch = client.getBatchedFrames(config);
    
    std::cout << "  Batch size: " << batch.batch_size
              << " | Valid: " << batch.valid_count
              << " | Resolution: " << batch.width << "x" << batch.height
              << " | Format: " << batch.format
              << " | Stride: " << batch.frame_stride << " bytes\n";
    
    // Iterate through batch results
    for (size_t i = 0; i < batch.batch_size; ++i) {
        const auto& meta = batch.metadata[i];
        const char* status = meta.valid ? "OK" : "MISS";
        std::cout << "    [" << i << "] Camera " << meta.camera_id
                  << " Frame #" << meta.frame_id
                  << " -> " << status << "\n";
    }
}

// -----------------------------------------------------------------------------
// Example 3: Continuous monitoring loop with statistics
// -----------------------------------------------------------------------------
void example_monitoring_loop(RtspClient& client, int duration_sec) {
    std::cout << "\n=== Example 3: Monitoring Loop (" << duration_sec << "s) ===\n";
    
    const auto start = std::chrono::steady_clock::now();
    const auto end = start + std::chrono::seconds(duration_sec);
    
    while (g_running && std::chrono::steady_clock::now() < end) {
        std::cout << "\n--- Statistics Snapshot ---\n";
        
        for (int cam = 0; cam < client.getStreamCount(); ++cam) {
            print_stats(client, cam);
        }
        
        // Also show CPU buffer info if enabled
        if (client.isCpuBufferEnabled()) {
            for (int cam = 0; cam < client.getStreamCount(); ++cam) {
                const auto info = client.getCpuBufferInfo(cam);
                std::cout << "    Buffer[" << cam << "]: "
                          << info.buffer_count << "/" << info.buffer_capacity
                          << " frames (" << info.memory_usage_bytes / 1024 << " KB)\n";
            }
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

// -----------------------------------------------------------------------------
// Main entry point
// -----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::cout << "=================================================\n"
              << "       RTSPCore C++ Example Application\n"
              << "=================================================\n";
    
    // Parse command-line arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.yaml>\n";
        std::cerr << "Example: " << argv[0] << " ../../configs/config.yaml\n";
        return EXIT_FAILURE;
    }
    
    const std::string config_path = argv[1];
    
    // Install signal handlers for graceful shutdown
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    
    // -------------------------------------------------------------------------
    // Initialize RTSP Client
    // -------------------------------------------------------------------------
    RtspClient client;
    
    // Optional: Set log output path
    client.setLogPath("./logs");
    
    std::cout << "\n[INFO] Loading configuration: " << config_path << "\n";
    
    if (!client.loadConfig(config_path)) {
        std::cerr << "[ERROR] Failed to load config: " << config_path << "\n";
        return EXIT_FAILURE;
    }
    
    std::cout << "[INFO] Configured " << client.getStreamCount() << " stream(s)\n";
    std::cout << "[INFO] CPU buffer mode: " 
              << (client.isCpuBufferEnabled() ? "ENABLED" : "DISABLED") << "\n";
    std::cout << "[INFO] GPU available: " 
              << (client.isGpuAvailable() ? "YES" : "NO") << "\n";
    
    // -------------------------------------------------------------------------
    // Start Streaming
    // -------------------------------------------------------------------------
    std::cout << "\n[INFO] Starting streams...\n";
    
    if (!client.start()) {
        std::cerr << "[ERROR] Failed to start RTSP client\n";
        return EXIT_FAILURE;
    }
    
    // Give streams time to connect and buffer initial frames
    std::cout << "[INFO] Waiting for streams to stabilize...\n";
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // -------------------------------------------------------------------------
    // Run Examples
    // -------------------------------------------------------------------------
    
    // Example 1: Get single frames from each camera
    example_single_frame(client);
    
    // Example 2: Batch retrieval (more efficient for multi-stream)
    example_batch_retrieval(client);
    
    // Example 3: Continuous monitoring (5 seconds)
    example_monitoring_loop(client, 5);
    
    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    std::cout << "\n[INFO] Stopping streams...\n";
    client.stop();
    
    std::cout << "[INFO] Application exited cleanly.\n";
    return EXIT_SUCCESS;
}
