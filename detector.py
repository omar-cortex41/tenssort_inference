#!/usr/bin/env python3
"""TensorRT YOLO Detector - Process video with real-time display"""

import sys
import time
import yaml
sys.path.insert(0, 'trt_detector/build')

import cv2
from trt_detector import DetectorService, ModelConfig

# Load configuration
with open('config/config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)


def main():
    print("=" * 60)
    print("TensorRT YOLO Detector")
    print("=" * 60)

    # Initialize detector
    print("\nInitializing detector...")
    detector = DetectorService()

    engine_path = cfg['model']['engine_path']
    print(f"Loading model from {engine_path}...")

    config = ModelConfig(
        engine_path,
        cfg['class_names'],
        conf_threshold=cfg['model']['conf_threshold'],
        nms_threshold=cfg['model']['nms_threshold']
    )

    if not detector.load_model(config):
        print("Failed to load model!")
        return

    print("Model loaded successfully!")

    # Open video
    video_path = cfg['video']['path']
    print(f"\nOpening video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    print("\n" + "=" * 60)
    print("Starting inference... Press 'q' to quit")
    print("=" * 60 + "\n")

    # Create window for display
    window_name = "TensorRT YOLO Detector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Set window size to 1280x720 (or maintain aspect ratio of video)
    display_width = 1280
    display_height = int(display_width * height / width)
    cv2.resizeWindow(window_name, display_width, display_height)

    frame_count = 0
    total_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection
            start = time.time()
            detections = detector.detect(frame)
            elapsed = time.time() - start
            total_time += elapsed
            frame_count += 1

            # Calculate FPS
            current_fps = 1.0 / elapsed if elapsed > 0 else 0
            avg_fps = frame_count / total_time if total_time > 0 else 0

            # Log detections
            det_summary = ", ".join([f"{d.label}:{d.confidence:.2f}" for d in detections[:5]])
            if len(detections) > 5:
                det_summary += f", ... (+{len(detections)-5} more)"

            print(f"Frame {frame_count:5d}/{total_frames} | "
                  f"FPS: {current_fps:5.1f} (avg: {avg_fps:5.1f}) | "
                  f"Detections: {len(detections):3d} | {det_summary}")

            # Draw detections on frame
            for det in detections:
                x, y, w, h = det.x, det.y, det.width, det.height
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Include class ID in label and make it bigger
                label = f"[{det.class_id}] {det.label} {det.confidence:.2f}"
                font_scale = 0.8
                thickness = 2
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(frame, (x, y - lh - 8), (x + lw, y), (0, 255, 0), -1)
                cv2.putText(frame, label, (x, y - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

            # Draw FPS on frame
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display frame
            cv2.imshow(window_name, frame)

            # Check for quit key ('q' or ESC)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n\nStopped by user")
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Final stats
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Processed frames: {frame_count}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average FPS: {frame_count / total_time:.2f}" if total_time > 0 else "N/A")
    print("=" * 60)


if __name__ == "__main__":
    main()

