#!/usr/bin/env python3
"""TensorRT YOLO Detector - Process video and save output"""

import sys
import time
import yaml
import os
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

    # Setup output video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(os.path.dirname(video_path) or '.', f"{video_name}_inference.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Output: {output_path}")
    print("\n" + "=" * 60)
    print("Starting inference...")
    print("=" * 60 + "\n")

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
                label = f"{det.label} {det.confidence:.2f}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x, y - lh - 6), (x + lw, y), (0, 255, 0), -1)
                cv2.putText(frame, label, (x, y - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw FPS on frame
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Write frame
            writer.write(frame)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    # Cleanup
    cap.release()
    writer.release()

    # Final stats
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Processed frames: {frame_count}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average FPS: {frame_count / total_time:.2f}" if total_time > 0 else "N/A")
    print(f"Output saved: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

