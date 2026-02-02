#!/usr/bin/env python3
"""Ultralytics YOLO Inference - Process video and save output"""

import cv2
import time
import os
import yaml
from ultralytics import YOLO


def main():
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    print("=" * 60)
    print("Ultralytics YOLO Detector")
    print("=" * 60)

    # Get model path - use .pt version (derive from engine path)
    engine_path = cfg['model']['engine_path']
    model_path = engine_path.replace('.engine', '.pt').replace('_fp16', '').replace('_fp32', '')

    # Check if .pt file exists, otherwise ask user
    if not os.path.exists(model_path):
        # Try common variations
        base = os.path.splitext(engine_path)[0]
        for suffix in ['', '_fp16', '_fp32']:
            test_path = base.replace(suffix, '') + '.pt'
            if os.path.exists(test_path):
                model_path = test_path
                break
        else:
            print(f"Model not found: {model_path}")
            print("Please specify the .pt model path in config or place it in models/")
            return

    conf_threshold = cfg['model']['conf_threshold']
    video_path = cfg['video']['path']
    class_names = cfg['class_names']

    print(f"\nModel: {model_path}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"Classes: {class_names}")

    # Load model
    print("\nLoading model...")
    model = YOLO(model_path)
    model.to("cuda")
    print("Model loaded successfully!")

    # Open video
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
    output_path = os.path.join(os.path.dirname(video_path) or '.', f"{video_name}_yolo_inference.mp4")
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
            results = model(frame, conf=conf_threshold, verbose=False)
            elapsed = time.time() - start
            total_time += elapsed
            frame_count += 1

            # Calculate FPS
            current_fps = 1.0 / elapsed if elapsed > 0 else 0
            avg_fps = frame_count / total_time if total_time > 0 else 0

            # Process detections
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls_id]
                    detections.append((x1, y1, x2, y2, label, conf))

                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"{label} {conf:.2f}"
                    (lw, lh), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw, y1), (0, 255, 0), -1)
                    cv2.putText(frame, text, (x1, y1 - 4),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Log detections
            det_summary = ", ".join([f"{d[4]}:{d[5]:.2f}" for d in detections[:5]])
            if len(detections) > 5:
                det_summary += f", ... (+{len(detections)-5} more)"

            print(f"Frame {frame_count:5d}/{total_frames} | "
                  f"FPS: {current_fps:5.1f} (avg: {avg_fps:5.1f}) | "
                  f"Detections: {len(detections):3d} | {det_summary}")

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
