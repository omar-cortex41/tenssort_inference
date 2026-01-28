#!/usr/bin/env python3
"""TensorRT YOLO Detector - reads config from config/config.yaml"""

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
    print("Creating DetectorService...")
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
    cap = cv2.VideoCapture(cfg['video']['path'])
    if not cap.isOpened():
        print(f"Failed to open video: {VIDEO_PATH}")
        return
    
    frame_count = 0
    total_time = 0
    
    print("\nRunning inference... Press 'q' to quit")
    
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
        
        # Draw all detections with bounding boxes and labels
        for det in detections:
            x, y, w, h = det.x, det.y, det.width, det.height
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{det.label} {det.confidence:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y - lh - 6), (x + lw, y), (0, 255, 0), -1)
            cv2.putText(frame, label, (x, y - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show FPS
        fps = 1.0 / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("C++ TRT Detector", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print stats
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"\n{'='*50}")
    print(f"Total frames: {frame_count}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()

