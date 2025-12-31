#!/usr/bin/env python3
"""Test script for the C++ TRT detector module."""

import sys
import time
sys.path.insert(0, 'trt_detector/build')

import cv2
import numpy as np
from trt_detector import DetectorService, ModelConfig, Detection

# Configuration
VIDEO_PATH = "videos/sg1.mkv"
ENGINE_PATH = "models/sgm32.engine"
CLASS_NAMES = ['boots', 'helmet', 'no boots', 'no helmet', 'no vest', 'person', 'vest']

def main():
    print("Creating DetectorService...")
    detector = DetectorService()
    
    print(f"Loading model from {ENGINE_PATH}...")
    config = ModelConfig(
        ENGINE_PATH,
        CLASS_NAMES,
        conf_threshold=0.6,
        nms_threshold=0.45
    )
    
    if not detector.load_model(config):
        print("Failed to load model!")
        return

    print("Model loaded successfully!")
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
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
        
        # Draw detections
        for det in detections:
            x, y, w, h = det.x, det.y, det.width, det.height
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{det.label}: {det.confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
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

