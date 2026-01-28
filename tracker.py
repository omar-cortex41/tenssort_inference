#!/usr/bin/env python3
"""
TensorRT YOLO Detector + ByteTrack Tracker
Reads config from config/config.yaml

Pipeline:
  Video Frame -> TRT Detector (C++) -> Detections -> ByteTrack (C++) -> Tracks with IDs
"""

import sys                                      # System utilities
import time                                     # For measuring FPS
import yaml                                     # For reading config file
sys.path.insert(0, 'trt_detector/build')        # Add C++ detector module to path

import cv2                                      # OpenCV for video/image processing
from trt_detector import DetectorService, ModelConfig  # C++ TensorRT detector
import bytetrack_cpp as bt                      # C++ ByteTrack tracker

# ============================================================================
# CONFIGURATION
# ============================================================================
with open('config/config.yaml', 'r') as f:      # Open config file
    cfg = yaml.safe_load(f)                     # Parse YAML into dictionary

# Color palette for visualizing different track IDs (BGR format)
COLORS = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 255),  # Purple
    (255, 128, 0),  # Light blue
    (0, 128, 255),  # Orange
    (128, 255, 0),  # Light green
    (255, 0, 128),  # Pink
    (0, 255, 128),  # Teal
]

def get_color(track_id):
    """Get a consistent color for a track ID (cycles through COLORS list)"""
    return COLORS[track_id % len(COLORS)]       # Modulo ensures we stay in bounds


# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    # ========================================================================
    # STEP 1: SETUP DETECTOR (C++ TensorRT)
    # ========================================================================
    print("Creating DetectorService...")
    detector = DetectorService()                # Create C++ detector instance

    engine_path = cfg['model']['engine_path']   # Path to TensorRT engine file
    print(f"Loading model from {engine_path}...")

    det_config = ModelConfig(                   # Configure the detector
        engine_path,                            # TensorRT engine file (.engine)
        cfg['class_names'],                     # List of class names ["person", "car", ...]
        conf_threshold=cfg['model']['conf_threshold'],  # Min confidence to keep detection (e.g., 0.5)
        nms_threshold=cfg['model']['nms_threshold']     # Non-max suppression threshold (e.g., 0.45)
    )

    if not detector.load_model(det_config):     # Load model into GPU memory
        print("Failed to load model!")
        return
    print("Model loaded successfully!")

    # ========================================================================
    # STEP 2: OPEN VIDEO SOURCE
    # ========================================================================
    cap = cv2.VideoCapture(cfg['video']['path'])  # Open video file or camera
    if not cap.isOpened():
        print(f"Failed to open video: {cfg['video']['path']}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30       # Frames per second (default 30 if unknown)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # Frame width in pixels
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Frame height in pixels

    # Create video writer for output
    out = cv2.VideoWriter(
        'output.mp4',                           # Output filename
        cv2.VideoWriter_fourcc(*'mp4v'),        # Codec (MP4V)
        fps,                                    # Frame rate
        (width, height)                         # Frame size
    )
    print(f"Saving output to output.mp4 ({width}x{height} @ {fps:.1f} fps)")

    # ========================================================================
    # STEP 3: SETUP TRACKER (C++ ByteTrack)
    # ========================================================================
    track_cfg = cfg.get('tracking', {})         # Get tracking section from config

    # Convert class names to numeric IDs for filtering
    # e.g., ["person", "car"] -> {0, 2} (based on position in class_names list)
    track_classes_cfg = track_cfg.get('track_classes', [])  # Classes to track from config
    track_class_ids = set()                     # Set of class IDs to track
    for cls in track_classes_cfg:
        if isinstance(cls, int):                # If already an integer ID
            track_class_ids.add(cls)
        elif isinstance(cls, str) and cls in cfg['class_names']:  # If class name string
            track_class_ids.add(cfg['class_names'].index(cls))    # Convert to ID

    # Create C++ ByteTrack tracker (frame_rate from video, rest from config)
    tracker_config = bt.TrackerConfig(
        track_cfg['track_thresh'],              # Min confidence to start tracking
        track_cfg['match_thresh'],              # IoU threshold for matching detections to tracks
        track_cfg['track_buffer'],              # Frames to keep lost tracks before removing
        int(fps)                                # Frame rate (auto-detected from video)
    )
    tracker = bt.BYTETracker(tracker_config)    # Create tracker instance

    if track_class_ids:
        print(f"Tracker initialized - tracking: {[cfg['class_names'][i] for i in track_class_ids]}")
    else:
        print("Tracker initialized - tracking ALL classes")

    frame_count = 0                             # Counter for processed frames
    total_time = 0                              # Accumulated processing time

    print("\nRunning detection + tracking... Press 'q' to quit")

    # ========================================================================
    # STEP 4: MAIN PROCESSING LOOP
    # ========================================================================
    while True:
        ret, frame = cap.read()                 # Read next frame from video
        if not ret:                             # End of video or error
            break

        start = time.time()                     # Start timing

        # --------------------------------------------------------------------
        # DETECTION: Run YOLO on frame (detects ALL classes)
        # Returns list of Detection objects with: x, y, width, height, confidence, class_id, label
        # --------------------------------------------------------------------
        detections = detector.detect(frame)

        # --------------------------------------------------------------------
        # FILTER: Keep only detections of classes we want to track
        # e.g., if track_class_ids = {0}, only keep "person" detections
        # --------------------------------------------------------------------
        if track_class_ids:
            dets_to_track = [d for d in detections if d.class_id in track_class_ids]
        else:
            dets_to_track = detections          # Track all classes

        # --------------------------------------------------------------------
        # TRACKING: Convert detections to C++ format and update tracker
        # ByteTrack associates detections across frames and assigns persistent IDs
        # --------------------------------------------------------------------
        cpp_dets = [
            bt.Detection(                       # Create C++ Detection object
                d.x, d.y,                       # Top-left corner (x, y)
                d.width, d.height,              # Bounding box size
                d.confidence,                   # Detection confidence (0-1)
                d.class_id,                     # Class ID (0, 1, 2, ...)
                d.label                         # Class name string
            )
            for d in dets_to_track
        ]
        tracks = tracker.update(cpp_dets)       # Returns list of tracks with persistent IDs

        elapsed = time.time() - start           # Calculate processing time
        total_time += elapsed
        frame_count += 1

        # --------------------------------------------------------------------
        # VISUALIZATION: Draw results on frame
        # --------------------------------------------------------------------

        # Get tracked bounding boxes to avoid drawing them twice
        tracked_boxes = set()
        for track in tracks:
            x, y, w, h = [int(v) for v in track.tlwh]
            tracked_boxes.add((x, y, w, h))

        # Draw ALL detections (gray boxes with class label)
        for det in detections:
            x, y, w, h = det.x, det.y, det.width, det.height
            # Skip if this detection is being tracked (will draw with ID below)
            if (x, y, w, h) in tracked_boxes:
                continue
            cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 128, 128), 2)
            label = f"{det.label} {det.confidence:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y - lh - 6), (x + lw, y), (128, 128, 128), -1)
            cv2.putText(frame, label, (x, y - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw tracked objects (colored boxes with ID)
        for track in tracks:
            x, y, w, h = [int(v) for v in track.tlwh]
            color = get_color(track.track_id)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            class_name = cfg['class_names'][track.class_id] if track.class_id < len(cfg['class_names']) else str(track.class_id)
            label = f"ID:{track.track_id} {class_name} {track.score:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y - lh - 6), (x + lw, y), color, -1)
            cv2.putText(frame, label, (x, y - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw stats overlay (top-left corner)
        fps_val = 1.0 / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps_val:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Tracks: {len(tracks)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Detections: {len(detections)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # --------------------------------------------------------------------
        # OUTPUT: Write frame and display
        # --------------------------------------------------------------------
        out.write(frame)                        # Write frame to output video
        cv2.imshow("TRT Detector + ByteTrack", frame)  # Display in window

        if cv2.waitKey(1) & 0xFF == ord('q'):   # Check for 'q' key to quit
            break

    # ========================================================================
    # STEP 5: CLEANUP
    # ========================================================================
    cap.release()                               # Release video capture
    out.release()                               # Release video writer
    cv2.destroyAllWindows()                     # Close all OpenCV windows

    # Print final statistics
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"\n{'='*50}")
    print(f"Total frames: {frame_count}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
