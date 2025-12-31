# ============================================================
# YOLO Object Detection with Ultralytics
# ============================================================
# This script reads a video, detects objects in each frame
# using YOLO (You Only Look Once) neural network, and draws
# bounding boxes around detected objects.
# ============================================================

import cv2                      # OpenCV - for reading video and drawing on frames
from ultralytics import YOLO    # Ultralytics - the library that provides YOLO models
import time                     # For measuring FPS
import os                       # For extracting model name from path

# ------------------------------------------------------------
# SETUP: Load video and model
# ------------------------------------------------------------

VIDEO_PATH = "videos/sg1.mkv"
MODEL_PATH = "models/sgm.pt"

vid = cv2.VideoCapture(VIDEO_PATH)    # Open video file (use 0 for webcam)
model = YOLO(MODEL_PATH)              # Load YOLO model weights (yolo11n = small/fast version)
model.to("cuda")                      # Move model to GPU for faster inference

print(model.names)

# Confidence threshold - only show detections with confidence >= this value
conf = 0.7

# Check if video opened successfully
if not vid.isOpened():
    print("Cannot open stream")
    exit()

# Get video properties for saving output
frame_w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
vid_fps = vid.get(cv2.CAP_PROP_FPS)

# Create output video writer with model name
model_name = os.path.splitext(os.path.basename(MODEL_PATH))[0]  # e.g., "yolo11m"
output_path = f"{model_name}_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_vid = cv2.VideoWriter(output_path, fourcc, vid_fps, (frame_w, frame_h))

# FPS tracking
frame_times = []
start_total = time.time()

# ------------------------------------------------------------
# MAIN LOOP: Process each frame
# ------------------------------------------------------------

while True:
    frame_start = time.time()    # Record time at start of frame

    # Read one frame from the video
    # ret = True if frame was read successfully, False if video ended
    # frame = the actual image (numpy array of pixels)
    ret, frame = vid.read()

    # If no frame was read, video has ended
    if not ret:
        break

    # ------------------------------------------------------------
    # DETECTION: Run YOLO on the frame
    # ------------------------------------------------------------

    # Pass the frame to YOLO - it returns a list of detection results
    # The model automatically: resizes image, runs neural network, filters results
    results = model(frame, conf=conf)

    # ------------------------------------------------------------
    # DRAW: Loop through detections and draw boxes
    # ------------------------------------------------------------

    # results is a list (one item per image, we only have 1 image)
    for result in results:
        # result.boxes contains all detected objects in this frame
        boxes = result.boxes

        # Loop through each detected object
        for box in boxes:
            # box.xyxy[0] = coordinates as [x1, y1, x2, y2]
            #   x1, y1 = top-left corner
            #   x2, y2 = bottom-right corner
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers for drawing

            # box.cls[0] = class ID (a number like 0=person, 2=car, etc.)
            cls_id = int(box.cls[0])

            # model.names is a dictionary: {0: "person", 1: "bicycle", 2: "car", ...}
            # Look up the human-readable name for this class ID
            label = model.names[cls_id]

            # Draw a green rectangle around the detected object
            # (0, 255, 0) = green color in BGR format
            # 2 = line thickness in pixels
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw the label text above the box
            cv2.putText(
                frame,                      # Image to draw on
                label,                      # Text to display
                (x1, y1 - 10),              # Position (slightly above the box)
                cv2.FONT_HERSHEY_SIMPLEX,   # Font style
                0.5,                        # Font scale (size)
                (0, 255, 0),                # Text color (green)
                2                           # Text thickness
            )

    # ------------------------------------------------------------
    # DISPLAY: Show the frame with detections
    # ------------------------------------------------------------

    # Calculate and display FPS
    frame_time = time.time() - frame_start
    frame_times.append(frame_time)
    fps = 1 / frame_time if frame_time > 0 else 0
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    out_vid.write(frame)       # Write frame to output video
    cv2.imshow("YOLO Detection", frame)

    # Wait 1ms for a key press. If 'q' is pressed, exit the loop
    # waitKey(1) returns the ASCII code of the pressed key, or -1 if no key
    if cv2.waitKey(1) == ord('q'):
        break

# ------------------------------------------------------------
# CLEANUP & STATS
# ------------------------------------------------------------

total_time = time.time() - start_total
total_frames = len(frame_times)
avg_fps = total_frames / total_time if total_time > 0 else 0

print(f"\n{'='*50}")
print(f"Model: {MODEL_PATH}")
print(f"Total frames: {total_frames}")
print(f"Total time: {total_time:.2f}s")
print(f"Average FPS: {avg_fps:.2f}")
print(f"Output saved to: {output_path}")
print(f"{'='*50}\n")

vid.release()              # Close the video file
out_vid.release()          # Close output video
cv2.destroyAllWindows()    # Close all OpenCV windows
