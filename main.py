import cv2
import numpy as np

# ============================================================
# CUDA INITIALIZATION
# ============================================================
# CUDA = NVIDIA's system for running code on the GPU
# Must initialize CUDA before TensorRT can use the GPU

import pycuda.driver as cuda                # PyCUDA = Python bindings for CUDA
cuda.init()                                 # Initialize the CUDA driver
device = cuda.Device(0)                     # Get GPU device 0 (first GPU)
ctx = device.make_context()                 # Create a CUDA context (like "logging into" the GPU)

import tensorrt as trt                      # TensorRT = NVIDIA's inference optimizer
import atexit                               # Built-in module to register cleanup functions

def cleanup():                              # Function to clean up when program exits
    ctx.pop()                               # Remove/release the CUDA context
atexit.register(cleanup)                    # Register cleanup() to run automatically on exit

# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "videos/sg1.mkv"                              # Video source: 0 = webcam, or "file.mp4" for video file
ENGINE_PATH = "models/sgm16.engine"                 # Path to the TensorRT engine file (compiled neural network)
INPUT_W, INPUT_H = 640, 640                 # Neural network input size (fixed, cannot change at runtime)
CONF_THRESHOLD = 0.50                     # Minimum confidence to keep a detection (0.0 to 1.0)
NMS_THRESHOLD = 0.45                        # IoU threshold for non-max suppression (removes duplicate boxes)
CLASS_NAMES = [                             # Class names
   'boots','helmet','no boots','no helmet','no vest','person','vest'
]

# ============================================================
# LOAD TENSORRT ENGINE
# ============================================================
# The engine file contains the neural network optimized for your specific GPU

TRT_LOGGER = trt.Logger(trt.Logger.WARNING) # Create a logger (prints warnings/errors from TensorRT)
with open(ENGINE_PATH, "rb") as f:          # Open engine file in binary read mode
    runtime = trt.Runtime(TRT_LOGGER)       # Create TensorRT runtime (manages engine execution)
    engine = runtime.deserialize_cuda_engine(f.read())  # Load engine from file into GPU memory

context = engine.create_execution_context() # Create execution context (holds state for running inference)

# ============================================================
# ALLOCATE GPU MEMORY BUFFERS
# ============================================================
# We need memory on both CPU (host) and GPU (device) for input/output data

inputs, outputs, bindings = [], [], []      # Lists to store input/output buffer info
stream = cuda.Stream()                      # CUDA stream = a queue of GPU operations (allows async execution)

def get_tensor_size(engine, name):          # Helper function: calculate total elements in a tensor
    shape = engine.get_tensor_shape(name)   # Get shape like (1, 3, 640, 640)
    size = 1                                # Start with 1
    for dim in shape:                       # Multiply all dimensions together
        size *= dim                         # e.g., 1 * 3 * 640 * 640 = 1,228,800
    return size                             # Return total number of elements

for i in range(engine.num_io_tensors):      # Loop through all input/output tensors
    name = engine.get_tensor_name(i)        # Get tensor name (e.g., "images" or "output0")
    dtype = trt.nptype(engine.get_tensor_dtype(name))  # Get data type (e.g., float32)
    size = get_tensor_size(engine, name)    # Get total number of elements
    host_mem = cuda.pagelocked_empty(size, dtype)      # Allocate CPU memory (page-locked = faster transfers)
    dev_mem = cuda.mem_alloc(host_mem.nbytes)          # Allocate GPU memory (same size in bytes)
    bindings.append(int(dev_mem))           # Store GPU memory address
    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:  # Check if this is an input tensor
        inputs.append({"host": host_mem, "device": dev_mem, "name": name})   # Store in inputs list
    else:                                   # Otherwise it's an output tensor
        outputs.append({"host": host_mem, "device": dev_mem, "name": name})  # Store in outputs list

# ============================================================
# PREPROCESSING FUNCTION (with letterboxing like Ultralytics)
# ============================================================
# Converts a video frame into the format the neural network expects
# Uses letterboxing to preserve aspect ratio (critical for detection quality)

def letterbox(frame, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize image with letterboxing (preserves aspect ratio, pads with gray)"""
    h, w = frame.shape[:2]

    # Calculate scale to fit within new_shape while preserving aspect ratio
    scale = min(new_shape[0] / h, new_shape[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize with preserved aspect ratio
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create padded image filled with gray (114 is YOLO's default pad color)
    padded = np.full((new_shape[0], new_shape[1], 3), color, dtype=np.uint8)

    # Calculate padding offsets to center the image
    pad_top = (new_shape[0] - new_h) // 2
    pad_left = (new_shape[1] - new_w) // 2

    # Place resized image in center of padded image
    padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    return padded, scale, pad_left, pad_top

def preprocess(frame):
    img, _, _, _ = letterbox(frame, (INPUT_H, INPUT_W))  # Letterbox resize (preserves aspect ratio)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)           # Convert BGR (OpenCV format) to RGB (neural network format)
    img = img.astype(np.float32) / 255.0                 # Convert pixels from 0-255 integers to 0.0-1.0 floats
    img = np.transpose(img, (2, 0, 1))                   # Reorder from (H, W, C) to (C, H, W) - channels first
    return img.ravel()                                   # Flatten to 1D array for copying to GPU

# ============================================================
# POSTPROCESSING FUNCTION
# ============================================================
# Converts raw neural network output into bounding boxes drawn on the frame
#
# YOLO11 output shape: (1, 84, 8400)
#   - 84 = 4 box coordinates + 80 class scores
#   - 8400 = number of possible detections (grid cells × anchors)

def postprocess(frame, output):
    frame_h, frame_w = frame.shape[:2]              # Get original frame dimensions (height, width)

    # Calculate letterbox parameters (must match preprocessing)
    scale = min(INPUT_H / frame_h, INPUT_W / frame_w)
    pad_x = (INPUT_W - frame_w * scale) / 2         # Horizontal padding
    pad_y = (INPUT_H - frame_h * scale) / 2         # Vertical padding

    num_classes = len(CLASS_NAMES)                   # Number of classes (80 for COCO)
    num_outputs = 4 + num_classes                    # 4 box coords + class scores
    preds = np.array(output).reshape(num_outputs, 8400).T  # Reshape to (8400, 84) - each row is one detection

    boxes_xywh = preds[:, :4]                       # First 4 columns = box coords (center_x, center_y, width, height)
    class_scores = preds[:, 4:]                     # Remaining 80 columns = confidence score for each class

    class_ids = np.argmax(class_scores, axis=1)     # For each detection, find which class has highest score
    confidences = np.max(class_scores, axis=1)      # Get that highest score as the confidence

    mask = confidences > CONF_THRESHOLD             # Create True/False mask: True where confidence > threshold
    boxes_xywh = boxes_xywh[mask]                   # Keep only boxes with high confidence
    confidences = confidences[mask]                 # Keep only corresponding confidences
    class_ids = class_ids[mask]                     # Keep only corresponding class IDs

    if len(boxes_xywh) == 0:                        # If no detections passed the threshold
        return frame                                # Return frame unchanged

    boxes = []                                      # List to store converted box coordinates
    for cx, cy, bw, bh in boxes_xywh:               # For each box: center_x, center_y, box_width, box_height
        # Remove padding offset and scale back to original frame size
        x1 = int((cx - bw / 2 - pad_x) / scale)
        y1 = int((cy - bh / 2 - pad_y) / scale)
        w = int(bw / scale)
        h = int(bh / scale)
        # Clamp to frame boundaries
        x1 = max(0, min(x1, frame_w))
        y1 = max(0, min(y1, frame_h))
        boxes.append([x1, y1, w, h])

    # Non-Maximum Suppression: removes overlapping boxes, keeps only the best one
    idxs = cv2.dnn.NMSBoxes(boxes, confidences.tolist(), CONF_THRESHOLD, NMS_THRESHOLD)

    for i in idxs.flatten() if len(idxs) > 0 else []:  # Loop through indices of boxes that survived NMS
        x, y, w, h = boxes[i]                       # Get box coordinates
        cls_id = class_ids[i]                       # Get class ID (e.g., 0 = person)
        conf = confidences[i]                       # Get confidence score
        label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)  # Get class name
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green rectangle
        cv2.putText(frame, f"{label} {conf:.2f}", (x, y - 10),       # Draw label text above box
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame                                    # Return frame with boxes drawn on it

# ============================================================
# MAIN VIDEO LOOP
# ============================================================

vid = cv2.VideoCapture(VIDEO_PATH)
if not vid.isOpened():
    exit()

while True:
    ret, frame = vid.read()
    if not ret:
        break

    img = preprocess(frame)
    np.copyto(inputs[0]["host"], img)

    cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)

    for inp in inputs:
        context.set_tensor_address(inp["name"], int(inp["device"]))
    for out in outputs:
        context.set_tensor_address(out["name"], int(out["device"]))

    context.execute_async_v3(stream_handle=stream.handle)

    cuda.memcpy_dtoh_async(outputs[0]["host"], outputs[0]["device"], stream)
    stream.synchronize()

    out_frame = postprocess(frame, outputs[0]["host"])
    cv2.imshow("TensorRT Detection", out_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
