import cv2                                  # OpenCV library for video/image processing
import numpy as np                          # NumPy for array math operations
import time                                 # For measuring FPS

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

VIDEO_PATH = "people.mp4"                              # Video source: 0 = webcam, or "file.mp4" for video file
ENGINE_PATH = "yolo11l.engine"                 # Path to the TensorRT engine file (compiled neural network)
INPUT_W, INPUT_H = 640, 640                 # Neural network input size (fixed, cannot change at runtime)
CONF_THRESHOLD = 0.25                       # Minimum confidence to keep a detection (0.0 to 1.0)
NMS_THRESHOLD = 0.45                        # IoU threshold for non-max suppression (removes duplicate boxes)
CLASS_NAMES = [                             # List of 80 COCO dataset class names
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
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
# PREPROCESSING FUNCTION
# ============================================================
# Converts a video frame into the format the neural network expects

def preprocess(frame):
    img = cv2.resize(frame, (INPUT_W, INPUT_H))     # Resize frame to 640x640 (network's required input size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # Convert BGR (OpenCV format) to RGB (neural network format)
    img = img.astype(np.float32) / 255.0            # Convert pixels from 0-255 integers to 0.0-1.0 floats
    img = np.transpose(img, (2, 0, 1))              # Reorder from (H, W, C) to (C, H, W) - channels first
    return img.ravel()                              # Flatten to 1D array for copying to GPU

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

    num_classes = len(CLASS_NAMES)                   # Number of classes (80 for COCO)
    num_outputs = 4 + num_classes                    # 4 box coords + class scores
    preds = np.array(output).reshape(num_outputs, 8400).T  # Reshape to (8400, 84) - each row is one detection

    boxes_xywh = preds[:, :4]                       # First 4 columns = box coords (center_x, center_y, width, height)
    class_scores = preds[:, 4:]                     # Remaining 80 columns = confidence score for each class

    class_ids = np.argmax(class_scores, axis=1)     # For each detection, find which class has highest score
    confidences = np.max(class_scores, axis=1)      # Get that highest score as the confidence

    mask = confidences > CONF_THRESHOLD             # Create True/False mask: True where confidence > 0.25
    boxes_xywh = boxes_xywh[mask]                   # Keep only boxes with high confidence
    confidences = confidences[mask]                 # Keep only corresponding confidences
    class_ids = class_ids[mask]                     # Keep only corresponding class IDs

    if len(boxes_xywh) == 0:                        # If no detections passed the threshold
        return frame                                # Return frame unchanged

    scale_x = frame_w / INPUT_W                     # Calculate horizontal scale factor (e.g., 1920/640 = 3.0)
    scale_y = frame_h / INPUT_H                     # Calculate vertical scale factor (e.g., 1080/640 = 1.6875)

    boxes = []                                      # List to store converted box coordinates
    for cx, cy, bw, bh in boxes_xywh:               # For each box: center_x, center_y, box_width, box_height
        x1 = int((cx - bw / 2) * scale_x)           # Left edge = center_x - half_width, scaled to frame size
        y1 = int((cy - bh / 2) * scale_y)           # Top edge = center_y - half_height, scaled to frame size
        w = int(bw * scale_x)                       # Box width scaled to frame size
        h = int(bh * scale_y)                       # Box height scaled to frame size
        boxes.append([x1, y1, w, h])                # Add to list in [x, y, width, height] format

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

vid = cv2.VideoCapture(VIDEO_PATH)                  # Open video source (webcam or file)
if not vid.isOpened():                              # Check if it opened successfully
    print("Error opening video!")
    exit()

while True:                                         # Loop forever until break
    start_time = time.time()                        # Record time at start of frame

    ret, frame = vid.read()                         # Read one frame. ret=True if successful, frame=the image
    if not ret:                                     # If read failed (end of video or error)
        break                                       # Exit the loop

    # STEP 1: Preprocess the frame for the neural network
    img = preprocess(frame)                         # Convert frame to format network expects
    np.copyto(inputs[0]["host"], img)               # Copy preprocessed image into CPU input buffer

    # STEP 2: Copy input from CPU to GPU
    cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)  # htod = Host To Device (CPU→GPU)

    # STEP 3: Tell TensorRT where our input/output buffers are in GPU memory
    for inp in inputs:                              # For each input tensor
        context.set_tensor_address(inp["name"], int(inp["device"]))  # Set its GPU memory address
    for out in outputs:                             # For each output tensor
        context.set_tensor_address(out["name"], int(out["device"]))  # Set its GPU memory address

    # STEP 4: Run the neural network (inference)
    context.execute_async_v3(stream_handle=stream.handle)  # Execute on GPU (async = non-blocking)

    # STEP 5: Copy results from GPU back to CPU
    cuda.memcpy_dtoh_async(outputs[0]["host"], outputs[0]["device"], stream)  # dtoh = Device To Host (GPU→CPU)
    stream.synchronize()                            # Wait for all GPU operations to complete

    # STEP 6: Process results and display
    out_frame = postprocess(frame, outputs[0]["host"])  # Convert raw output to boxes, draw on frame

    # Calculate and display FPS
    fps = 1 / (time.time() - start_time)            # FPS = 1 / seconds_per_frame
    cv2.putText(out_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("TensorRT Detection", out_frame)     # Display the frame in a window

    if cv2.waitKey(1) & 0xFF == ord('q'):           # If 'q' key pressed
        break                                       # Exit the loop

# ============================================================
# CLEANUP
# ============================================================

vid.release()                                       # Close the video source
cv2.destroyAllWindows()                             # Close all OpenCV windows
