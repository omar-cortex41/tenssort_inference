from ultralytics import YOLO

# 1. Load the YOLO model (replace 'yolov8n.pt' with your model path, e.g., 'path/to/best.pt')
model = YOLO("yolo11l.pt") 

# 2. Export the model to ONNX format
# This creates a 'yolov8n.onnx' file in your current directory
onnx_path = model.export(format="onnx")

print(f"Model successfully exported to: {onnx_path}")
