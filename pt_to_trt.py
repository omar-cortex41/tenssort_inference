import subprocess
import sys
from sys import argv, exit

model = "models/sgm.pt"


# Convert PyTorch model to ONNX format (runs in separate process to avoid CUDA corruption)
def convert_pt_to_onnx(model_path):
    onnx_path = model_path.replace('.pt', '.onnx')
    # Run in separate process so YOLO/PyTorch doesn't corrupt CUDA for trtexec
    code = f'''
from ultralytics import YOLO
model = YOLO("{model_path}")
model.export(format="onnx")
'''
    subprocess.run([sys.executable, '-c', code], check=True)
    print(f"Model successfully exported to: {onnx_path}")
    return onnx_path


# Convert ONNX model to TensorRT engine (FP16)
def convert_onnx_to_engine_16(onnx_path):
    engine_path = onnx_path.replace('.onnx', '_fp16.engine')
    command = f"trtexec --onnx={onnx_path} --saveEngine={engine_path} --fp16"
    subprocess.run(command, shell=True, check=True)
    print(f"Engine successfully saved to: {engine_path}")


# Convert ONNX model to TensorRT engine (FP32)
def convert_onnx_to_engine_32(onnx_path):
    engine_path = onnx_path.replace('.onnx', '_fp32.engine')
    command = f"trtexec --onnx={onnx_path} --saveEngine={engine_path}"
    subprocess.run(command, shell=True, check=True)
    print(f"Engine successfully saved to: {engine_path}")



if len(argv) != 2:
    print("Usage: python pt_to_trt.py [--fp16|fp32]")
    print("Example: python pt_to_trt.py --fp16")
    exit(1)

else:

    if argv[1] == "--fp16":
        onnx = convert_pt_to_onnx(model)
        convert_onnx_to_engine_16(onnx)

    elif argv[1] == "--fp32":
        onnx = convert_pt_to_onnx(model)
        convert_onnx_to_engine_32(onnx)
    
    else:
        print("Usage: python pt_to_trt.py [--fp16|fp32]")
        print("Example: python pt_to_trt.py --fp16")
        exit(1)


