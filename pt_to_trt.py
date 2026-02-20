import subprocess
import sys
import argparse

MODEL_PATH = "models/yolo26m.pt"


def convert_pt_to_onnx(model_path: str, dynamic: bool = False) -> str:
    """Convert PyTorch model to ONNX format.

    Runs in separate process to avoid CUDA corruption from PyTorch/YOLO.

    Args:
        model_path: Path to .pt model
        dynamic: If True, export with dynamic batch axis

    Returns:
        Path to exported ONNX file
    """
    onnx_path = model_path.replace('.pt', '.onnx')

    if dynamic:
        # Export with dynamic batch dimension
        code = f'''
from ultralytics import YOLO
model = YOLO("{model_path}")
model.export(format="onnx", dynamic=True)
'''
    else:
        code = f'''
from ultralytics import YOLO
model = YOLO("{model_path}")
model.export(format="onnx")
'''

    subprocess.run([sys.executable, '-c', code], check=True)
    print(f"ONNX exported to: {onnx_path}")
    return onnx_path


def convert_onnx_to_engine(onnx_path: str, fp16: bool = True, max_batch: int = None) -> str:
    """Convert ONNX model to TensorRT engine.

    Args:
        onnx_path: Path to ONNX model
        fp16: Use FP16 precision (faster, slightly less accurate)
        max_batch: If set, build dynamic batch engine with range [1, max_batch]

    Returns:
        Path to TensorRT engine
    """
    precision = "fp16" if fp16 else "fp32"

    if max_batch:
        # Dynamic batch engine
        engine_path = onnx_path.replace('.onnx', f'_{precision}_dynamic.engine')

        # Calculate optimal batch size (half of max, minimum 1)
        opt_batch = max(1, max_batch // 2)

        # Build shape strings for dynamic dimensions
        # YOLO input shape: [batch, 3, 640, 640]
        min_shape = f"images:1x3x640x640"
        opt_shape = f"images:{opt_batch}x3x640x640"
        max_shape = f"images:{max_batch}x3x640x640"

        command = [
            "trtexec",
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            f"--minShapes={min_shape}",
            f"--optShapes={opt_shape}",
            f"--maxShapes={max_shape}",
        ]

        if fp16:
            command.append("--fp16")

        print(f"Building dynamic batch engine (batch: 1-{max_batch}, optimal: {opt_batch})")
    else:
        # Static batch=1 engine
        engine_path = onnx_path.replace('.onnx', f'_{precision}.engine')
        command = [
            "trtexec",
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
        ]

        if fp16:
            command.append("--fp16")

        print(f"Building static batch=1 engine")

    subprocess.run(command, check=True)
    print(f"Engine saved to: {engine_path}")
    return engine_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch YOLO model to TensorRT engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pt_to_trt.py --fp16                    # Static batch=1, FP16
  python pt_to_trt.py --fp32                    # Static batch=1, FP32
  python pt_to_trt.py --fp16 --dynamic 8        # Dynamic batch 1-8, FP16
  python pt_to_trt.py --fp16 --dynamic 16       # Dynamic batch 1-16, FP16
"""
    )

    precision = parser.add_mutually_exclusive_group(required=True)
    precision.add_argument("--fp16", action="store_true", help="Use FP16 precision (faster)")
    precision.add_argument("--fp32", action="store_true", help="Use FP32 precision (more accurate)")

    parser.add_argument(
        "--dynamic",
        type=int,
        metavar="MAX_BATCH",
        help="Build dynamic batch engine with range [1, MAX_BATCH]. "
             "Optimal batch is set to MAX_BATCH/2."
    )

    args = parser.parse_args()

    # Step 1: Convert PT to ONNX
    print("=" * 60)
    print("Step 1: Converting PyTorch to ONNX")
    print("=" * 60)
    onnx_path = convert_pt_to_onnx(MODEL_PATH, dynamic=args.dynamic is not None)

    # Step 2: Convert ONNX to TensorRT
    print()
    print("=" * 60)
    print("Step 2: Converting ONNX to TensorRT")
    print("=" * 60)
    engine_path = convert_onnx_to_engine(onnx_path, fp16=args.fp16, max_batch=args.dynamic)

    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"Engine: {engine_path}")
    if args.dynamic:
        print(f"Batch range: 1 to {args.dynamic}")
    else:
        print("Batch: 1 (static)")


if __name__ == "__main__":
    main()


