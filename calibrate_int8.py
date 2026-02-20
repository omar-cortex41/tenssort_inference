#!/usr/bin/env python3
"""
INT8 Calibration for YOLO models using COCO dataset.
Builds INT8 TensorRT engine with proper calibration for minimal accuracy loss.
"""

import os
import sys
import argparse
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path
import urllib.request
import zipfile
import shutil


class YOLOCalibrator(trt.IInt8EntropyCalibrator2):
    """INT8 Calibrator for YOLO models using COCO images."""

    def __init__(self, calibration_images, cache_file, batch_size=8, input_shape=(640, 640)):
        super().__init__()
        self.calibration_images = calibration_images
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.current_index = 0

        # Calculate input size
        self.input_size = 3 * input_shape[0] * input_shape[1] * np.dtype(np.float32).itemsize

        # Allocate GPU memory for a batch
        self.device_input = cuda.mem_alloc(self.batch_size * self.input_size)

        print(f"Calibrator initialized: {len(calibration_images)} images, batch={batch_size}")

    def preprocess_image(self, image_path):
        """Preprocess image for YOLO (same as inference preprocessing)."""
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Failed to read {image_path}")
            return None

        # Resize to input shape
        img = cv2.resize(img, self.input_shape)

        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))

        return img

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        """Get next batch of calibration data."""
        if self.current_index >= len(self.calibration_images):
            return None

        batch_images = []
        for i in range(self.batch_size):
            if self.current_index >= len(self.calibration_images):
                # Pad with last image if we run out
                if batch_images:
                    batch_images.append(batch_images[-1])
                break

            img_path = self.calibration_images[self.current_index]
            img = self.preprocess_image(img_path)

            if img is not None:
                batch_images.append(img)

            self.current_index += 1

        if not batch_images:
            return None

        # Stack into batch
        batch = np.stack(batch_images, axis=0).astype(np.float32)

        # Copy to GPU
        cuda.memcpy_htod(self.device_input, batch.ravel())

        print(f"Calibration batch {self.current_index // self.batch_size}/{(len(self.calibration_images) + self.batch_size - 1) // self.batch_size}", end='\r')

        return [int(self.device_input)]

    def read_calibration_cache(self):
        """Read calibration cache if it exists."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        """Write calibration cache."""
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"\nSaved calibration cache to {self.cache_file}")


def download_coco128():
    """Download COCO128 dataset for calibration."""
    dataset_dir = Path("coco128")

    if dataset_dir.exists():
        print(f"COCO128 dataset already exists at {dataset_dir}")
        return dataset_dir

    print("Downloading COCO128 dataset (~7MB)...")
    url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
    zip_path = "coco128.zip"

    try:
        urllib.request.urlretrieve(url, zip_path)
        print("Extracting...")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")

        os.remove(zip_path)
        print(f"COCO128 dataset extracted to {dataset_dir}")
        return dataset_dir

    except Exception as e:
        print(f"Error downloading COCO128: {e}")
        print("Please manually download from: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip")
        sys.exit(1)


def get_calibration_images(dataset_dir, num_images=128):
    """Get list of calibration images from COCO128."""
    images_dir = dataset_dir / "images" / "train2017"

    if not images_dir.exists():
        print(f"Error: Images directory not found at {images_dir}")
        sys.exit(1)

    image_files = list(images_dir.glob("*.jpg"))

    if not image_files:
        print(f"Error: No images found in {images_dir}")
        sys.exit(1)

    # Use specified number of images
    image_files = image_files[:num_images]

    print(f"Found {len(image_files)} calibration images")
    return image_files


def build_int8_engine(onnx_path, engine_path, calibration_images, cache_file,
                      batch_size=8, min_batch=1, max_batch=12):
    """Build INT8 TensorRT engine with calibration."""

    print("=" * 60)
    print("Building INT8 Engine with Calibration")
    print("=" * 60)
    print(f"ONNX: {onnx_path}")
    print(f"Engine: {engine_path}")
    print(f"Calibration images: {len(calibration_images)}")
    print(f"Dynamic batch: {min_batch}-{max_batch} (opt={batch_size})")
    print("=" * 60)
    print()

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    print("Parsing ONNX model...")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse ONNX file")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False

    # Create builder config
    config = builder.create_builder_config()

    # Set INT8 mode
    config.set_flag(trt.BuilderFlag.INT8)

    # Set calibrator
    calibrator = YOLOCalibrator(
        calibration_images,
        cache_file,
        batch_size=batch_size,
        input_shape=(640, 640)
    )
    config.int8_calibrator = calibrator

    # Set workspace
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB

    # Set dynamic batch profile
    input_tensor = network.get_input(0)
    input_name = input_tensor.name

    print(f"Input tensor: {input_name}, shape: {input_tensor.shape}")
    print(f"Setting dynamic batch profile: min={min_batch}, opt={batch_size}, max={max_batch}")

    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_name,
        (min_batch, 3, 640, 640),    # min
        (batch_size, 3, 640, 640),   # opt
        (max_batch, 3, 640, 640)     # max
    )
    config.add_optimization_profile(profile)

    # Build engine
    print("\nBuilding INT8 engine (this will take 5-10 minutes)...")
    print("Running calibration on COCO images...")

    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("ERROR: Failed to build engine")
        return False

    # Save engine
    print(f"\nSaving engine to {engine_path}...")
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    print("\n" + "=" * 60)
    print("✅ INT8 Engine Built Successfully!")
    print("=" * 60)
    print(f"Engine: {engine_path}")
    print(f"Calibration cache: {cache_file}")
    print(f"Dynamic batch: {min_batch}-{max_batch}")
    print("\nNext steps:")
    print(f"1. Update config.yaml: engine_path: {os.path.basename(engine_path)}")
    print("2. Run: python rtsp_detector.py")
    print("3. Compare FPS and detections with FP16")
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(description="Build INT8 TensorRT engine with COCO calibration")
    parser.add_argument("--onnx", type=str, default="../models/yolo26m.onnx",
                        help="Path to ONNX model")
    parser.add_argument("--output", type=str, default="../models/yolo26m_int8_dynamic.engine",
                        help="Output engine path")
    parser.add_argument("--cache", type=str, default="../models/yolo26m_calibration.cache",
                        help="Calibration cache path")
    parser.add_argument("--batch", type=int, default=8,
                        help="Optimal batch size for calibration")
    parser.add_argument("--min-batch", type=int, default=1,
                        help="Minimum batch size for dynamic engine")
    parser.add_argument("--max-batch", type=int, default=12,
                        help="Maximum batch size for dynamic engine")
    parser.add_argument("--num-images", type=int, default=128,
                        help="Number of calibration images to use")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading COCO128 (use existing)")

    args = parser.parse_args()

    # Download COCO128 dataset
    if not args.skip_download:
        dataset_dir = download_coco128()
    else:
        dataset_dir = Path("coco128")
        if not dataset_dir.exists():
            print("Error: coco128 directory not found. Remove --skip-download to download it.")
            return 1

    # Get calibration images
    calibration_images = get_calibration_images(dataset_dir, args.num_images)

    # Check ONNX exists
    if not os.path.exists(args.onnx):
        print(f"Error: ONNX model not found at {args.onnx}")
        return 1

    # Create output directory
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)

    # Build INT8 engine
    success = build_int8_engine(
        args.onnx,
        args.output,
        calibration_images,
        args.cache,
        batch_size=args.batch,
        min_batch=args.min_batch,
        max_batch=args.max_batch
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
