"""
RTSPModule - Hardware-accelerated multi-stream RTSP decoder.

Provides GPU (NVDEC) and CPU buffer modes for high-performance video streaming
with zero-copy frame access via CUDA or NumPy.

Prerequisites:
    - GStreamer 1.20+ with plugins (gstreamer1.0-plugins-good, bad, ugly)
    - NVIDIA CUDA Toolkit 12.x (for GPU mode)
    - NVIDIA drivers with video decode capability

Example:
    >>> from RTSPModule import RTSPModule
    >>> client = RTSPModule()
    >>> client.start("config.yaml")
    >>> frame = client.get_cpu_frame(0)
    >>> client.stop()
"""
import ctypes
import sys
import warnings
from typing import TYPE_CHECKING

__version__ = "1.0.0"
__all__ = ["RTSPModule", "check_prerequisites", "PrerequisiteError"]


class PrerequisiteError(ImportError):
    """Raised when required system libraries are not available."""
    pass


def check_prerequisites(raise_on_error: bool = True) -> dict:
    """
    Check for required system dependencies.
    
    Args:
        raise_on_error: If True, raise PrerequisiteError on missing deps.
                       If False, return status dict without raising.
    
    Returns:
        dict: Status of each prerequisite:
            - gstreamer: bool
            - cuda: bool
            - cuda_version: str or None
    
    Raises:
        PrerequisiteError: If raise_on_error=True and a required dep is missing.
    """
    status = {
        "gstreamer": False,
        "cuda": False,
        "cuda_version": None,
    }
    
    # Check GStreamer
    gst_libs = [
        "libgstreamer-1.0.so.0",
        "libgstreamer-1.0.so",
    ]
    for lib in gst_libs:
        try:
            ctypes.CDLL(lib)
            status["gstreamer"] = True
            break
        except OSError:
            continue
    
    if not status["gstreamer"] and raise_on_error:
        raise PrerequisiteError(
            "GStreamer not found.\n"
            "Install on Ubuntu/Debian:\n"
            "  sudo apt install libgstreamer1.0-0 gstreamer1.0-plugins-base \\\n"
            "                   gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \\\n"
            "                   gstreamer1.0-plugins-ugly\n"
            "Install on RHEL/CentOS:\n"
            "  sudo yum install gstreamer1 gstreamer1-plugins-base gstreamer1-plugins-good"
        )
    
    # Check CUDA (optional for CPU-only mode)
    cuda_libs = [
        ("libcudart.so.12", "12"),
        ("libcudart.so.11", "11"),
        ("libcudart.so", None),
    ]
    for lib, version in cuda_libs:
        try:
            ctypes.CDLL(lib)
            status["cuda"] = True
            status["cuda_version"] = version
            break
        except OSError:
            continue
    
    if not status["cuda"]:
        warnings.warn(
            "CUDA runtime not found. GPU acceleration will be disabled.\n"
            "For GPU mode, install NVIDIA CUDA Toolkit 12.x:\n"
            "  https://developer.nvidia.com/cuda-downloads\n"
            "The module will fall back to CPU buffer mode.",
            RuntimeWarning,
            stacklevel=2
        )
    
    return status


def _import_native_module():
    """Import the compiled C++ extension module."""
    try:
        from . import _core
        return _core.RTSPModule
    except ModuleNotFoundError:
        # Fallback: try importing the .so directly from this package
        try:
            # The compiled module should be in the same directory
            from . import RTSPModule as _native
            return _native.RTSPModule
        except ImportError as e:
            raise ImportError(
                f"Failed to import RTSPModule native extension: {e}\n"
                "Ensure the package was installed correctly with 'pip install RTSPModule'"
            ) from e


# Perform prerequisite check on import
_prereq_status = check_prerequisites(raise_on_error=True)

# Import the native module
RTSPModule = _import_native_module()

# Re-export for convenience
if TYPE_CHECKING:
    from ._core import RTSPModule as RTSPModule
