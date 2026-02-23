# RTSPModule Scripts

This directory contains utility scripts for building and managing the RTSPModule project.

## Build Scripts

### `build_wheels_conda.sh`

This script automates the process of building Python wheels for multiple Python versions (3.9, 3.10, 3.11, 3.12) using Conda environments.

**Prerequisites:**

*   **Conda:** Miniconda or Anaconda should be installed.
*   **Automatic Detection:** The script automatically detects and sources Conda from:
    1.  The active shell's `PATH` (via `command -v conda`).
    2.  Standard locations: `~/miniconda3`, `~/anaconda3`, or `/opt/conda`.
*   **Build Project:** Ensure the C++ core is buildable (dependencies installed).

**Usage:**

Run the script from the project root or the `scripts/` directory:

```bash
./scripts/build_wheels_conda.sh
```

**What it does:**

1.  **Clean Setup:** Removes old wheels from the `wheels/` output directory.
2.  **Multi-Version Build:** Iterates through target Python versions (currently 3.9, 3.10, 3.11, 3.12).
3.  **Environment Management:** Creates isolated Conda environments (e.g., `wheel_py39`, `wheel_py310`) for each version if they don't already exist.
4.  **Dependency Handling:** Installs necessary build tools (`scikit-build-core`, `pybind11`, `numpy`) into these environments.
5.  **Wheel Generation:** Builds the Python commands using `pip wheel --no-build-isolation --no-deps`.
6.  **Output:** Saves the generated `.whl` files to the `wheels/` directory in the project root.
