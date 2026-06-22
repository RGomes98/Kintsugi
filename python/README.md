# Demucs Model Exporter

This directory contains the Python environment and scripts necessary to download, trace, and export the **Demucs** neural network models into **TorchScript** format. These exported models are then loaded and executed by the Kintsugi Rust core.

## Overview

The primary goal of this module is to bridge the gap between the Python-based research environment (PyTorch/Demucs) and the high-performance Rust CLI. It ensures that the model architecture and weights are packaged correctly for cross-platform execution.

> **Using Docker?** The model export runs automatically during the image build — you don't need to run anything here. See the root [`README.md`](../README.md) for Docker instructions.

## Requirements

The environment is strictly locked to **PyTorch 2.1.0** to match the Rust `tch` bindings. Two requirements files are provided to minimize download size:

- **Python:** 3.11 (recommended)
- `requirements-cpu.txt` — CPU-only PyTorch
- `requirements-cuda.txt` — NVIDIA GPU (CUDA 12.1)

> **Note:** Exporting the CUDA version requires an NVIDIA GPU with valid drivers installed on the host machine.

### Local Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate

# CPU-only machine:
pip install -r requirements-cpu.txt

# NVIDIA GPU:
pip install -r requirements-cuda.txt
```

## Export Process

The `export_model.py` script performs the following steps:

1. **Model Loading:** Downloads the `htdemucs` model (Hybrid Transformer Demucs) from the official repository.
2. **Validation:** Verifies that the model adheres to the expected 4-stem output (drums, bass, other, vocals) and 44.1 kHz sample rate.
3. **JIT Tracing:** Traces the model logic using a dummy audio tensor.
4. **Atomic Saving:** Saves the traced model(s) to the `models/` directory inside this folder.
5. **Manifest Generation:** Creates JSON manifests containing critical metadata. _Note: These are currently generated for reference and auditing purposes; the Rust core uses internal constants for configuration._

## Output Artifacts

Running the export script generates the following files in `python/models/`:

| File                          | Description                   |
| :---------------------------- | :---------------------------- |
| `htdemucs.pt`                 | CPU-traced TorchScript model  |
| `htdemucs_cuda.pt`            | CUDA-traced TorchScript model |
| `htdemucs_manifest.json`      | Metadata for the CPU model    |
| `htdemucs_cuda_manifest.json` | Metadata for the CUDA model   |

## Usage

```bash
# Export for all available devices (default):
python scripts/export_model.py

# CPU only:
python scripts/export_model.py --device cpu

# CUDA only (requires an NVIDIA GPU):
python scripts/export_model.py --device cuda
```

## License

- MIT License ([LICENSE-MIT](../LICENSE-MIT) or https://opensource.org/license/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](../LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
