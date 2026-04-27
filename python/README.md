# Demucs Model Exporter

This directory contains the Python environment and scripts necessary to download, trace, and export the **Demucs** neural network models into **TorchScript** format. These exported models are then loaded and executed by the Kintsugi Rust core.

## Overview

The primary goal of this module is to bridge the gap between the Python-based research environment (PyTorch/Demucs) and the high-performance Rust CLI. It ensures that the model architecture and weights are packaged correctly for cross-platform execution.

## Requirements

The environment is strictly locked to **PyTorch 2.1.0** with **CUDA 12.1** support to match the Rust `tch` bindings.

- **Python:** 3.11.0 (recommended)
- **Dependencies:** Managed via `requirements.txt`.

### Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Export Process

The `export_model.py` script performs the following steps:

1.  **Model Loading:** Downloads the `htdemucs` model (Hybrid Transformer Demucs) from the official repository.
2.  **Validation:** Verifies that the model adheres to the expected 4-stem output (drums, bass, other, vocals) and 44.1 kHz sample rate.
3.  **JIT Tracing:** Traces the model logic using a dummy audio tensor. This creates a serialized graph that can be run without a Python interpreter.
4.  **Atomic Saving:** Saves two distinct versions of the model to the `models/` directory:
    - `htdemucs.pt`: Optimized for CPU execution.
    - `htdemucs_cuda.pt`: Optimized for NVIDIA GPU execution.
5.  **Manifest Generation:** Creates JSON manifests containing critical metadata. _Note: These are currently generated for reference and auditing purposes; the Rust core uses internal constants for configuration._

## Output Artifacts

Running the export script generates the following files in the project root's `/models` folder:

| File                          | Description                            |
| :---------------------------- | :------------------------------------- |
| `htdemucs.pt`                 | The CPU-traced TorchScript model.      |
| `htdemucs_cuda.pt`            | The CUDA-traced TorchScript model.     |
| `htdemucs_manifest.json`      | Reference metadata for the CPU model.  |
| `htdemucs_cuda_manifest.json` | Reference metadata for the CUDA model. |

## Usage

To generate the models, simply run:

```bash
python scripts/export_model.py
```

> **Note:** Exporting the CUDA version requires an NVIDIA GPU and valid drivers installed on the host machine.

## License

- MIT License ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/license/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
