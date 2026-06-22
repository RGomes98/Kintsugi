# Kintsugi

**Kintsugi** is an advanced audio processing suite designed to breathe new life into recordings. Inspired by the Japanese art of repairing broken pottery with gold, the project's ultimate goal is to function as a professional-grade **audio remasterer**, restoring and enhancing sound to modern standards.

### Project Roadmap

While the full remastering engine is under development, the current version provides a high-performance foundation for audio manipulation:

- **Audio Separation (Available Now):** Utilizing state-of-the-art neural networks, **Kintsugi** can deconstruct mixed tracks into individual stems (drums, bass, vocals, and other instruments) with high precision.
- **Future Features:**
  - **Neural Remastering:** Automated EQ, compression, and tonal balancing.
  - **Noise Reduction:** Surgical removal of hiss, hum, and artifacts.
  - **Stereo Widening:** Enhancing the spatial image of vintage mono or narrow-field recordings.
  - **Harmonic Restoration:** Synthesizing missing frequencies lost to low-bitrate compression or aging tape.

## Quick Start (Docker)

### 1. Build the image for your hardware

```bash
./docker_build.sh --cpu   # CPU image
./docker_build.sh --cuda  # CUDA image (requires NVIDIA GPU)
./docker_build.sh         # both
```

### 2. Separate a track

```bash
./kintsugi.sh --cpu song.mp3
./kintsugi.sh --cuda song.mp3
./kintsugi.sh --cuda song.mp3 --output ~/stems
```

Output files appear in `./output/` (or the directory passed with `--output`):

| File                    | Content         |
| :---------------------- | :-------------- |
| `drums.wav`             | Drum stem       |
| `bass.wav`              | Bass stem       |
| `vocals.wav`            | Vocal stem      |
| `other_instruments.wav` | Everything else |

## Development

### 1. Export the model

Follow the setup in [`python/README.md`](python/README.md), then run:

```bash
cd python
source .venv/bin/activate
python scripts/export_model.py --device cpu   # or --device cuda
```

This writes the TorchScript model to `python/models/`.

### 2. Build and run the CLI

`tch-rs` (the Rust PyTorch bindings) needs to locate LibTorch at build time. Point it at the venv you just created:

```bash
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH="$(pwd)/python/.venv/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH"

cargo build --release
```

Run it from the project root (the CLI resolves model paths relative to the working directory):

```bash
./target/release/cli path/to/song.mp3 cpu
./target/release/cli path/to/song.mp3 cuda
```

Output stems are written to `./output/`.

## License

- MIT License ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/license/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
