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

## Usage

Before running the CLI, ensure you have set the environment variable to link your PyTorch installation:

```bash
set LIBTORCH_USE_PYTORCH=1
```

### Running the CLI

The command accepts various audio formats (FLAC, MP3, WAV, etc.) and the target device (`cuda` or `cpu`).

**Using GPU (CUDA):**

```bash
cargo run --release --bin cli "PATH/TO/YOUR_AUDIO_FILE.MP3" cuda
```

**Using CPU:**

```bash
cargo run --release --bin cli "PATH/TO/YOUR_AUDIO_FILE.FLAC" cpu
```

## License

- MIT License ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/license/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
