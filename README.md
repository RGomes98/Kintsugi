# Kintsugi

**Kintsugi** is an advanced audio processing suite designed to breathe new life into recordings. Inspired by the Japanese art of repairing broken pottery with gold, the project's ultimate goal is to function as a professional-grade **audio remasterer**, restoring and enhancing sound to modern standards.

### Project Roadmap

The remastering engine is being built in stages. What works today:

- **Audio Separation (Available Now):** Utilizing state-of-the-art neural networks, **Kintsugi** can deconstruct mixed tracks into individual stems (drums, bass, vocals, and other instruments) with high precision.
- **Mastering Chain (Available Now):** EQ, compression, loudness normalization to an [EBU R128](https://tech.ebu.ch/docs/r/r128.pdf) target, and a true-peak limiter, driven by presets and an optional TOML file. See [Mastering](#mastering).
- **Future Features:**
  - **Automatic Tonal Balancing:** Spectrum analysis that matches a target curve or a reference track.
  - **Per-Stem Mastering:** Running the chain on each separated stem, then remixing.
  - **Neural Parameter Prediction:** A model that picks the chain's settings for each track.
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
./kintsugi.sh separate song.mp3             # CPU image
./kintsugi.sh separate --cuda song.mp3
./kintsugi.sh separate --cuda song.mp3 --output ~/stems
```

Output files appear in `./output/` (or the directory passed with `--output`):

| File                    | Content         |
| :---------------------- | :-------------- |
| `drums.wav`             | Drum stem       |
| `bass.wav`              | Bass stem       |
| `vocals.wav`            | Vocal stem      |
| `other_instruments.wav` | Everything else |

### 3. Master a track

```bash
./kintsugi.sh master song.mp3                          # -14 LUFS, true-peak limited
./kintsugi.sh master song.mp3 --preset gentle
./kintsugi.sh master song.mp3 --preset loud --target-lufs -9
./kintsugi.sh master song.mp3 --config my-settings.toml
./kintsugi.sh analyze song.mp3                         # measure only, changes nothing
```

The result is written to `./output/<name>_mastered.wav`, and a before/after report is printed.

## Mastering

`master` runs a fixed chain on the stereo mix: **EQ → compressor → loudness gain → limiter**. After the limiter it measures the real output and corrects the gain until the integrated loudness is within tolerance (default ±0.2 LU), then guarantees the true peak stays under the limiter ceiling (default −1.0 dBTP). True peak is measured at 16x oversampling, finer than the 4x minimum of ITU-R BS.1770. If a target cannot be reached, for instance because the limiter or `max_gain_db` is in the way, the report says so instead of pretending.

Compressor thresholds assume a program at −18 LUFS: the mix is leveled to −18 LUFS before compressing and restored afterwards, so a preset compresses quiet and hot sources alike.

### Presets

| Preset    | Target     | What it does                                                                     |
| :-------- | :--------- | :------------------------------------------------------------------------------- |
| `neutral` | −14 LUFS   | Loudness and limiting only. No tonal or dynamic changes. (default)               |
| `gentle`  | −14 LUFS   | Clears sub-bass and low-mid mud, adds a little air, light 1.8:1 compression.     |
| `loud`    | −10 LUFS   | Tighter lows, brighter top, firmer 3:1 compression, hotter target.               |
| `vintage` | −16 LUFS   | Restores top end and presence on dull recordings; parallel compression.          |

The preset values are starting points to tune by ear, not a universal recipe.

### Custom settings

Pass a TOML file with `--config`. It only needs the values you want to change; settings are layered as **preset → config file → command-line flags**. Writing a section turns that stage on unless it says `enabled = false`, and a typo in a field name is an error rather than being ignored.

```toml
preset = "gentle"

[loudness]
target_lufs = -12.0
max_gain_db = 18.0        # safety cap on how much gain the loudness stage may apply

[[eq.band]]               # these bands replace the preset's bands
kind = "highpass"         # highpass, lowpass, lowshelf, highshelf or peak
freq_hz = 40.0

[[eq.band]]
kind = "peak"
freq_hz = 250.0
gain_db = -2.0
q = 1.0

[compressor]
threshold_db = -20.0      # for a mix leveled to -18 LUFS
ratio = 2.5
attack_ms = 20.0
release_ms = 150.0
knee_db = 6.0
mix = 0.8                 # below 1.0 blends in the dry signal (parallel compression)

[limiter]
ceiling_db = -1.0
lookahead_ms = 5.0
release_ms = 100.0
```

### Current limitations

- Stereo input only; WAV output is 32-bit float.
- The limiter works on sample peaks and a final trim enforces the true-peak ceiling, so on masters with heavy inter-sample peaks the trim can be a dB or more, which can keep `loud` short of its target on older material.

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
./target/release/cli separate path/to/song.mp3 --device cpu     # or --device cuda
./target/release/cli master path/to/song.mp3 --preset gentle
./target/release/cli analyze path/to/song.mp3
```

Stems and mastered files are written to `./output/` (override with `--output-dir`).

### 3. Run the tests

The DSP and mastering crates do not depend on LibTorch, so their tests run without any model or Python setup:

```bash
cargo test -p dsp -p master
```

## License

- MIT License ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/license/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
