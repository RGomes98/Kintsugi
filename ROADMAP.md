# **Kintsugi Roadmap**

Each item below corresponds to an issue of the [Kintsugi Backlog](https://github.com/users/RGomes98/projects/6) and each phase corresponds to a Milestone on GitHub.
Priority: `P0` critical, `P1` high, `P2` medium, `P3` low, `Icebox` unscheduled. Size: `XS` to `XL`.

**Where we are:** stem separation with HTDemucs works from the command line and in Docker. The first stage of the mastering chain (loudness, EQ, compressor and limiter, with presets and a TOML file) is in review in #1. Next come proving the Docker path after the CLI change (#6), a CI pipeline (#7) and the true-peak limiter (#8), and then tonal balancing.

## **Phase 0 — Foundation (completed)**

**Goal:** Separate a track into stems from the command line, with a reproducible Docker build.

- [x] **Stem separation with HTDemucs** — P0 · L (#2)
- [x] **Docker build and run scripts** — P1 · M (#3)
- [x] **Multi-format audio decoding (mp3, aac, mp4)** — P1 · S (#4)

## **Phase 1 — Mastering Chain**

**Goal:** A deterministic, testable chain on the stereo mix (loudness normalization, EQ, compression and true-peak limiting) controlled by presets and a TOML file, with CI and a verified Docker path.

- [ ] **Mastering chain: loudness, EQ, compressor and limiter** — P0 · XL (#5)
- [ ] **Verify the Docker image and `separate` end to end** — P1 · S (#6)
- [ ] **CI pipeline: format, lint and tests** — P1 · M (#7)
- [ ] **True-peak limiter with oversampled detection** — P1 · M (#8)
- [ ] **Mono input, resampling and 16/24-bit output** — P2 · L (#9)
- [ ] **Decouple the CLI from LibTorch and rename the `core` package** — P2 · M (#10)
- [ ] **Listening tests and preset tuning** — P2 · M (#11)

## **Phase 2 — Tonal Balancing**

**Goal:** Measure the spectrum and correct the tonal balance automatically, from a target curve or from a reference track.

- [ ] **Spectral analysis and tonal balance metrics** — P1 · M (#12)
- [ ] **Target-curve EQ matching** — P1 · L (#13)
- [ ] **Reference-track matching** — P2 · L (#14)

## **Phase 3 — Per-Stem Mastering**

**Goal:** Run the chain on each separated stem and remix, so every instrument gets the treatment it needs.

- [ ] **Per-stem chains and remix** — P1 · L (#15)

## **Phase 4 — Restoration & Stereo Imaging**

**Goal:** Remove noise, restore lost high frequencies where the licenses allow it, and widen narrow recordings.

- [ ] **Spike: noise reduction on music** — P1 · M (#17)
- [ ] **Stereo widening** — P2 · M (#16)
- [ ] **Noise reduction stage** — P2 · L (#18)
- [ ] **Spike: harmonic restoration and bandwidth extension** — P2 · M (#19)

## **Phase 5 — Learned Mastering**

**Goal:** A model that picks the chain's settings for each track, while the signal processing stays deterministic and editable.

- [ ] **Parameter-prediction model** — Icebox · XL (#20)

## **Phase 6 — Quality, Docs & Distribution**

**Goal:** User documentation, a release process and distribution.

- [ ] **User guide: presets and TOML reference** — P2 · S (#21)
