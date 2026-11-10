use core::AudioBuffer;

pub fn sine_stereo(freq_hz: f64, peak_dbfs: f32, seconds: f32, sample_rate: u32) -> AudioBuffer {
    let frames: usize = (seconds * sample_rate as f32) as usize;
    let amp: f64 = 10f64.powf(peak_dbfs as f64 / 20.0);
    let mut samples: Vec<f32> = Vec::with_capacity(frames * 2);

    for i in 0..frames {
        let t: f64 = i as f64 / sample_rate as f64;
        let v: f32 = (amp * (2.0 * std::f64::consts::PI * freq_hz * t).sin()) as f32;
        samples.push(v);
        samples.push(v);
    }

    AudioBuffer {
        samples,
        sample_rate,
        channels: 2,
    }
}

/// Deterministic uniform noise in `[-amp, amp]`, independent per channel.
pub fn noise_stereo(peak_dbfs: f32, seconds: f32, sample_rate: u32, seed: u64) -> AudioBuffer {
    let frames: usize = (seconds * sample_rate as f32) as usize;
    let amp: f32 = 10f32.powf(peak_dbfs / 20.0);
    let mut state: u64 = seed.max(1);
    let mut samples: Vec<f32> = Vec::with_capacity(frames * 2);

    for _ in 0..frames * 2 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let unit: f32 = (state >> 40) as f32 / (1u64 << 24) as f32;
        samples.push((unit * 2.0 - 1.0) * amp);
    }

    AudioBuffer {
        samples,
        sample_rate,
        channels: 2,
    }
}

/// RMS level in dBFS of `samples`, ignoring the first `skip_frames` stereo frames
/// so filter and envelope transients do not bias steady-state checks.
pub fn rms_db(samples: &[f32], skip_frames: usize) -> f32 {
    let tail: &[f32] = &samples[(skip_frames * 2).min(samples.len())..];
    let sum_sq: f64 = tail.iter().map(|&x: &f32| (x as f64) * (x as f64)).sum();
    let mean_sq: f64 = sum_sq / tail.len().max(1) as f64;
    (10.0 * mean_sq.max(1e-20).log10()) as f32
}

/// Noise with a slow 0.5 Hz swell (about 20 dB of program dynamics), so the chain
/// has something to compress and the limiter something to catch.
pub fn modulated_noise_stereo(
    peak_dbfs: f32,
    seconds: f32,
    sample_rate: u32,
    seed: u64,
) -> AudioBuffer {
    let mut audio: AudioBuffer = noise_stereo(peak_dbfs, seconds, sample_rate, seed);

    for (i, frame) in audio.samples.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        let t: f64 = i as f64 / sample_rate as f64;
        let envelope: f32 = (0.55 + 0.45 * (2.0 * std::f64::consts::PI * 0.5 * t).sin()) as f32;
        for sample in frame.iter_mut() {
            *sample *= envelope;
        }
    }

    audio
}
