use anyhow::{Context, Result, bail};
use ebur128::{EbuR128, Mode};

use core::{AudioBuffer, N_CHANNELS};

/// EBU R128 gates in 400 ms blocks, so anything shorter has no integrated loudness.
pub const MIN_MEASURABLE_SECONDS: f32 = 0.4;

/// BS.1770's 4x minimum can miss inter-sample peaks by about 0.1 dB.
pub const TRUE_PEAK_OVERSAMPLING: usize = 16;
/// Worst under-read of the true-peak meter, reached on content above 20 kHz.
pub const TRUE_PEAK_MAX_UNDER_READ_DB: f64 = 0.03;
const TRUE_PEAK_TAPS_PER_PHASE: usize = 48;
const TRUE_PEAK_KAISER_BETA: f64 = 8.0;

#[derive(Clone, Copy, Debug)]
pub struct LoudnessStats {
    pub integrated_lufs: f64,
    pub loudness_range_lu: f64,
    pub true_peak_dbtp: f64,
    pub sample_peak_db: f64,
}

impl LoudnessStats {
    /// True when there is no measurable program loudness (digital silence or too short).
    pub fn is_silent(&self) -> bool {
        !self.integrated_lufs.is_finite()
    }
}

pub fn db_to_linear(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

pub fn linear_to_db(linear: f32) -> f32 {
    20.0 * linear.max(1e-10).log10()
}

/// Formats a level with one decimal: `-inf` for silence, and never `-0.0`.
pub fn format_db(value: f64) -> String {
    if value == f64::NEG_INFINITY {
        return "-inf".to_string();
    }

    let rounded: f64 = (value * 10.0).round() / 10.0;
    if rounded == 0.0 {
        "0.0".to_string()
    } else {
        format!("{rounded:.1}")
    }
}

pub fn apply_gain_db(samples: &mut [f32], gain_db: f32) {
    let gain: f32 = db_to_linear(gain_db);
    for sample in samples.iter_mut() {
        *sample *= gain;
    }
}

pub fn measure_loudness(audio: &AudioBuffer) -> Result<LoudnessStats> {
    if audio.channels as usize != N_CHANNELS {
        bail!(
            "Loudness measurement expects stereo audio, got {} channel(s)",
            audio.channels
        );
    }

    if audio.samples.is_empty() {
        bail!("Audio buffer is empty");
    }

    if !audio.samples.len().is_multiple_of(N_CHANNELS) {
        bail!("Stereo interleaved buffer must have an even number of samples");
    }

    let mut meter: EbuR128 =
        EbuR128::new(N_CHANNELS as u32, audio.sample_rate, Mode::I | Mode::LRA)
            .context("Failed to initialize loudness meter")?;

    meter
        .add_frames_f32(&audio.samples)
        .context("Failed to feed audio to loudness meter")?;

    let integrated_lufs: f64 = meter
        .loudness_global()
        .context("Failed to read integrated loudness")?;
    let loudness_range_lu: f64 = meter
        .loudness_range()
        .context("Failed to read loudness range")?;

    let sample_peak: f32 = audio
        .samples
        .iter()
        .fold(0.0f32, |acc: f32, &x: &f32| acc.max(x.abs()));
    // Interpolated points never land on input samples, so a peak on a sample can read low.
    let true_peak: f32 = true_peak_linear(audio).max(sample_peak);

    Ok(LoudnessStats {
        integrated_lufs,
        loudness_range_lu,
        true_peak_dbtp: peak_to_db(true_peak as f64),
        sample_peak_db: peak_to_db(sample_peak as f64),
    })
}

/// Integrated loudness only, for callers that do not need peaks or loudness range.
pub fn measure_integrated_lufs(audio: &AudioBuffer) -> Result<f64> {
    if audio.channels as usize != N_CHANNELS {
        bail!(
            "Loudness measurement expects stereo audio, got {} channel(s)",
            audio.channels
        );
    }

    let mut meter: EbuR128 = EbuR128::new(N_CHANNELS as u32, audio.sample_rate, Mode::I)
        .context("Failed to initialize loudness meter")?;
    meter
        .add_frames_f32(&audio.samples)
        .context("Failed to feed audio to loudness meter")?;

    meter
        .loudness_global()
        .context("Failed to read integrated loudness")
}

/// Highest absolute value of the signal reconstructed at `TRUE_PEAK_OVERSAMPLING`x.
fn true_peak_linear(audio: &AudioBuffer) -> f32 {
    let phases: Vec<[f32; TRUE_PEAK_TAPS_PER_PHASE]> = true_peak_phases();
    let frames: usize = audio.samples.len() / N_CHANNELS;
    let threads: usize = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut peak: f32 = 0.0;

    for channel in 0..N_CHANNELS {
        // Zero padding lets the filter ring out past both ends.
        let mut padded: Vec<f32> = vec![0.0; frames + 2 * TRUE_PEAK_TAPS_PER_PHASE];
        for (slot, frame) in padded[TRUE_PEAK_TAPS_PER_PHASE..]
            .iter_mut()
            .zip(audio.samples.as_chunks::<N_CHANNELS>().0)
        {
            *slot = frame[channel];
        }

        // Parts overlap by one window so none is skipped.
        let windows: usize = padded.len() - TRUE_PEAK_TAPS_PER_PHASE + 1;
        let per_part: usize = windows.div_ceil(threads);
        let channel_peak: f32 = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..windows)
                .step_by(per_part)
                .map(|start: usize| {
                    let end: usize = (start + per_part).min(windows) + TRUE_PEAK_TAPS_PER_PHASE - 1;
                    let part: &[f32] = &padded[start..end];
                    let phases: &[[f32; TRUE_PEAK_TAPS_PER_PHASE]] = &phases;
                    scope.spawn(move || interpolated_peak(part, phases))
                })
                .collect();

            handles
                .into_iter()
                .map(|handle| handle.join().expect("true-peak worker panicked"))
                .fold(0.0f32, f32::max)
        });
        peak = peak.max(channel_peak);
    }

    peak
}

fn interpolated_peak(samples: &[f32], phases: &[[f32; TRUE_PEAK_TAPS_PER_PHASE]]) -> f32 {
    let mut peak: f32 = 0.0;

    for window in samples.windows(TRUE_PEAK_TAPS_PER_PHASE) {
        for taps in phases {
            // Separate accumulators let the compiler vectorize.
            let mut acc: [f32; 8] = [0.0; 8];
            for (xs, hs) in window
                .as_chunks::<8>()
                .0
                .iter()
                .zip(taps.as_chunks::<8>().0)
            {
                for lane in 0..8 {
                    acc[lane] += xs[lane] * hs[lane];
                }
            }
            peak = peak.max(acc.iter().sum::<f32>().abs());
        }
    }

    peak
}

/// Polyphase taps of a Kaiser-windowed sinc, reversed and normalized to unit DC gain.
fn true_peak_phases() -> Vec<[f32; TRUE_PEAK_TAPS_PER_PHASE]> {
    let factor: usize = TRUE_PEAK_OVERSAMPLING;
    let len: usize = factor * TRUE_PEAK_TAPS_PER_PHASE;
    let center: f64 = (len - 1) as f64 / 2.0;
    let window_norm: f64 = bessel_i0(TRUE_PEAK_KAISER_BETA);

    let prototype: Vec<f64> = (0..len)
        .map(|n: usize| {
            let t: f64 = (n as f64 - center) / factor as f64;
            let sinc: f64 = if t == 0.0 {
                1.0
            } else {
                (std::f64::consts::PI * t).sin() / (std::f64::consts::PI * t)
            };
            let r: f64 = (n as f64 - center) / center;
            let window: f64 =
                bessel_i0(TRUE_PEAK_KAISER_BETA * (1.0 - r * r).max(0.0).sqrt()) / window_norm;
            sinc * window
        })
        .collect();

    (0..factor)
        .map(|phase: usize| {
            let taps: Vec<f64> = (0..TRUE_PEAK_TAPS_PER_PHASE)
                .map(|k: usize| prototype[k * factor + phase])
                .collect();
            let sum: f64 = taps.iter().sum();

            let mut reversed: [f32; TRUE_PEAK_TAPS_PER_PHASE] = [0.0; TRUE_PEAK_TAPS_PER_PHASE];
            for (slot, tap) in reversed.iter_mut().zip(taps.iter().rev()) {
                *slot = (tap / sum) as f32;
            }
            reversed
        })
        .collect()
}

/// Zeroth-order modified Bessel function of the first kind, for the Kaiser window.
fn bessel_i0(x: f64) -> f64 {
    let half: f64 = x / 2.0;
    let mut term: f64 = 1.0;
    let mut sum: f64 = 1.0;

    for k in 1..64 {
        term *= (half / k as f64).powi(2);
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }

    sum
}

fn peak_to_db(peak: f64) -> f64 {
    if peak <= 0.0 {
        f64::NEG_INFINITY
    } else {
        20.0 * peak.log10()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{noise_stereo, sine_stereo};

    const SR: u32 = 48_000;

    #[test]
    fn stereo_sine_matches_ebu_tech_3341_reference() {
        // EBU Tech 3341 test 1: stereo 1 kHz sine at -23 dBFS reads -23.0 LUFS.
        let audio: AudioBuffer = sine_stereo(1000.0, -23.0, 20.0, SR);
        let stats: LoudnessStats = measure_loudness(&audio).unwrap();
        assert!(
            (stats.integrated_lufs - -23.0).abs() < 0.1,
            "got {} LUFS",
            stats.integrated_lufs
        );
    }

    #[test]
    fn six_db_of_gain_raises_loudness_by_six_db() {
        let mut audio: AudioBuffer = noise_stereo(-20.0, 10.0, SR, 7);
        let before: f64 = measure_loudness(&audio).unwrap().integrated_lufs;

        apply_gain_db(&mut audio.samples, 6.0206);
        let after: f64 = measure_loudness(&audio).unwrap().integrated_lufs;

        assert!(
            ((after - before) - 6.0206).abs() < 0.05,
            "delta was {}",
            after - before
        );
    }

    #[test]
    fn sample_peak_tracks_signal_amplitude() {
        let audio: AudioBuffer = sine_stereo(1000.0, -6.0, 2.0, SR);
        let stats: LoudnessStats = measure_loudness(&audio).unwrap();
        assert!((stats.sample_peak_db - -6.0).abs() < 0.1);
    }

    #[test]
    fn true_peak_is_never_below_sample_peak() {
        let audio: AudioBuffer = noise_stereo(-3.0, 5.0, SR, 99);
        let stats: LoudnessStats = measure_loudness(&audio).unwrap();
        assert!(stats.true_peak_dbtp >= stats.sample_peak_db - 1e-6);
    }

    /// Stereo sine with 10 ms fades, so the buffer edges add no ringing.
    fn faded_sine(freq_hz: f64, peak_dbfs: f32, phase: f64, sample_rate: u32) -> AudioBuffer {
        let frames: usize = sample_rate as usize / 2;
        let fade: usize = sample_rate as usize / 100;
        let amp: f64 = 10f64.powf(peak_dbfs as f64 / 20.0);
        let mut samples: Vec<f32> = Vec::with_capacity(frames * 2);

        for i in 0..frames {
            let edge: usize = i.min(frames - 1 - i);
            let gain: f64 = if edge < fade {
                (std::f64::consts::FRAC_PI_2 * edge as f64 / fade as f64)
                    .sin()
                    .powi(2)
            } else {
                1.0
            };
            let t: f64 = i as f64 / sample_rate as f64;
            let v: f32 =
                (gain * amp * (2.0 * std::f64::consts::PI * freq_hz * t + phase).sin()) as f32;
            samples.push(v);
            samples.push(v);
        }

        AudioBuffer {
            samples,
            sample_rate,
            channels: 2,
        }
    }

    #[test]
    fn true_peak_finds_the_peak_between_samples() {
        // Every sample of a quarter-rate sine at 45 degrees sits 3 dB under its crest.
        let audio: AudioBuffer = faded_sine(SR as f64 / 4.0, -6.0, std::f64::consts::FRAC_PI_4, SR);
        let stats: LoudnessStats = measure_loudness(&audio).unwrap();

        assert!(
            (stats.sample_peak_db - -9.01).abs() < 0.05,
            "{}",
            stats.sample_peak_db
        );
        assert!(
            (stats.true_peak_dbtp - -6.0).abs() < 0.02,
            "{}",
            stats.true_peak_dbtp
        );
    }

    #[test]
    fn true_peak_is_accurate_near_the_top_of_the_band() {
        for phase in [0.0, 0.3, 0.9, 1.7, 2.6] {
            let audio: AudioBuffer = faded_sine(19_000.0, -3.0, phase, 44_100);
            let true_peak: f64 = measure_loudness(&audio).unwrap().true_peak_dbtp;
            assert!(
                (true_peak - -3.0).abs() < 0.02,
                "phase {phase}: {true_peak} dBTP"
            );
        }
    }

    #[test]
    fn under_read_above_20_khz_stays_within_the_documented_bound() {
        // 21 kHz at 44.1 kHz falls in the interpolator's transition band.
        let mut worst: f64 = 0.0;
        for step in 0..32 {
            let phase: f64 = std::f64::consts::PI * step as f64 / 32.0;
            let audio: AudioBuffer = faded_sine(21_000.0, -3.0, phase, 44_100);
            let true_peak: f64 = measure_loudness(&audio).unwrap().true_peak_dbtp;
            worst = worst.min(true_peak - -3.0);
        }
        assert!(
            -worst <= TRUE_PEAK_MAX_UNDER_READ_DB,
            "worst under-read {worst} dB"
        );
    }

    #[test]
    fn integrated_only_measurement_matches_the_full_one() {
        let audio: AudioBuffer = noise_stereo(-12.0, 5.0, SR, 3);
        let full: f64 = measure_loudness(&audio).unwrap().integrated_lufs;
        let quick: f64 = measure_integrated_lufs(&audio).unwrap();
        assert!((full - quick).abs() < 1e-9, "{full} vs {quick}");
    }

    #[test]
    fn digital_silence_is_reported_as_silent() {
        let audio: AudioBuffer = AudioBuffer {
            samples: vec![0.0; SR as usize * 2 * 2],
            sample_rate: SR,
            channels: 2,
        };
        let stats: LoudnessStats = measure_loudness(&audio).unwrap();
        assert!(stats.is_silent());
    }

    #[test]
    fn audio_shorter_than_one_gating_block_is_reported_as_silent() {
        let audio: AudioBuffer = sine_stereo(1000.0, -20.0, 0.2, SR);
        let stats: LoudnessStats = measure_loudness(&audio).unwrap();
        assert!(stats.is_silent());
    }

    #[test]
    fn non_stereo_input_is_rejected() {
        let audio: AudioBuffer = AudioBuffer {
            samples: vec![0.1; SR as usize],
            sample_rate: SR,
            channels: 1,
        };
        assert!(measure_loudness(&audio).is_err());
    }

    #[test]
    fn format_db_handles_silence_and_negative_zero() {
        assert_eq!(format_db(f64::NEG_INFINITY), "-inf");
        assert_eq!(format_db(-0.04), "0.0");
        assert_eq!(format_db(0.0), "0.0");
        assert_eq!(format_db(-1.26), "-1.3");
        assert_eq!(format_db(2.14), "2.1");
    }

    #[test]
    fn db_conversions_round_trip() {
        for db in [-60.0f32, -12.0, 0.0, 6.0, 12.0] {
            assert!((linear_to_db(db_to_linear(db)) - db).abs() < 1e-4);
        }
    }
}
