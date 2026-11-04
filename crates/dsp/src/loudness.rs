use anyhow::{Context, Result, bail};
use ebur128::{EbuR128, Mode};

use core::{AudioBuffer, N_CHANNELS};

/// EBU R128 gates in 400 ms blocks, so anything shorter has no integrated loudness.
pub const MIN_MEASURABLE_SECONDS: f32 = 0.4;

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

    let mut meter: EbuR128 = EbuR128::new(
        N_CHANNELS as u32,
        audio.sample_rate,
        Mode::I | Mode::LRA | Mode::TRUE_PEAK,
    )
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

    let mut true_peak: f64 = 0.0;
    let mut sample_peak: f64 = 0.0;
    for channel in 0..N_CHANNELS as u32 {
        true_peak = true_peak.max(
            meter
                .true_peak(channel)
                .context("Failed to read true peak")?,
        );
        sample_peak = sample_peak.max(
            meter
                .sample_peak(channel)
                .context("Failed to read sample peak")?,
        );
    }

    Ok(LoudnessStats {
        integrated_lufs,
        loudness_range_lu,
        true_peak_dbtp: peak_to_db(true_peak),
        sample_peak_db: peak_to_db(sample_peak),
    })
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
