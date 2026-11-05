use anyhow::{Result, bail};

use crate::loudness::{db_to_linear, linear_to_db};
use core::{AudioBuffer, N_CHANNELS};

pub const MIN_THRESHOLD_DB: f32 = -60.0;
pub const MAX_RATIO: f32 = 20.0;
pub const MIN_ATTACK_MS: f32 = 0.1;
pub const MAX_ATTACK_MS: f32 = 300.0;
pub const MIN_RELEASE_MS: f32 = 5.0;
pub const MAX_RELEASE_MS: f32 = 3000.0;
pub const MAX_KNEE_DB: f32 = 24.0;

/// Below this the envelope is flushed to zero to avoid denormal slowdowns in silence.
const ENVELOPE_FLOOR: f32 = 1e-12;

#[derive(Clone, Copy, Debug)]
pub struct CompressorParams {
    pub threshold_db: f32,
    pub ratio: f32,
    pub attack_ms: f32,
    pub release_ms: f32,
    pub knee_db: f32,
    /// Wet/dry balance in `[0, 1]`; below 1 gives parallel compression.
    pub mix: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CompressorStats {
    pub max_gain_reduction_db: f32,
    pub mean_gain_reduction_db: f32,
}

pub fn validate_compressor(params: &CompressorParams) -> Result<()> {
    if !(MIN_THRESHOLD_DB..=0.0).contains(&params.threshold_db) {
        bail!(
            "Compressor threshold {} dB is outside [{}, 0] dB",
            params.threshold_db,
            MIN_THRESHOLD_DB
        );
    }

    if !(1.0..=MAX_RATIO).contains(&params.ratio) {
        bail!(
            "Compressor ratio {} is outside [1, {}]",
            params.ratio,
            MAX_RATIO
        );
    }

    if !(MIN_ATTACK_MS..=MAX_ATTACK_MS).contains(&params.attack_ms) {
        bail!(
            "Compressor attack {} ms is outside [{}, {}] ms",
            params.attack_ms,
            MIN_ATTACK_MS,
            MAX_ATTACK_MS
        );
    }

    if !(MIN_RELEASE_MS..=MAX_RELEASE_MS).contains(&params.release_ms) {
        bail!(
            "Compressor release {} ms is outside [{}, {}] ms",
            params.release_ms,
            MIN_RELEASE_MS,
            MAX_RELEASE_MS
        );
    }

    if !(0.0..=MAX_KNEE_DB).contains(&params.knee_db) {
        bail!(
            "Compressor knee {} dB is outside [0, {}] dB",
            params.knee_db,
            MAX_KNEE_DB
        );
    }

    if !(0.0..=1.0).contains(&params.mix) {
        bail!("Compressor mix {} is outside [0, 1]", params.mix);
    }

    Ok(())
}

/// Static compression curve (Giannoulis et al., "Digital Dynamic Range Compressor
/// Design"): returns the output level in dB for an input level in dB.
fn output_level_db(input_db: f32, params: &CompressorParams) -> f32 {
    let over: f32 = input_db - params.threshold_db;
    let slope: f32 = 1.0 / params.ratio - 1.0;

    if params.knee_db > 0.0 && over.abs() <= params.knee_db / 2.0 {
        let knee_pos: f32 = over + params.knee_db / 2.0;
        input_db + slope * knee_pos * knee_pos / (2.0 * params.knee_db)
    } else if over > 0.0 {
        params.threshold_db + over / params.ratio
    } else {
        input_db
    }
}

fn smoothing_coefficient(time_ms: f32, sample_rate: u32) -> f32 {
    (-1.0 / (time_ms * 0.001 * sample_rate as f32)).exp()
}

/// Feed-forward compressor with a stereo-linked peak envelope detector.
///
/// The detector follows peaks instantly and decays with the release time constant,
/// so the effective ratio matches the configured one for steady signals (detecting
/// on the instantaneous level instead lets the gain recover at every zero crossing).
/// The attack time constant smooths how fast gain reduction is applied. There is no
/// internal makeup gain; the loudness stage that follows in the chain restores level.
pub fn apply_compressor(
    audio: &mut AudioBuffer,
    params: &CompressorParams,
) -> Result<CompressorStats> {
    validate_compressor(params)?;

    if audio.channels as usize != N_CHANNELS {
        bail!(
            "Compressor expects stereo audio, got {} channel(s)",
            audio.channels
        );
    }

    let attack: f32 = smoothing_coefficient(params.attack_ms, audio.sample_rate);
    let release: f32 = smoothing_coefficient(params.release_ms, audio.sample_rate);

    let mut envelope: f32 = 0.0;
    let mut smoothed_gain_db: f32 = 0.0;
    let mut max_reduction_db: f32 = 0.0;
    let mut reduction_sum_db: f64 = 0.0;
    let frames: usize = audio.samples.len() / N_CHANNELS;

    for frame in audio.samples.as_chunks_mut::<N_CHANNELS>().0.iter_mut() {
        let peak: f32 = frame[0].abs().max(frame[1].abs());
        envelope = peak.max(release * envelope);
        if envelope < ENVELOPE_FLOOR {
            envelope = 0.0;
        }

        let level_db: f32 = linear_to_db(envelope);
        let target_gain_db: f32 = output_level_db(level_db, params) - level_db;

        smoothed_gain_db = if target_gain_db < smoothed_gain_db {
            attack * smoothed_gain_db + (1.0 - attack) * target_gain_db
        } else {
            target_gain_db
        };

        let gain: f32 = db_to_linear(smoothed_gain_db);
        for sample in frame.iter_mut() {
            let dry: f32 = *sample;
            let wet: f32 = dry * gain;
            *sample = dry + params.mix * (wet - dry);
        }

        max_reduction_db = max_reduction_db.max(-smoothed_gain_db);
        reduction_sum_db += (-smoothed_gain_db) as f64;
    }

    Ok(CompressorStats {
        max_gain_reduction_db: max_reduction_db,
        mean_gain_reduction_db: (reduction_sum_db / frames.max(1) as f64) as f32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{rms_db, sine_stereo};

    const SR: u32 = 48_000;

    fn params() -> CompressorParams {
        CompressorParams {
            threshold_db: -20.0,
            ratio: 4.0,
            attack_ms: 10.0,
            release_ms: 100.0,
            knee_db: 0.0,
            mix: 1.0,
        }
    }

    fn steady_state_gain_db(peak_dbfs: f32, params: &CompressorParams) -> f32 {
        let input: AudioBuffer = sine_stereo(1000.0, peak_dbfs, 3.0, SR);
        let mut output: AudioBuffer = input.clone();
        apply_compressor(&mut output, params).unwrap();
        let skip: usize = SR as usize;
        rms_db(&output.samples, skip) - rms_db(&input.samples, skip)
    }

    #[test]
    fn reduces_level_above_threshold_by_the_ratio() {
        // Expected gain reduction: (level - threshold) * (1 - 1/ratio) = 12 * 0.75 = 9 dB.
        let gain: f32 = steady_state_gain_db(-8.0, &params());
        assert!((gain - -9.0).abs() < 0.5, "gain was {gain} dB");
    }

    #[test]
    fn leaves_signal_below_threshold_alone() {
        let gain: f32 = steady_state_gain_db(-40.0, &params());
        assert!(gain.abs() < 0.05, "gain was {gain} dB");
    }

    #[test]
    fn ratio_of_one_is_transparent() {
        let p: CompressorParams = CompressorParams {
            ratio: 1.0,
            ..params()
        };
        assert!(steady_state_gain_db(-8.0, &p).abs() < 0.01);
    }

    #[test]
    fn soft_knee_reduces_gently_at_the_threshold() {
        // At the threshold the knee gives (1 - 1/ratio) * knee / 8 = 0.75 dB of reduction.
        let p: CompressorParams = CompressorParams {
            knee_db: 8.0,
            ..params()
        };
        let gain: f32 = steady_state_gain_db(-20.0, &p);
        assert!((gain - -0.75).abs() < 0.3, "gain was {gain} dB");
    }

    #[test]
    fn attack_lets_the_onset_through_before_clamping_down() {
        let p: CompressorParams = CompressorParams {
            attack_ms: 20.0,
            ..params()
        };
        let loud: AudioBuffer = sine_stereo(1000.0, -6.0, 1.0, SR);
        let mut audio: AudioBuffer = loud.clone();
        apply_compressor(&mut audio, &p).unwrap();

        let peak = |from_ms: usize, to_ms: usize| -> f32 {
            audio.samples[from_ms * 48 * 2..to_ms * 48 * 2]
                .iter()
                .fold(0.0f32, |acc: f32, &x: &f32| acc.max(x.abs()))
        };

        let onset_db: f32 = linear_to_db(peak(0, 2));
        let settled_db: f32 = linear_to_db(peak(500, 520));
        assert!(
            onset_db - settled_db > 3.0,
            "onset {onset_db} dB, settled {settled_db} dB"
        );
    }

    #[test]
    fn release_recovers_gradually_after_a_loud_passage() {
        let p: CompressorParams = CompressorParams {
            release_ms: 300.0,
            ..params()
        };
        let mut audio: AudioBuffer = sine_stereo(1000.0, -6.0, 1.0, SR);
        let quiet: AudioBuffer = sine_stereo(1000.0, -30.0, 5.0, SR);
        audio.samples.extend_from_slice(&quiet.samples);

        let reference: f32 = rms_db(&quiet.samples, 0);
        apply_compressor(&mut audio, &p).unwrap();

        let after_drop: usize = SR as usize;
        let window = |from_ms: usize, to_ms: usize| -> f32 {
            rms_db(
                &audio.samples[(after_drop + from_ms * 48) * 2..(after_drop + to_ms * 48) * 2],
                0,
            )
        };

        let early: f32 = window(50, 100) - reference;
        let late: f32 = window(3500, 4000) - reference;
        assert!(early < -2.0, "still {early} dB shortly after the drop");
        assert!(late.abs() < 0.2, "{late} dB long after the drop");
    }

    #[test]
    fn stereo_link_applies_the_same_gain_to_both_channels() {
        let mut audio: AudioBuffer = sine_stereo(1000.0, -6.0, 3.0, SR);
        let quiet_right: f32 = db_to_linear(-40.0) / db_to_linear(-6.0);
        for frame in audio.samples.as_chunks_mut::<2>().0.iter_mut() {
            frame[1] *= quiet_right;
        }
        let input: AudioBuffer = audio.clone();

        apply_compressor(&mut audio, &params()).unwrap();

        let skip: usize = SR as usize;
        let channel_gain = |channel: usize| -> f32 {
            let out: Vec<f32> = audio
                .samples
                .iter()
                .skip(channel)
                .step_by(2)
                .copied()
                .collect();
            let inp: Vec<f32> = input
                .samples
                .iter()
                .skip(channel)
                .step_by(2)
                .copied()
                .collect();
            rms_db(&out, skip / 2) - rms_db(&inp, skip / 2)
        };

        // rms_db treats its input as interleaved, but both sides use the same
        // convention, so the difference per channel is still the applied gain.
        assert!((channel_gain(0) - channel_gain(1)).abs() < 0.05);
        assert!(channel_gain(1) < -6.0, "quiet channel was not ducked");
    }

    #[test]
    fn zero_mix_is_a_bypass() {
        let p: CompressorParams = CompressorParams {
            mix: 0.0,
            ..params()
        };
        let input: AudioBuffer = sine_stereo(1000.0, -3.0, 1.0, SR);
        let mut output: AudioBuffer = input.clone();
        apply_compressor(&mut output, &p).unwrap();
        assert_eq!(input.samples, output.samples);
    }

    #[test]
    fn reports_gain_reduction_statistics() {
        let mut audio: AudioBuffer = sine_stereo(1000.0, -8.0, 3.0, SR);
        let stats: CompressorStats = apply_compressor(&mut audio, &params()).unwrap();
        assert!((stats.max_gain_reduction_db - 9.0).abs() < 1.0);
        assert!(stats.mean_gain_reduction_db > 5.0);
        assert!(stats.mean_gain_reduction_db <= stats.max_gain_reduction_db);
    }

    #[test]
    fn silence_stays_silent_without_nan() {
        let mut audio: AudioBuffer = AudioBuffer {
            samples: vec![0.0; SR as usize * 2],
            sample_rate: SR,
            channels: 2,
        };
        apply_compressor(&mut audio, &params()).unwrap();
        assert!(audio.samples.iter().all(|&x: &f32| x == 0.0));
    }

    #[test]
    fn invalid_parameters_are_rejected() {
        let mut audio: AudioBuffer = sine_stereo(1000.0, -12.0, 0.5, SR);
        let bad: [CompressorParams; 7] = [
            CompressorParams {
                threshold_db: 3.0,
                ..params()
            },
            CompressorParams {
                threshold_db: -90.0,
                ..params()
            },
            CompressorParams {
                ratio: 0.5,
                ..params()
            },
            CompressorParams {
                attack_ms: 0.0,
                ..params()
            },
            CompressorParams {
                release_ms: 10_000.0,
                ..params()
            },
            CompressorParams {
                knee_db: -1.0,
                ..params()
            },
            CompressorParams {
                mix: 1.5,
                ..params()
            },
        ];

        for p in bad {
            assert!(apply_compressor(&mut audio, &p).is_err(), "{p:?}");
        }
    }
}
