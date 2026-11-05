use std::collections::VecDeque;

use anyhow::{Result, bail};

use crate::loudness::db_to_linear;
use core::{AudioBuffer, N_CHANNELS};

pub const MIN_CEILING_DB: f32 = -20.0;
pub const MIN_LOOKAHEAD_MS: f32 = 0.5;
pub const MAX_LOOKAHEAD_MS: f32 = 20.0;
pub const MIN_RELEASE_MS: f32 = 10.0;
pub const MAX_RELEASE_MS: f32 = 1000.0;

#[derive(Clone, Copy, Debug)]
pub struct LimiterParams {
    pub ceiling_db: f32,
    pub lookahead_ms: f32,
    pub release_ms: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LimiterStats {
    pub max_gain_reduction_db: f32,
}

pub fn validate_limiter(params: &LimiterParams) -> Result<()> {
    if !(MIN_CEILING_DB..=0.0).contains(&params.ceiling_db) {
        bail!(
            "Limiter ceiling {} dB is outside [{}, 0] dB",
            params.ceiling_db,
            MIN_CEILING_DB
        );
    }

    if !(MIN_LOOKAHEAD_MS..=MAX_LOOKAHEAD_MS).contains(&params.lookahead_ms) {
        bail!(
            "Limiter lookahead {} ms is outside [{}, {}] ms",
            params.lookahead_ms,
            MIN_LOOKAHEAD_MS,
            MAX_LOOKAHEAD_MS
        );
    }

    if !(MIN_RELEASE_MS..=MAX_RELEASE_MS).contains(&params.release_ms) {
        bail!(
            "Limiter release {} ms is outside [{}, {}] ms",
            params.release_ms,
            MIN_RELEASE_MS,
            MAX_RELEASE_MS
        );
    }

    Ok(())
}

/// Brickwall lookahead limiter on the stereo-linked sample peak.
///
/// For each frame `i` the gain `G[i]` is the mean of `M[j]` over `j in [i-L, i]`, where
/// `M[j]` is the minimum required gain over `[j, j+L]`. Every window contains `i`, so
/// `G[i]` never exceeds the gain frame `i` needs: no sample ends above the ceiling, and
/// the gain ramps down over the `L`-sample lookahead instead of clicking. Release is an
/// exponential recovery combined with `min`, which can only lower the gain and so keeps
/// the guarantee. The buffer is processed offline, so no latency compensation is needed.
///
/// The guarantee is on the sample peak; inter-sample (true) peaks can still exceed the
/// ceiling slightly, which the mastering chain verifies and trims.
pub fn apply_limiter(audio: &mut AudioBuffer, params: &LimiterParams) -> Result<LimiterStats> {
    validate_limiter(params)?;

    if audio.channels as usize != N_CHANNELS {
        bail!(
            "Limiter expects stereo audio, got {} channel(s)",
            audio.channels
        );
    }

    let frames: usize = audio.samples.len() / N_CHANNELS;
    if frames == 0 {
        return Ok(LimiterStats::default());
    }

    let ceiling: f32 = db_to_linear(params.ceiling_db);
    let lookahead: usize =
        ((params.lookahead_ms * 0.001 * audio.sample_rate as f32).round() as usize).max(1);
    let release: f32 = (-1.0 / (params.release_ms * 0.001 * audio.sample_rate as f32)).exp();

    let mut gain: Vec<f32> = audio
        .samples
        .as_chunks::<N_CHANNELS>()
        .0
        .iter()
        .map(|frame: &[f32; N_CHANNELS]| {
            let peak: f32 = frame[0].abs().max(frame[1].abs());
            if peak > ceiling { ceiling / peak } else { 1.0 }
        })
        .collect();

    sliding_minimum_in_place(&mut gain, lookahead);

    let window_len: usize = lookahead + 1;
    let mut history: Vec<f32> = vec![gain[0]; window_len];
    let mut sum: f64 = gain[0] as f64 * window_len as f64;
    let mut released: f32 = 1.0;
    let mut min_gain: f32 = 1.0;

    for (i, frame) in audio
        .samples
        .as_chunks_mut::<N_CHANNELS>()
        .0
        .iter_mut()
        .enumerate()
    {
        let slot: usize = i % window_len;
        sum += gain[i] as f64 - history[slot] as f64;
        history[slot] = gain[i];

        let smoothed: f32 = (sum / window_len as f64) as f32;
        let recovering: f32 = 1.0 - (1.0 - released) * release;
        released = smoothed.min(recovering);
        min_gain = min_gain.min(released);

        for sample in frame.iter_mut() {
            *sample *= released;
        }
    }

    Ok(LimiterStats {
        max_gain_reduction_db: -20.0 * min_gain.max(1e-10).log10(),
    })
}

/// Replaces `values[j]` with `min(values[j..=j + window])` (clamped at the end).
fn sliding_minimum_in_place(values: &mut [f32], window: usize) {
    let len: usize = values.len();
    let mut candidates: VecDeque<(usize, f32)> = VecDeque::with_capacity(window + 2);
    let mut next: usize = 0;

    for j in 0..len {
        let upper: usize = (j + window).min(len - 1);

        while next <= upper {
            let value: f32 = values[next];
            while candidates
                .back()
                .is_some_and(|&(_, back): &(usize, f32)| back >= value)
            {
                candidates.pop_back();
            }
            candidates.push_back((next, value));
            next += 1;
        }

        while candidates
            .front()
            .is_some_and(|&(index, _): &(usize, f32)| index < j)
        {
            candidates.pop_front();
        }

        values[j] = candidates.front().map_or(1.0, |&(_, v): &(usize, f32)| v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loudness::linear_to_db;
    use crate::testutil::{noise_stereo, sine_stereo};

    const SR: u32 = 48_000;

    fn params() -> LimiterParams {
        LimiterParams {
            ceiling_db: -1.0,
            lookahead_ms: 5.0,
            release_ms: 100.0,
        }
    }

    fn peak(samples: &[f32]) -> f32 {
        samples
            .iter()
            .fold(0.0f32, |acc: f32, &x: &f32| acc.max(x.abs()))
    }

    #[test]
    fn sliding_minimum_matches_brute_force() {
        let original: Vec<f32> = vec![1.0, 0.9, 1.0, 0.4, 1.0, 1.0, 0.7, 1.0, 1.0, 1.0, 0.2];
        for window in [1usize, 2, 3, 5, 20] {
            let mut fast: Vec<f32> = original.clone();
            sliding_minimum_in_place(&mut fast, window);

            for (j, &value) in fast.iter().enumerate() {
                let end: usize = (j + window).min(original.len() - 1);
                let expected: f32 = original[j..=end]
                    .iter()
                    .fold(f32::INFINITY, |acc: f32, &x: &f32| acc.min(x));
                assert_eq!(value, expected, "window {window}, index {j}");
            }
        }
    }

    #[test]
    fn output_never_exceeds_the_ceiling() {
        let ceiling_options: [f32; 3] = [-0.1, -1.0, -6.0];
        let lookahead_options: [f32; 3] = [0.5, 5.0, 20.0];
        let release_options: [f32; 2] = [10.0, 500.0];

        for ceiling_db in ceiling_options {
            for lookahead_ms in lookahead_options {
                for release_ms in release_options {
                    let p: LimiterParams = LimiterParams {
                        ceiling_db,
                        lookahead_ms,
                        release_ms,
                    };
                    let mut audio: AudioBuffer = noise_stereo(6.0, 3.0, SR, 42);
                    apply_limiter(&mut audio, &p).unwrap();

                    let limit: f32 = db_to_linear(ceiling_db) * (1.0 + 1e-6);
                    assert!(peak(&audio.samples) <= limit, "{p:?}");
                }
            }
        }
    }

    #[test]
    fn signal_below_the_ceiling_is_untouched() {
        let input: AudioBuffer = sine_stereo(440.0, -12.0, 1.0, SR);
        let mut output: AudioBuffer = input.clone();
        let stats: LimiterStats = apply_limiter(&mut output, &params()).unwrap();

        assert_eq!(input.samples, output.samples);
        assert_eq!(stats.max_gain_reduction_db, 0.0);
    }

    #[test]
    fn isolated_spike_is_caught_and_the_gain_recovers_afterwards() {
        let mut audio: AudioBuffer = sine_stereo(1000.0, -30.0, 2.0, SR);
        let input: AudioBuffer = audio.clone();
        let spike_frame: usize = SR as usize;
        audio.samples[spike_frame * 2] = 1.0;
        audio.samples[spike_frame * 2 + 1] = 1.0;
        let with_spike: AudioBuffer = audio.clone();

        let stats: LimiterStats = apply_limiter(&mut audio, &params()).unwrap();

        assert!(audio.samples[spike_frame * 2] <= db_to_linear(-1.0) * (1.0 + 1e-6));
        assert!((stats.max_gain_reduction_db - 1.0).abs() < 0.05);

        // Well before the lookahead window the signal is bit-identical.
        let untouched: usize = spike_frame - 3 * (SR as usize / 200);
        assert_eq!(
            &audio.samples[..untouched * 2],
            &with_spike.samples[..untouched * 2]
        );

        // Long after the release time the gain is back to unity.
        let late: usize = spike_frame + SR as usize / 2;
        let ratio_db: f32 = linear_to_db(audio.samples[late * 2].abs())
            - linear_to_db(input.samples[late * 2].abs());
        assert!(ratio_db.abs() < 0.05, "late ratio {ratio_db} dB");
    }

    #[test]
    fn gain_reduction_starts_before_the_peak_arrives() {
        let mut audio: AudioBuffer = sine_stereo(997.0, -30.0, 2.0, SR);
        let input: AudioBuffer = audio.clone();
        let spike_frame: usize = SR as usize;
        audio.samples[spike_frame * 2] = 1.0;
        audio.samples[spike_frame * 2 + 1] = 1.0;

        apply_limiter(&mut audio, &params()).unwrap();

        // Find a frame half the lookahead before the spike with a clearly nonzero input.
        let lookahead: usize = (5.0 * 0.001 * SR as f32) as usize;
        let probe: usize = (spike_frame - lookahead / 2..spike_frame)
            .find(|&f: &usize| input.samples[f * 2].abs() > 0.005)
            .unwrap();
        assert!(audio.samples[probe * 2].abs() < input.samples[probe * 2].abs() * 0.999);
    }

    #[test]
    fn stereo_link_reduces_both_channels_equally() {
        let mut audio: AudioBuffer = sine_stereo(1000.0, -30.0, 1.0, SR);
        let input: AudioBuffer = audio.clone();
        let spike_frame: usize = SR as usize / 2;
        audio.samples[spike_frame * 2] = 1.0;

        apply_limiter(&mut audio, &params()).unwrap();

        let probe: usize = (spike_frame - 3..spike_frame)
            .find(|&f: &usize| input.samples[f * 2].abs() > 0.005)
            .unwrap();
        let left_ratio: f32 = audio.samples[probe * 2] / input.samples[probe * 2];
        let right_ratio: f32 = audio.samples[probe * 2 + 1] / input.samples[probe * 2 + 1];
        assert!((left_ratio - right_ratio).abs() < 1e-6);
        assert!(left_ratio < 1.0);
    }

    #[test]
    fn buffers_shorter_than_the_lookahead_are_handled() {
        let mut audio: AudioBuffer = AudioBuffer {
            samples: vec![1.5, -1.5, 0.2, 0.2, 2.0, 2.0],
            sample_rate: SR,
            channels: 2,
        };
        apply_limiter(&mut audio, &params()).unwrap();
        assert!(peak(&audio.samples) <= db_to_linear(-1.0) * (1.0 + 1e-6));
    }

    #[test]
    fn empty_and_silent_buffers_are_fine() {
        let mut empty: AudioBuffer = AudioBuffer {
            samples: Vec::new(),
            sample_rate: SR,
            channels: 2,
        };
        assert!(apply_limiter(&mut empty, &params()).is_ok());

        let mut silent: AudioBuffer = AudioBuffer {
            samples: vec![0.0; SR as usize * 2],
            sample_rate: SR,
            channels: 2,
        };
        apply_limiter(&mut silent, &params()).unwrap();
        assert!(silent.samples.iter().all(|&x: &f32| x == 0.0));
    }

    #[test]
    fn invalid_parameters_are_rejected() {
        let mut audio: AudioBuffer = sine_stereo(1000.0, -12.0, 0.5, SR);
        let bad: [LimiterParams; 5] = [
            LimiterParams {
                ceiling_db: 1.0,
                ..params()
            },
            LimiterParams {
                ceiling_db: -40.0,
                ..params()
            },
            LimiterParams {
                lookahead_ms: 0.0,
                ..params()
            },
            LimiterParams {
                lookahead_ms: 100.0,
                ..params()
            },
            LimiterParams {
                release_ms: 5000.0,
                ..params()
            },
        ];

        for p in bad {
            assert!(apply_limiter(&mut audio, &p).is_err(), "{p:?}");
        }
    }
}
