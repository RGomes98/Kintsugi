use anyhow::{Result, bail};

use crate::config::MasterConfig;
use crate::report::MasterReport;
use core::AudioBuffer;
use dsp::dynamics::{CompressorStats, apply_compressor};
use dsp::eq::{EqBand, apply_eq};
use dsp::limiter::{LimiterStats, apply_limiter};
use dsp::loudness::{
    LoudnessStats, MIN_MEASURABLE_SECONDS, apply_gain_db, measure_integrated_lufs, measure_loudness,
};

/// Program loudness the compressor thresholds are written for.
pub const COMPRESSOR_REFERENCE_LUFS: f64 = -18.0;

const MAX_LOUDNESS_ITERATIONS: usize = 4;
/// The trim aims this far under the ceiling to cover meter under-read and float rounding.
const TRUE_PEAK_MARGIN_DB: f64 = dsp::loudness::TRUE_PEAK_MAX_UNDER_READ_DB + 0.02;

#[derive(Debug)]
pub struct MasterOutcome {
    pub audio: AudioBuffer,
    pub report: MasterReport,
}

struct Rendered {
    audio: AudioBuffer,
    stats: LoudnessStats,
    limiter: Option<LimiterStats>,
    trim_db: f32,
}

struct LevelOutcome {
    rendered: Rendered,
    gain_db: f32,
    iterations: usize,
    converged: bool,
}

/// Runs `EQ -> compressor -> loudness gain -> limiter` on a stereo mix.
///
/// The loudness stage measures what actually comes out of the limiter and corrects the
/// gain until the integrated loudness is within tolerance (at most
/// `MAX_LOUDNESS_ITERATIONS` attempts). When the limiter is on, the result is
/// guaranteed not to exceed its ceiling in true peak, even if the loudness target
/// could not be reached; that case is reported as a warning.
pub fn master_audio(input: &AudioBuffer, config: &MasterConfig) -> Result<MasterOutcome> {
    config.validate(input.sample_rate)?;

    let before: LoudnessStats = measure_loudness(input)?;
    if config.loudness.enabled && before.is_silent() {
        bail!(
            "Cannot normalize loudness: the input is digital silence or shorter than {MIN_MEASURABLE_SECONDS} s"
        );
    }

    let mut warnings: Vec<String> = Vec::new();
    let mut work: AudioBuffer = input.clone();

    if config.eq.enabled && !config.eq.bands.is_empty() {
        let bands: Vec<EqBand> = config.eq.bands.iter().map(|band| band.to_band()).collect();
        apply_eq(&mut work, &bands)?;
    }

    let mut compressor_leveling_db: f32 = 0.0;
    let compressor: Option<CompressorStats> = if config.compressor.enabled {
        let program_lufs: f64 = measure_integrated_lufs(&work)?;
        if program_lufs.is_finite() {
            compressor_leveling_db = (COMPRESSOR_REFERENCE_LUFS - program_lufs) as f32;
        }

        apply_gain_db(&mut work.samples, compressor_leveling_db);
        let stats: CompressorStats = apply_compressor(&mut work, &config.compressor.params)?;
        apply_gain_db(&mut work.samples, -compressor_leveling_db);
        Some(stats)
    } else {
        None
    };

    let level: LevelOutcome = if config.loudness.enabled {
        normalize_loudness(&work, config, &mut warnings)?
    } else {
        let rendered: Rendered = render(&work, 0.0, config)?;
        LevelOutcome {
            rendered,
            gain_db: 0.0,
            iterations: 0,
            converged: true,
        }
    };

    let Rendered {
        audio,
        stats: after,
        limiter,
        trim_db,
    } = level.rendered;

    if !config.limiter.enabled && after.true_peak_dbtp > 0.0 {
        warnings.push(format!(
            "Output reaches {:.1} dBTP and may clip; enable the limiter or lower the loudness target",
            after.true_peak_dbtp
        ));
    }

    Ok(MasterOutcome {
        audio,
        report: MasterReport {
            before,
            after,
            target_lufs: config
                .loudness
                .enabled
                .then_some(config.loudness.target_lufs),
            tolerance_lu: config.loudness.tolerance_lu,
            loudness_gain_db: level.gain_db,
            true_peak_trim_db: trim_db,
            compressor,
            compressor_leveling_db,
            limiter,
            iterations: level.iterations,
            converged: level.converged,
            warnings,
        },
    })
}

/// Applies `gain_db`, then the limiter, then the true-peak trim, and measures the result.
fn render(pre: &AudioBuffer, gain_db: f32, config: &MasterConfig) -> Result<Rendered> {
    let mut audio: AudioBuffer = pre.clone();
    apply_gain_db(&mut audio.samples, gain_db);

    let limiter: Option<LimiterStats> = if config.limiter.enabled {
        Some(apply_limiter(&mut audio, &config.limiter.params)?)
    } else {
        None
    };

    let mut stats: LoudnessStats = measure_loudness(&audio)?;
    let mut trim_db: f32 = 0.0;

    if config.limiter.enabled {
        let ceiling: f64 = config.limiter.params.ceiling_db as f64;

        if stats.true_peak_dbtp > ceiling - TRUE_PEAK_MARGIN_DB {
            trim_db = (ceiling - TRUE_PEAK_MARGIN_DB - stats.true_peak_dbtp) as f32;
            apply_gain_db(&mut audio.samples, trim_db);
            // A plain gain shifts loudness and peaks by `trim_db`; no need to re-measure.
            stats.integrated_lufs += trim_db as f64;
            stats.true_peak_dbtp += trim_db as f64;
            stats.sample_peak_db += trim_db as f64;
        }
    }

    Ok(Rendered {
        audio,
        stats,
        limiter,
        trim_db,
    })
}

fn normalize_loudness(
    pre: &AudioBuffer,
    config: &MasterConfig,
    warnings: &mut Vec<String>,
) -> Result<LevelOutcome> {
    let target: f64 = config.loudness.target_lufs as f64;
    let tolerance: f64 = config.loudness.tolerance_lu as f64;
    let max_gain: f32 = config.loudness.max_gain_db;

    let pre_lufs: f64 = measure_integrated_lufs(pre)?;
    if !pre_lufs.is_finite() {
        bail!("The signal is silent after EQ and compression; cannot normalize loudness");
    }

    let wanted_gain: f32 = (target - pre_lufs) as f32;
    let mut gain_db: f32 = wanted_gain.clamp(-max_gain, max_gain);
    if gain_db != wanted_gain {
        warnings.push(format!(
            "Reaching {target:.1} LUFS needs {wanted_gain:+.1} dB of gain, limited to {gain_db:+.1} dB by loudness.max_gain_db"
        ));
    }

    let mut best: Option<(f64, f32, Rendered)> = None;
    let mut iterations: usize = 0;
    let mut converged: bool = false;

    for _ in 0..MAX_LOUDNESS_ITERATIONS {
        iterations += 1;

        let rendered: Rendered = render(pre, gain_db, config)?;
        let error: f64 = target - rendered.stats.integrated_lufs;
        let within_tolerance: bool = error.abs() <= tolerance;

        if best
            .as_ref()
            .is_none_or(|(best_error, _, _): &(f64, f32, Rendered)| error.abs() < *best_error)
        {
            best = Some((error.abs(), gain_db, rendered));
        }

        if within_tolerance {
            converged = true;
            break;
        }

        let next_gain: f32 = (gain_db + error as f32).clamp(-max_gain, max_gain);
        if (next_gain - gain_db).abs() < 0.001 {
            break;
        }
        gain_db = next_gain;
    }

    let (_, best_gain, rendered): (f64, f32, Rendered) =
        best.expect("the loop always renders at least once");

    if !converged {
        let hint: &str = if best_gain.abs() >= max_gain - 0.001 {
            "raise loudness.max_gain_db or lower the target"
        } else {
            "the limiter is holding the level down; raise limiter.ceiling_db or lower the target"
        };

        warnings.push(format!(
            "Loudness ended {:+.2} LU from the {target:.1} LUFS target after {iterations} attempt(s): {hint}",
            rendered.stats.integrated_lufs - target
        ));
    }

    Ok(LevelOutcome {
        rendered,
        gain_db: best_gain,
        iterations,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ConfigFile, load_config, resolve};
    use dsp::testutil::{modulated_noise_stereo, noise_stereo};

    const SR: u32 = 48_000;

    fn neutral() -> MasterConfig {
        load_config(Some("neutral"), None).unwrap()
    }

    fn sample_peak(audio: &AudioBuffer) -> f32 {
        audio
            .samples
            .iter()
            .fold(0.0f32, |acc: f32, &x: &f32| acc.max(x.abs()))
    }

    #[test]
    fn neutral_preset_reaches_the_target_and_respects_the_ceiling() {
        let input: AudioBuffer = modulated_noise_stereo(-12.0, 8.0, SR, 1);
        let outcome: MasterOutcome = master_audio(&input, &neutral()).unwrap();
        let report: &MasterReport = &outcome.report;

        assert!(report.converged, "{report}");
        assert!(
            (report.after.integrated_lufs - -14.0).abs() <= 0.2,
            "{report}"
        );
        assert!(report.after.true_peak_dbtp <= -1.0, "{report}");
        assert!(report.warnings.is_empty(), "{report}");
    }

    #[test]
    fn loudness_correction_accounts_for_what_the_limiter_removes() {
        // Quiet program with sparse loud clicks: reaching the target means boosting the
        // clicks far above the ceiling, so the limiter pulls the level down and the
        // loudness stage has to compensate by re-measuring.
        let mut input: AudioBuffer = noise_stereo(-30.0, 8.0, SR, 2);
        let click_every: usize = SR as usize / 2;
        for frame in (click_every..input.samples.len() / 2).step_by(click_every) {
            input.samples[frame * 2] = 0.7;
            input.samples[frame * 2 + 1] = 0.7;
        }

        let mut config: MasterConfig = neutral();
        config.loudness.max_gain_db = 30.0;
        let outcome: MasterOutcome = master_audio(&input, &config).unwrap();
        let report: &MasterReport = &outcome.report;

        assert!(
            report.limiter.unwrap().max_gain_reduction_db > 3.0,
            "{report}"
        );
        assert!(report.iterations > 1, "{report}");
        assert!(report.after.true_peak_dbtp <= -1.0, "{report}");
        assert!(
            (report.after.integrated_lufs - -14.0).abs() <= 0.3,
            "{report}"
        );
    }

    #[test]
    fn true_peak_ceiling_holds_even_when_the_target_is_out_of_reach() {
        let input: AudioBuffer = modulated_noise_stereo(-1.0, 6.0, SR, 3);
        let mut config: MasterConfig = neutral();
        config.loudness.target_lufs = -4.0;
        config.loudness.max_gain_db = 30.0;

        let outcome: MasterOutcome = master_audio(&input, &config).unwrap();

        assert!(
            outcome.report.after.true_peak_dbtp <= -1.0,
            "{}",
            outcome.report
        );
        assert!(sample_peak(&outcome.audio) <= dsp::loudness::db_to_linear(-1.0));
        assert!(!outcome.report.warnings.is_empty());
    }

    #[test]
    fn quiet_input_is_capped_by_max_gain_and_says_so() {
        let input: AudioBuffer = modulated_noise_stereo(-45.0, 6.0, SR, 4);
        let outcome: MasterOutcome = master_audio(&input, &neutral()).unwrap();
        let report: &MasterReport = &outcome.report;

        assert!(!report.converged);
        assert_eq!(report.loudness_gain_db, 12.0);
        assert!(report.after.integrated_lufs < -14.5);
        assert!(
            report
                .warnings
                .iter()
                .any(|w: &String| w.contains("max_gain_db")),
            "{report}"
        );
    }

    #[test]
    fn raising_max_gain_lets_quiet_input_converge() {
        let input: AudioBuffer = modulated_noise_stereo(-32.0, 6.0, SR, 5);
        let mut config: MasterConfig = neutral();
        config.loudness.max_gain_db = 30.0;

        let outcome: MasterOutcome = master_audio(&input, &config).unwrap();
        assert!(outcome.report.converged, "{}", outcome.report);
        assert!((outcome.report.after.integrated_lufs - -14.0).abs() <= 0.2);
    }

    #[test]
    fn everything_disabled_is_a_bit_exact_passthrough() {
        let input: AudioBuffer = modulated_noise_stereo(-10.0, 4.0, SR, 6);
        let mut config: MasterConfig = neutral();
        config.loudness.enabled = false;
        config.limiter.enabled = false;

        let outcome: MasterOutcome = master_audio(&input, &config).unwrap();

        assert_eq!(outcome.audio.samples, input.samples);
        assert_eq!(
            outcome.report.before.integrated_lufs,
            outcome.report.after.integrated_lufs
        );
        assert!(outcome.report.warnings.is_empty());
    }

    #[test]
    fn same_input_and_config_give_identical_output() {
        let input: AudioBuffer = modulated_noise_stereo(-6.0, 6.0, SR, 7);
        let config: MasterConfig = load_config(Some("gentle"), None).unwrap();

        let first: MasterOutcome = master_audio(&input, &config).unwrap();
        let second: MasterOutcome = master_audio(&input, &config).unwrap();

        assert_eq!(first.audio.samples, second.audio.samples);
    }

    #[test]
    fn full_chain_presets_meet_the_target_and_the_ceiling() {
        let input: AudioBuffer = modulated_noise_stereo(-8.0, 8.0, SR, 8);
        let neutral_out: MasterOutcome = master_audio(&input, &neutral()).unwrap();

        for name in ["gentle", "loud", "vintage"] {
            let config: MasterConfig = load_config(Some(name), None).unwrap();
            let outcome: MasterOutcome = master_audio(&input, &config).unwrap();
            let report: &MasterReport = &outcome.report;
            let target: f64 = config.loudness.target_lufs as f64;

            assert!(report.after.true_peak_dbtp <= -1.0, "{name}: {report}");
            assert!(
                (report.after.integrated_lufs - target).abs() <= 0.3,
                "{name}: {report}"
            );
            assert!(report.compressor.is_some(), "{name}");
            assert_ne!(outcome.audio.samples, neutral_out.audio.samples, "{name}");
        }
    }

    #[test]
    fn compression_does_not_depend_on_how_hot_the_input_is() {
        let mut config: MasterConfig = load_config(Some("gentle"), None).unwrap();
        // Test noise has a low crest factor; a lower threshold makes the compressor work.
        config.compressor.params.threshold_db = -26.0;
        let hot: AudioBuffer = modulated_noise_stereo(-2.0, 8.0, SR, 12);
        let mut quiet: AudioBuffer = hot.clone();
        apply_gain_db(&mut quiet.samples, -20.0);

        let hot_out: MasterOutcome = master_audio(&hot, &config).unwrap();
        let quiet_out: MasterOutcome = master_audio(&quiet, &config).unwrap();
        let hot_gr: f32 = hot_out.report.compressor.unwrap().mean_gain_reduction_db;
        let quiet_gr: f32 = quiet_out.report.compressor.unwrap().mean_gain_reduction_db;

        assert!(hot_gr > 0.5, "{}", hot_out.report);
        assert!(
            (hot_gr - quiet_gr).abs() < 0.01,
            "hot {hot_gr} dB vs quiet {quiet_gr} dB"
        );
        assert!(
            (hot_out.report.compressor_leveling_db - quiet_out.report.compressor_leveling_db
                + 20.0)
                .abs()
                < 0.01
        );
    }

    #[test]
    fn leveling_equals_shifting_the_threshold_and_keeps_the_level() {
        // Leveling must equal shifting the threshold by the program's offset.
        let input: AudioBuffer = modulated_noise_stereo(-3.0, 6.0, SR, 13);
        let mut config: MasterConfig = load_config(Some("gentle"), None).unwrap();
        config.eq.enabled = false;
        config.loudness.enabled = false;
        config.limiter.enabled = false;

        let outcome: MasterOutcome = master_audio(&input, &config).unwrap();

        let program_lufs: f64 = measure_integrated_lufs(&input).unwrap();
        let mut params: dsp::dynamics::CompressorParams = config.compressor.params;
        params.threshold_db += (program_lufs - COMPRESSOR_REFERENCE_LUFS) as f32;
        let mut expected: AudioBuffer = input.clone();
        apply_compressor(&mut expected, &params).unwrap();

        let max_diff: f32 = outcome
            .audio
            .samples
            .iter()
            .zip(&expected.samples)
            .fold(0.0f32, |acc: f32, (a, b): (&f32, &f32)| {
                acc.max((a - b).abs())
            });
        assert!(max_diff < 1e-4, "max difference {max_diff}");
    }

    #[test]
    fn reported_output_stats_match_a_fresh_measurement() {
        // Samples of a quarter-rate sine at 45 degrees sit 3 dB under its crest, so only
        // the true-peak trim acts on it.
        let frames: usize = SR as usize * 3;
        let samples: Vec<f32> = (0..frames)
            .flat_map(|i: usize| {
                let v: f32 =
                    (std::f32::consts::FRAC_PI_2 * i as f32 + std::f32::consts::FRAC_PI_4).sin();
                [v, v]
            })
            .collect();
        let input: AudioBuffer = AudioBuffer {
            samples,
            sample_rate: SR,
            channels: 2,
        };
        let mut config: MasterConfig = neutral();
        config.loudness.enabled = false;
        let outcome: MasterOutcome = master_audio(&input, &config).unwrap();
        let fresh: LoudnessStats = measure_loudness(&outcome.audio).unwrap();
        let report: &MasterReport = &outcome.report;

        assert!(report.true_peak_trim_db < 0.0, "{report}");
        assert!((report.after.integrated_lufs - fresh.integrated_lufs).abs() < 0.001);
        assert!((report.after.true_peak_dbtp - fresh.true_peak_dbtp).abs() < 0.001);
        assert!((report.after.sample_peak_db - fresh.sample_peak_db).abs() < 0.001);
        assert!(fresh.true_peak_dbtp <= -1.0, "{}", fresh.true_peak_dbtp);
    }

    #[test]
    fn silent_input_is_rejected_when_normalizing_but_allowed_otherwise() {
        let silent: AudioBuffer = AudioBuffer {
            samples: vec![0.0; SR as usize * 2 * 3],
            sample_rate: SR,
            channels: 2,
        };

        let error: String = format!("{:#}", master_audio(&silent, &neutral()).unwrap_err());
        assert!(error.contains("silence"), "{error}");

        let mut config: MasterConfig = neutral();
        config.loudness.enabled = false;
        let outcome: MasterOutcome = master_audio(&silent, &config).unwrap();
        assert!(outcome.audio.samples.iter().all(|&x: &f32| x == 0.0));
    }

    #[test]
    fn mono_input_is_rejected() {
        let mono: AudioBuffer = AudioBuffer {
            samples: vec![0.1; SR as usize * 2],
            sample_rate: SR,
            channels: 1,
        };
        assert!(master_audio(&mono, &neutral()).is_err());
    }

    #[test]
    fn works_at_other_sample_rates() {
        for sample_rate in [44_100, 96_000] {
            let input: AudioBuffer = modulated_noise_stereo(-12.0, 6.0, sample_rate, 9);
            let config: MasterConfig = load_config(Some("gentle"), None).unwrap();
            let outcome: MasterOutcome = master_audio(&input, &config).unwrap();

            assert!(
                outcome.report.converged,
                "{sample_rate} Hz: {}",
                outcome.report
            );
            assert!(outcome.report.after.true_peak_dbtp <= -1.0);
        }
    }

    #[test]
    fn disabled_limiter_warns_when_the_output_can_clip() {
        let input: AudioBuffer = noise_stereo(-1.0, 6.0, SR, 10);
        let mut config: MasterConfig = neutral();
        config.limiter.enabled = false;
        config.loudness.target_lufs = -2.0;

        let outcome: MasterOutcome = master_audio(&input, &config).unwrap();

        assert!(
            outcome.report.after.true_peak_dbtp > 0.0,
            "{}",
            outcome.report
        );
        assert!(
            outcome
                .report
                .warnings
                .iter()
                .any(|w: &String| w.contains("clip")),
            "{}",
            outcome.report
        );
    }

    #[test]
    fn user_config_can_change_the_result_without_touching_the_preset() {
        let file: ConfigFile = toml::from_str("[loudness]\ntarget_lufs = -18.0\n").unwrap();
        let config: MasterConfig = resolve(Some("neutral"), &file).unwrap();
        let input: AudioBuffer = modulated_noise_stereo(-12.0, 6.0, SR, 11);

        let outcome: MasterOutcome = master_audio(&input, &config).unwrap();
        assert!((outcome.report.after.integrated_lufs - -18.0).abs() <= 0.2);
    }
}
