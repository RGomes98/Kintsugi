use std::fmt;

use dsp::dynamics::CompressorStats;
use dsp::limiter::LimiterStats;
use dsp::loudness::{LoudnessStats, format_db};

#[derive(Clone, Debug)]
pub struct MasterReport {
    pub before: LoudnessStats,
    pub after: LoudnessStats,
    pub target_lufs: Option<f32>,
    pub tolerance_lu: f32,
    /// Gain applied by the loudness stage before the limiter.
    pub loudness_gain_db: f32,
    /// Final correction that brought the true peak back under the ceiling.
    pub true_peak_trim_db: f32,
    pub compressor: Option<CompressorStats>,
    /// Offset applied around the compressor to bring the program to the reference level.
    pub compressor_leveling_db: f32,
    pub limiter: Option<LimiterStats>,
    pub iterations: usize,
    /// True when no loudness target was set, or the result landed within tolerance.
    pub converged: bool,
    pub warnings: Vec<String>,
}

impl fmt::Display for MasterReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Loudness       : {} -> {} LUFS",
            format_db(self.before.integrated_lufs),
            format_db(self.after.integrated_lufs)
        )?;
        if let Some(target) = self.target_lufs {
            write!(
                f,
                " (target {target:.1} ±{:.2} LU, {} attempt(s){})",
                self.tolerance_lu,
                self.iterations,
                if self.converged { "" } else { ", not reached" }
            )?;
        }

        write!(
            f,
            "\nTrue peak      : {} -> {} dBTP",
            format_db(self.before.true_peak_dbtp),
            format_db(self.after.true_peak_dbtp)
        )?;
        write!(
            f,
            "\nLoudness range : {:.1} -> {:.1} LU",
            self.before.loudness_range_lu, self.after.loudness_range_lu
        )?;

        if let Some(stats) = &self.compressor {
            write!(
                f,
                "\nCompressor     : gain reduction max {:.1} dB, mean {:.1} dB (input leveled {:+.1} dB)",
                stats.max_gain_reduction_db,
                stats.mean_gain_reduction_db,
                self.compressor_leveling_db
            )?;
        }

        if let Some(stats) = &self.limiter {
            write!(
                f,
                "\nLimiter        : gain reduction max {:.1} dB",
                stats.max_gain_reduction_db
            )?;
        }

        write!(f, "\nLevel gain     : {:+.1} dB", self.loudness_gain_db)?;
        if self.true_peak_trim_db != 0.0 {
            write!(f, " (true-peak trim {:+.2} dB)", self.true_peak_trim_db)?;
        }

        for warning in &self.warnings {
            write!(f, "\nWarning        : {warning}")?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(lufs: f64) -> LoudnessStats {
        LoudnessStats {
            integrated_lufs: lufs,
            loudness_range_lu: 5.0,
            true_peak_dbtp: -1.0,
            sample_peak_db: -1.2,
        }
    }

    #[test]
    fn display_shows_the_key_numbers_and_warnings() {
        let report: MasterReport = MasterReport {
            before: stats(-20.0),
            after: stats(-14.0),
            target_lufs: Some(-14.0),
            tolerance_lu: 0.2,
            loudness_gain_db: 6.0,
            true_peak_trim_db: 0.0,
            compressor: None,
            compressor_leveling_db: 0.0,
            limiter: None,
            iterations: 1,
            converged: true,
            warnings: vec!["be careful".to_string()],
        };

        let text: String = report.to_string();
        assert!(text.contains("-20.0 -> -14.0 LUFS"));
        assert!(text.contains("+6.0 dB"));
        assert!(text.contains("Warning        : be careful"));
        assert!(!text.contains("not reached"));
    }

    #[test]
    fn display_handles_silent_measurements() {
        let report: MasterReport = MasterReport {
            before: stats(f64::NEG_INFINITY),
            after: stats(f64::NEG_INFINITY),
            target_lufs: None,
            tolerance_lu: 0.2,
            loudness_gain_db: 0.0,
            true_peak_trim_db: 0.0,
            compressor: None,
            compressor_leveling_db: 0.0,
            limiter: None,
            iterations: 0,
            converged: true,
            warnings: Vec::new(),
        };

        assert!(report.to_string().contains("-inf -> -inf LUFS"));
    }
}
