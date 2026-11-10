use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::presets;
use dsp::dynamics::{CompressorParams, validate_compressor};
use dsp::eq::{BandKind, EqBand, validate_band};
use dsp::limiter::{LimiterParams, validate_limiter};

pub const DEFAULT_PRESET: &str = "neutral";

pub const MIN_TARGET_LUFS: f32 = -40.0;
pub const MIN_TOLERANCE_LU: f32 = 0.05;
pub const MAX_TOLERANCE_LU: f32 = 3.0;
pub const MAX_LOUDNESS_GAIN_DB: f32 = 30.0;

/// Sample rate for config-time checks, before any audio has been decoded. The chain
/// validates again against the real sample rate.
const NOMINAL_SAMPLE_RATE: u32 = 48_000;

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BandKindConfig {
    HighPass,
    LowPass,
    LowShelf,
    HighShelf,
    Peak,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EqBandConfig {
    pub kind: BandKindConfig,
    pub freq_hz: f32,
    #[serde(default)]
    pub gain_db: f32,
    #[serde(default = "default_q")]
    pub q: f32,
}

fn default_q() -> f32 {
    0.707
}

impl EqBandConfig {
    pub fn to_band(&self) -> EqBand {
        let kind: BandKind = match self.kind {
            BandKindConfig::HighPass => BandKind::HighPass,
            BandKindConfig::LowPass => BandKind::LowPass,
            BandKindConfig::LowShelf => BandKind::LowShelf,
            BandKindConfig::HighShelf => BandKind::HighShelf,
            BandKindConfig::Peak => BandKind::Peak,
        };

        EqBand {
            kind,
            freq_hz: self.freq_hz,
            gain_db: self.gain_db,
            q: self.q,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LoudnessConfig {
    pub enabled: bool,
    pub target_lufs: f32,
    pub tolerance_lu: f32,
    /// Safety limit on the gain the loudness stage may apply, in either direction.
    pub max_gain_db: f32,
}

#[derive(Clone, Debug)]
pub struct EqConfig {
    pub enabled: bool,
    pub bands: Vec<EqBandConfig>,
}

#[derive(Clone, Debug)]
pub struct CompressorConfig {
    pub enabled: bool,
    pub params: CompressorParams,
}

#[derive(Clone, Debug)]
pub struct LimiterConfig {
    pub enabled: bool,
    pub params: LimiterParams,
}

/// Fully resolved chain settings: `EQ -> compressor -> loudness gain -> limiter`.
#[derive(Clone, Debug)]
pub struct MasterConfig {
    pub loudness: LoudnessConfig,
    pub eq: EqConfig,
    pub compressor: CompressorConfig,
    pub limiter: LimiterConfig,
}

impl Default for MasterConfig {
    fn default() -> Self {
        Self {
            loudness: LoudnessConfig {
                enabled: true,
                target_lufs: -14.0,
                tolerance_lu: 0.2,
                max_gain_db: 12.0,
            },
            eq: EqConfig {
                enabled: false,
                bands: Vec::new(),
            },
            compressor: CompressorConfig {
                enabled: false,
                params: CompressorParams {
                    threshold_db: -18.0,
                    ratio: 2.0,
                    attack_ms: 30.0,
                    release_ms: 200.0,
                    knee_db: 6.0,
                    mix: 1.0,
                },
            },
            limiter: LimiterConfig {
                enabled: true,
                params: LimiterParams {
                    ceiling_db: -1.0,
                    lookahead_ms: 5.0,
                    release_ms: 100.0,
                },
            },
        }
    }
}

/// A TOML document in the same shape as the presets. Every field is optional so a
/// user file only has to mention what it changes. Unknown fields are rejected so a
/// typo is an error instead of a silently ignored setting.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfigFile {
    pub preset: Option<String>,
    pub loudness: Option<LoudnessOverrides>,
    pub eq: Option<EqOverrides>,
    pub compressor: Option<CompressorOverrides>,
    pub limiter: Option<LimiterOverrides>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoudnessOverrides {
    pub enabled: Option<bool>,
    pub target_lufs: Option<f32>,
    pub tolerance_lu: Option<f32>,
    pub max_gain_db: Option<f32>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EqOverrides {
    pub enabled: Option<bool>,
    /// `[[eq.band]]` entries replace the preset's bands as a whole.
    pub band: Option<Vec<EqBandConfig>>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompressorOverrides {
    pub enabled: Option<bool>,
    pub threshold_db: Option<f32>,
    pub ratio: Option<f32>,
    pub attack_ms: Option<f32>,
    pub release_ms: Option<f32>,
    pub knee_db: Option<f32>,
    pub mix: Option<f32>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LimiterOverrides {
    pub enabled: Option<bool>,
    pub ceiling_db: Option<f32>,
    pub lookahead_ms: Option<f32>,
    pub release_ms: Option<f32>,
}

fn set<T: Copy>(slot: &mut T, value: Option<T>) {
    if let Some(value) = value {
        *slot = value;
    }
}

impl MasterConfig {
    /// Writing a section turns its stage on unless it says `enabled = false`.
    fn apply(&mut self, overrides: &ConfigFile) {
        if let Some(o) = &overrides.loudness {
            self.loudness.enabled = o.enabled.unwrap_or(true);
            set(&mut self.loudness.target_lufs, o.target_lufs);
            set(&mut self.loudness.tolerance_lu, o.tolerance_lu);
            set(&mut self.loudness.max_gain_db, o.max_gain_db);
        }

        if let Some(o) = &overrides.eq {
            self.eq.enabled = o.enabled.unwrap_or(true);
            if let Some(bands) = &o.band {
                self.eq.bands = bands.clone();
            }
        }

        if let Some(o) = &overrides.compressor {
            let params: &mut CompressorParams = &mut self.compressor.params;
            self.compressor.enabled = o.enabled.unwrap_or(true);
            set(&mut params.threshold_db, o.threshold_db);
            set(&mut params.ratio, o.ratio);
            set(&mut params.attack_ms, o.attack_ms);
            set(&mut params.release_ms, o.release_ms);
            set(&mut params.knee_db, o.knee_db);
            set(&mut params.mix, o.mix);
        }

        if let Some(o) = &overrides.limiter {
            let params: &mut LimiterParams = &mut self.limiter.params;
            self.limiter.enabled = o.enabled.unwrap_or(true);
            set(&mut params.ceiling_db, o.ceiling_db);
            set(&mut params.lookahead_ms, o.lookahead_ms);
            set(&mut params.release_ms, o.release_ms);
        }
    }

    /// Checks every enabled stage. Disabled stages are not checked.
    pub fn validate(&self, sample_rate: u32) -> Result<()> {
        if self.loudness.enabled {
            let l: &LoudnessConfig = &self.loudness;

            if !(MIN_TARGET_LUFS..=0.0).contains(&l.target_lufs) {
                bail!(
                    "loudness.target_lufs {} is outside [{}, 0] LUFS",
                    l.target_lufs,
                    MIN_TARGET_LUFS
                );
            }

            if !(MIN_TOLERANCE_LU..=MAX_TOLERANCE_LU).contains(&l.tolerance_lu) {
                bail!(
                    "loudness.tolerance_lu {} is outside [{}, {}] LU",
                    l.tolerance_lu,
                    MIN_TOLERANCE_LU,
                    MAX_TOLERANCE_LU
                );
            }

            if !(0.0..=MAX_LOUDNESS_GAIN_DB).contains(&l.max_gain_db) {
                bail!(
                    "loudness.max_gain_db {} is outside [0, {}] dB",
                    l.max_gain_db,
                    MAX_LOUDNESS_GAIN_DB
                );
            }
        }

        if self.eq.enabled {
            for (index, band) in self.eq.bands.iter().enumerate() {
                validate_band(&band.to_band(), sample_rate)
                    .with_context(|| format!("Invalid eq.band #{}", index + 1))?;
            }
        }

        if self.compressor.enabled {
            validate_compressor(&self.compressor.params).context("Invalid [compressor]")?;
        }

        if self.limiter.enabled {
            validate_limiter(&self.limiter.params).context("Invalid [limiter]")?;
        }

        Ok(())
    }
}

/// Layers settings as preset, then the user's file, then returns the result validated.
/// The preset comes from `cli_preset`, else the file's `preset` key, else `neutral`.
/// Command-line flags are applied by the caller afterwards, followed by `validate`.
pub fn resolve(cli_preset: Option<&str>, file: &ConfigFile) -> Result<MasterConfig> {
    let name: &str = cli_preset
        .or(file.preset.as_deref())
        .unwrap_or(DEFAULT_PRESET);

    let mut config: MasterConfig = MasterConfig::default();
    config.apply(&presets::preset(name)?);
    config.apply(file);
    config.validate(NOMINAL_SAMPLE_RATE)?;
    Ok(config)
}

pub fn load_config(preset: Option<&str>, config_path: Option<&Path>) -> Result<MasterConfig> {
    let file: ConfigFile = match config_path {
        Some(path) => {
            let text: String = fs::read_to_string(path)
                .with_context(|| format!("Failed to read config file {}", path.display()))?;
            toml::from_str(&text)
                .with_context(|| format!("Invalid config file {}", path.display()))?
        }
        None => ConfigFile::default(),
    };

    resolve(preset, &file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presets::PRESET_NAMES;

    fn parse(text: &str) -> ConfigFile {
        toml::from_str(text).unwrap()
    }

    #[test]
    fn every_builtin_preset_resolves_and_validates_at_common_sample_rates() {
        for name in PRESET_NAMES {
            let config: MasterConfig = resolve(Some(name), &ConfigFile::default()).unwrap();
            for sample_rate in [44_100, 48_000, 96_000] {
                config
                    .validate(sample_rate)
                    .unwrap_or_else(|e| panic!("preset {name} at {sample_rate} Hz: {e:#}"));
            }
        }
    }

    #[test]
    fn default_preset_is_neutral_loudness_and_limiter_only() {
        let config: MasterConfig = resolve(None, &ConfigFile::default()).unwrap();
        assert!(config.loudness.enabled);
        assert!(config.limiter.enabled);
        assert!(!config.eq.enabled);
        assert!(!config.compressor.enabled);
        assert_eq!(config.loudness.target_lufs, -14.0);
    }

    #[test]
    fn unknown_preset_lists_the_available_ones() {
        let error: String = format!(
            "{:#}",
            resolve(Some("warm"), &ConfigFile::default()).unwrap_err()
        );
        assert!(error.contains("warm"));
        for name in PRESET_NAMES {
            assert!(error.contains(name), "{error}");
        }
    }

    #[test]
    fn file_values_override_the_preset_and_keep_the_rest() {
        let file: ConfigFile = parse("preset = \"gentle\"\n[compressor]\nratio = 3.0\n");
        let config: MasterConfig = resolve(None, &file).unwrap();
        let gentle: MasterConfig = resolve(Some("gentle"), &ConfigFile::default()).unwrap();

        assert_eq!(config.compressor.params.ratio, 3.0);
        assert_eq!(
            config.compressor.params.threshold_db,
            gentle.compressor.params.threshold_db
        );
        assert_eq!(config.eq.bands.len(), gentle.eq.bands.len());
    }

    #[test]
    fn writing_a_section_enables_its_stage() {
        let file: ConfigFile = parse("[compressor]\nthreshold_db = -20.0\n");
        let config: MasterConfig = resolve(Some("neutral"), &file).unwrap();
        assert!(config.compressor.enabled);
        assert_eq!(config.compressor.params.threshold_db, -20.0);
    }

    #[test]
    fn enabled_false_turns_a_preset_stage_off() {
        let file: ConfigFile = parse("[compressor]\nenabled = false\n");
        let config: MasterConfig = resolve(Some("gentle"), &file).unwrap();
        assert!(!config.compressor.enabled);
        assert!(config.eq.enabled);
    }

    #[test]
    fn eq_bands_in_the_file_replace_the_preset_bands() {
        let file: ConfigFile =
            parse("[[eq.band]]\nkind = \"peak\"\nfreq_hz = 1000.0\ngain_db = 2.0\n");
        let config: MasterConfig = resolve(Some("gentle"), &file).unwrap();
        assert_eq!(config.eq.bands.len(), 1);
        assert_eq!(config.eq.bands[0].kind, BandKindConfig::Peak);
        assert_eq!(config.eq.bands[0].q, 0.707);
    }

    #[test]
    fn cli_preset_wins_over_the_files_preset_key() {
        let file: ConfigFile = parse("preset = \"loud\"\n");
        let from_file: MasterConfig = resolve(None, &file).unwrap();
        let from_cli: MasterConfig = resolve(Some("vintage"), &file).unwrap();
        assert_eq!(from_file.loudness.target_lufs, -10.0);
        assert_eq!(from_cli.loudness.target_lufs, -16.0);
    }

    #[test]
    fn typos_and_unknown_fields_are_rejected() {
        assert!(toml::from_str::<ConfigFile>("[loudness]\ntargett_lufs = -14.0\n").is_err());
        assert!(toml::from_str::<ConfigFile>("[mastering]\nx = 1\n").is_err());
        assert!(
            toml::from_str::<ConfigFile>("[[eq.band]]\nkind = \"bell\"\nfreq_hz = 1000.0\n")
                .is_err()
        );
        assert!(
            toml::from_str::<ConfigFile>("[[eq.band]]\nkind = \"peak\"\nfreq = 1000.0\n").is_err()
        );
    }

    #[test]
    fn out_of_range_values_are_rejected_with_context() {
        for text in [
            "[loudness]\ntarget_lufs = 3.0\n",
            "[loudness]\ntolerance_lu = 0.0\n",
            "[loudness]\nmax_gain_db = 60.0\n",
            "[compressor]\nratio = 50.0\n",
            "[limiter]\nceiling_db = 2.0\n",
            "[[eq.band]]\nkind = \"peak\"\nfreq_hz = 1000.0\ngain_db = 40.0\n",
        ] {
            assert!(resolve(None, &parse(text)).is_err(), "{text}");
        }
    }

    #[test]
    fn disabled_stages_are_not_validated() {
        let file: ConfigFile = parse("[compressor]\nenabled = false\nratio = 50.0\n");
        assert!(resolve(None, &file).is_ok());
    }

    #[test]
    fn load_config_reads_a_file_from_disk() {
        let path =
            std::env::temp_dir().join(format!("kintsugi-config-{}.toml", std::process::id()));
        fs::write(
            &path,
            "preset = \"loud\"\n[loudness]\ntarget_lufs = -12.0\n",
        )
        .unwrap();

        let config: Result<MasterConfig> = load_config(None, Some(&path));
        fs::remove_file(&path).ok();

        let config: MasterConfig = config.unwrap();
        assert_eq!(config.loudness.target_lufs, -12.0);
        assert!(config.compressor.enabled);
    }

    #[test]
    fn load_config_reports_a_missing_file() {
        let error: String = format!(
            "{:#}",
            load_config(None, Some(Path::new("/nonexistent/kintsugi.toml"))).unwrap_err()
        );
        assert!(error.contains("kintsugi.toml"));
    }
}
