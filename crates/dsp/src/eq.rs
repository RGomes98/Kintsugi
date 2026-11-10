use anyhow::{Result, anyhow, bail};
use biquad::{Biquad, Coefficients, DirectForm2Transposed, ToHertz, Type};

use core::{AudioBuffer, N_CHANNELS};

pub const MIN_BAND_FREQ_HZ: f32 = 10.0;
/// Fraction of the sample rate above which a band is rejected (0.45 * fs, below Nyquist).
pub const MAX_BAND_FREQ_RATIO: f32 = 0.45;
pub const MAX_BAND_GAIN_DB: f32 = 24.0;
pub const MIN_BAND_Q: f32 = 0.1;
pub const MAX_BAND_Q: f32 = 20.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BandKind {
    HighPass,
    LowPass,
    LowShelf,
    HighShelf,
    Peak,
}

impl BandKind {
    fn uses_gain(self) -> bool {
        matches!(self, Self::LowShelf | Self::HighShelf | Self::Peak)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EqBand {
    pub kind: BandKind,
    pub freq_hz: f32,
    pub gain_db: f32,
    pub q: f32,
}

pub fn validate_band(band: &EqBand, sample_rate: u32) -> Result<()> {
    let max_freq: f32 = sample_rate as f32 * MAX_BAND_FREQ_RATIO;

    if !(MIN_BAND_FREQ_HZ..=max_freq).contains(&band.freq_hz) {
        bail!(
            "EQ band frequency {} Hz is outside [{}, {}] Hz for {} Hz audio",
            band.freq_hz,
            MIN_BAND_FREQ_HZ,
            max_freq,
            sample_rate
        );
    }

    if !(MIN_BAND_Q..=MAX_BAND_Q).contains(&band.q) {
        bail!(
            "EQ band Q {} is outside [{}, {}]",
            band.q,
            MIN_BAND_Q,
            MAX_BAND_Q
        );
    }

    if band.kind.uses_gain() && band.gain_db.abs() > MAX_BAND_GAIN_DB {
        bail!(
            "EQ band gain {} dB is outside [-{}, {}] dB",
            band.gain_db,
            MAX_BAND_GAIN_DB,
            MAX_BAND_GAIN_DB
        );
    }

    Ok(())
}

fn band_coefficients(band: &EqBand, sample_rate: u32) -> Result<Coefficients<f64>> {
    validate_band(band, sample_rate)?;

    let gain_db: f64 = band.gain_db as f64;
    let filter_type: Type<f64> = match band.kind {
        BandKind::HighPass => Type::HighPass,
        BandKind::LowPass => Type::LowPass,
        BandKind::LowShelf => Type::LowShelf(gain_db),
        BandKind::HighShelf => Type::HighShelf(gain_db),
        BandKind::Peak => Type::PeakingEQ(gain_db),
    };

    Coefficients::<f64>::from_params(
        filter_type,
        (sample_rate as f64).hz(),
        (band.freq_hz as f64).hz(),
        band.q as f64,
    )
    .map_err(|e| anyhow!("Invalid EQ band {band:?}: {e:?}"))
}

/// Applies the bands in series. Coefficients and filter state are f64 because
/// low-frequency biquads accumulate audible noise in f32.
pub fn apply_eq(audio: &mut AudioBuffer, bands: &[EqBand]) -> Result<()> {
    if audio.channels as usize != N_CHANNELS {
        bail!("EQ expects stereo audio, got {} channel(s)", audio.channels);
    }

    let coefficients: Vec<Coefficients<f64>> = bands
        .iter()
        .map(|band: &EqBand| band_coefficients(band, audio.sample_rate))
        .collect::<Result<Vec<Coefficients<f64>>>>()?;

    for coeffs in coefficients {
        for channel in 0..N_CHANNELS {
            let mut filter: DirectForm2Transposed<f64> = DirectForm2Transposed::<f64>::new(coeffs);

            for sample in audio.samples.iter_mut().skip(channel).step_by(N_CHANNELS) {
                *sample = filter.run(*sample as f64) as f32;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{rms_db, sine_stereo};

    const SR: u32 = 48_000;
    const SKIP: usize = SR as usize / 2;

    fn gain_at(freq_hz: f64, band: EqBand, sample_rate: u32) -> f32 {
        let input: AudioBuffer = sine_stereo(freq_hz, -20.0, 2.0, sample_rate);
        let mut output: AudioBuffer = input.clone();
        apply_eq(&mut output, &[band]).unwrap();
        rms_db(&output.samples, SKIP) - rms_db(&input.samples, SKIP)
    }

    fn band(kind: BandKind, freq_hz: f32, gain_db: f32, q: f32) -> EqBand {
        EqBand {
            kind,
            freq_hz,
            gain_db,
            q,
        }
    }

    #[test]
    fn peak_boosts_center_and_spares_distant_frequencies() {
        let b: EqBand = band(BandKind::Peak, 1000.0, 6.0, 1.0);
        assert!((gain_at(1000.0, b, SR) - 6.0).abs() < 0.2);
        assert!(gain_at(100.0, b, SR).abs() < 0.5);
    }

    #[test]
    fn peak_cut_attenuates_center() {
        let b: EqBand = band(BandKind::Peak, 2000.0, -9.0, 2.0);
        assert!((gain_at(2000.0, b, SR) - -9.0).abs() < 0.2);
    }

    #[test]
    fn low_shelf_lifts_lows_and_leaves_highs() {
        let b: EqBand = band(BandKind::LowShelf, 200.0, 6.0, 0.707);
        assert!((gain_at(40.0, b, SR) - 6.0).abs() < 0.5);
        assert!(gain_at(5000.0, b, SR).abs() < 0.3);
    }

    #[test]
    fn high_shelf_cuts_highs_and_leaves_lows() {
        let b: EqBand = band(BandKind::HighShelf, 4000.0, -6.0, 0.707);
        assert!((gain_at(14000.0, b, SR) - -6.0).abs() < 0.5);
        assert!(gain_at(200.0, b, SR).abs() < 0.3);
    }

    #[test]
    fn high_pass_removes_subsonic_content_and_passes_the_rest() {
        let b: EqBand = band(BandKind::HighPass, 100.0, 0.0, 0.707);
        assert!(gain_at(20.0, b, SR) < -20.0);
        assert!(gain_at(1000.0, b, SR).abs() < 0.1);
    }

    #[test]
    fn low_pass_removes_highs_and_passes_the_rest() {
        let b: EqBand = band(BandKind::LowPass, 8000.0, 0.0, 0.707);
        assert!(gain_at(20000.0, b, SR) < -15.0);
        assert!(gain_at(1000.0, b, SR).abs() < 0.1);
    }

    #[test]
    fn behaves_the_same_at_other_sample_rates() {
        let b: EqBand = band(BandKind::Peak, 1000.0, 6.0, 1.0);
        assert!((gain_at(1000.0, b, 44_100) - 6.0).abs() < 0.2);
        assert!((gain_at(1000.0, b, 96_000) - 6.0).abs() < 0.2);
    }

    #[test]
    fn no_bands_leaves_audio_untouched() {
        let input: AudioBuffer = sine_stereo(440.0, -12.0, 1.0, SR);
        let mut output: AudioBuffer = input.clone();
        apply_eq(&mut output, &[]).unwrap();
        assert_eq!(input.samples, output.samples);
    }

    #[test]
    fn channels_are_filtered_independently() {
        let mut audio: AudioBuffer = sine_stereo(1000.0, -12.0, 1.0, SR);
        for frame in audio.samples.as_chunks_mut::<2>().0.iter_mut() {
            frame[1] = 0.0;
        }

        apply_eq(&mut audio, &[band(BandKind::Peak, 1000.0, 6.0, 1.0)]).unwrap();

        let frames: &[[f32; 2]] = audio.samples.as_chunks::<2>().0;
        assert!(frames.iter().all(|f: &[f32; 2]| f[1] == 0.0));
        assert!(frames.iter().any(|f: &[f32; 2]| f[0] != 0.0));
    }

    #[test]
    fn invalid_bands_are_rejected() {
        let mut audio: AudioBuffer = sine_stereo(1000.0, -12.0, 0.5, SR);

        for bad in [
            band(BandKind::Peak, 5.0, 3.0, 1.0),
            band(BandKind::Peak, 30_000.0, 3.0, 1.0),
            band(BandKind::Peak, 1000.0, 3.0, 0.0),
            band(BandKind::Peak, 1000.0, 3.0, 50.0),
            band(BandKind::Peak, 1000.0, 40.0, 1.0),
        ] {
            assert!(apply_eq(&mut audio, &[bad]).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn high_pass_ignores_gain_value() {
        let mut audio: AudioBuffer = sine_stereo(1000.0, -12.0, 0.5, SR);
        let b: EqBand = band(BandKind::HighPass, 100.0, 99.0, 0.707);
        assert!(apply_eq(&mut audio, &[b]).is_ok());
    }
}
