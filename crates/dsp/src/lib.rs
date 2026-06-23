use anyhow::{Result, bail};

use core::{AudioBuffer, N_CHANNELS, N_STEMS};

pub fn build_demucs_weight_window(size: usize, transition_power: f32) -> Vec<f32> {
    assert!(size > 0);
    assert!(transition_power >= 1.0);

    let left_len: usize = size / 2;
    let right_len: usize = size - left_len;
    let mut weight: Vec<f32> = Vec::with_capacity(size);

    for i in 0..left_len {
        weight.push((i + 1) as f32);
    }

    for i in 0..right_len {
        weight.push((right_len - i) as f32);
    }

    let max_w: f32 = weight.iter().copied().fold(0.0f32, f32::max).max(1.0);

    weight
        .into_iter()
        .map(|w: f32| (w / max_w).powf(transition_power))
        .collect()
}

pub fn compute_global_norm(audio: &AudioBuffer) -> (f32, f32) {
    let total_frames: usize = audio.samples.len() / N_CHANNELS;

    let mut mono_sum: f32 = 0.0f32;
    for i in 0..total_frames {
        let l: f32 = audio.samples[2 * i];
        let r: f32 = audio.samples[2 * i + 1];
        mono_sum += 0.5 * (l + r);
    }

    let mean: f32 = mono_sum / total_frames as f32;
    let mut var_sum: f32 = 0.0f32;

    for i in 0..total_frames {
        let l: f32 = audio.samples[2 * i];
        let r: f32 = audio.samples[2 * i + 1];
        let m: f32 = 0.5 * (l + r);
        let d: f32 = m - mean;
        var_sum += d * d;
    }

    let std: f32 = (var_sum / total_frames as f32).sqrt().max(1e-5);
    (mean, std)
}

pub fn normalize_interleaved(audio: &AudioBuffer, mean: f32, std: f32) -> Vec<f32> {
    audio
        .samples
        .iter()
        .map(|&x: &f32| (x - mean) / std)
        .collect()
}

pub fn finalize_stems(
    accum: Vec<Vec<f32>>,
    sum_weight: &[f32],
    mean: f32,
    std: f32,
    sample_rate: u32,
    channels: u16,
) -> Result<Vec<AudioBuffer>> {
    let total_frames: usize = sum_weight.len();

    let stems: Vec<AudioBuffer> = accum
        .into_iter()
        .map(|stem: Vec<f32>| {
            let mut out: Vec<f32> = vec![0.0f32; total_frames * N_CHANNELS];

            for frame in 0..total_frames {
                let w: f32 = sum_weight[frame];
                if w <= 1e-8 {
                    continue;
                }

                let l: f32 = stem[frame * 2] / w;
                let r: f32 = stem[frame * 2 + 1] / w;

                let l: f32 = l * std + mean;
                let r: f32 = r * std + mean;

                out[frame * 2] = if l.abs() < 1e-8 { 0.0 } else { l };
                out[frame * 2 + 1] = if r.abs() < 1e-8 { 0.0 } else { r };
            }

            AudioBuffer {
                samples: out,
                sample_rate,
                channels,
            }
        })
        .collect::<Vec<AudioBuffer>>();

    if stems.len() != N_STEMS {
        bail!("Expected {} stems, got {}", N_STEMS, stems.len());
    }

    Ok(stems)
}

pub fn compute_shift_frames(shift_idx: usize, shifts: usize, sample_rate: u32) -> usize {
    if shifts <= 1 {
        return 0;
    }

    let max_shift: usize = (0.5 * sample_rate as f32).round() as usize;
    (shift_idx * max_shift) / shifts
}
