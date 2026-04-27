use tch::{Kind, Tensor};

use crate::config::N_CHANNELS;
use core::AudioBuffer;

pub(crate) fn compute_global_norm(audio: &AudioBuffer) -> (f32, f32) {
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

pub(crate) fn normalize_interleaved(audio: &AudioBuffer, mean: f32, std: f32) -> Vec<f32> {
    audio
        .samples
        .iter()
        .map(|&x: &f32| (x - mean) / std)
        .collect()
}

pub(crate) fn make_input_chunk(
    normalized: &[f32],
    offset_frames: usize,
    chunk_frames: usize,
) -> Tensor {
    let total_frames: usize = normalized.len() / N_CHANNELS;
    let mut left: Vec<f32> = vec![0.0f32; chunk_frames];
    let mut right: Vec<f32> = vec![0.0f32; chunk_frames];

    for i in 0..chunk_frames {
        let src_frame: usize = offset_frames + i;

        if src_frame >= total_frames {
            break;
        }

        left[i] = normalized[src_frame * 2];
        right[i] = normalized[src_frame * 2 + 1];
    }

    let mut planar: Vec<f32> = Vec::with_capacity(N_CHANNELS * chunk_frames);
    planar.extend_from_slice(&left);
    planar.extend_from_slice(&right);

    Tensor::from_slice(&planar)
        .reshape([1, 2, chunk_frames as i64])
        .to_kind(Kind::Float)
}
