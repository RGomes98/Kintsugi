use tch::{Kind, Tensor};

use crate::config::N_CHANNELS;

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
