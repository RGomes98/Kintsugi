use anyhow::Result;
use tch::{Device, Kind, Tensor};

use crate::config::TARGET_FRAMES;
use core::{N_CHANNELS, N_STEMS};

pub(crate) fn overlap_add_output(
    output: &Tensor,
    offset_frames: usize,
    total_frames: usize,
    merge_weight: &[f32],
    sum_weight: &mut [f32],
    accum: &mut [Vec<f32>],
) -> Result<()> {
    let output: Tensor = output
        .to_device(Device::Cpu)
        .to_kind(Kind::Float)
        .contiguous()
        .view([1, N_STEMS as i64, N_CHANNELS as i64, TARGET_FRAMES as i64]);

    let batch0: Tensor = output.get(0);

    for (stem_idx, stem_accum) in accum.iter_mut().enumerate().take(N_STEMS) {
        let stem: Tensor = batch0.get(stem_idx as i64);
        let left_t: Tensor = stem.get(0).contiguous();
        let right_t: Tensor = stem.get(1).contiguous();

        let mut left: Vec<f32> = vec![0.0f32; TARGET_FRAMES];
        let mut right: Vec<f32> = vec![0.0f32; TARGET_FRAMES];
        left_t.copy_data(&mut left, TARGET_FRAMES);
        right_t.copy_data(&mut right, TARGET_FRAMES);

        for i in 0..TARGET_FRAMES {
            let dst_frame: usize = offset_frames + i;
            if dst_frame >= total_frames {
                break;
            }

            let w: f32 = merge_weight[i];
            if w <= 1e-8 {
                continue;
            }

            stem_accum[dst_frame * 2] += left[i] * w;
            stem_accum[dst_frame * 2 + 1] += right[i] * w;

            if stem_idx == 0 {
                sum_weight[dst_frame] += w;
            }
        }
    }

    Ok(())
}
