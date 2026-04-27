use anyhow::{Result, bail};
use tch::{Device, Kind, Tensor};

use crate::config::{N_CHANNELS, N_STEMS, TARGET_FRAMES};
use core::AudioBuffer;

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

pub(crate) fn finalize_stems(
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

pub(crate) fn compute_shift_frames(shift_idx: usize, shifts: usize, sample_rate: u32) -> usize {
    if shifts <= 1 {
        return 0;
    }

    let max_shift: usize = (0.5 * sample_rate as f32).round() as usize;
    (shift_idx * max_shift) / shifts
}
