mod config;
mod model;
mod postprocess;
mod preprocess;

use std::{io::Write, path::Path};

use anyhow::{Result, bail};
use tch::Device;

use crate::config::{MODEL_SAMPLE_RATE, TARGET_FRAMES};
use crate::model::{forward_chunk, load_model, resolve_device_context, validate_output_shape};
use crate::postprocess::overlap_add_output;
use crate::preprocess::make_input_chunk;
use core::{AudioBuffer, N_CHANNELS, N_STEMS, Stems};
use dsp::{
    build_demucs_weight_window, compute_global_norm, compute_shift_frames, finalize_stems,
    normalize_interleaved,
};

pub use config::{DeviceChoice, ExtractOptions};

pub fn extract_stems(
    audio: &AudioBuffer,
    models_dir: &Path,
    options: &ExtractOptions,
    device_choice: DeviceChoice,
) -> Result<Stems> {
    let options: ExtractOptions = options.validate()?;
    validate_audio(audio)?;

    let (device, model_name): (Device, &'static str) = resolve_device_context(device_choice)?;
    println!("Initializing {} on {:?}", model_name, device);

    let model: tch::CModule = load_model(models_dir, model_name, device)?;

    let stride: usize = ((1.0 - options.overlap) * TARGET_FRAMES as f32).round() as usize;
    if stride == 0 {
        bail!("Invalid overlap {}, stride is zero", options.overlap);
    }

    let total_frames: usize = audio.samples.len() / N_CHANNELS;
    let offsets: Vec<usize> = (0..total_frames).step_by(stride).collect();
    let n_chunks: usize = offsets.len();

    let merge_weight: Vec<f32> = build_demucs_weight_window(TARGET_FRAMES, 1.0);

    let (global_mean, global_std): (f32, f32) = compute_global_norm(audio);
    let normalized: Vec<f32> = normalize_interleaved(audio, global_mean, global_std);

    let mut sum_weight: Vec<f32> = vec![0.0f32; total_frames];
    let mut accum: Vec<Vec<f32>> = vec![vec![0.0f32; total_frames * N_CHANNELS]; N_STEMS];
    let num_shifts: usize = options.shifts.max(1);

    for shift_idx in 0..num_shifts {
        let shift_frames: usize = compute_shift_frames(shift_idx, num_shifts, audio.sample_rate);

        for (chunk_idx, &offset) in offsets.iter().enumerate() {
            if num_shifts > 1 {
                print!(
                    "\rShift {}/{} | Chunk {}/{}...",
                    shift_idx + 1,
                    num_shifts,
                    chunk_idx + 1,
                    n_chunks
                );
            } else {
                print!("\rChunk {}/{}...", chunk_idx + 1, n_chunks);
            }

            std::io::stdout().flush()?;

            let shifted_offset: usize = offset.saturating_add(shift_frames);
            let input: tch::Tensor = make_input_chunk(&normalized, shifted_offset, TARGET_FRAMES);

            let output: tch::Tensor = forward_chunk(&model, &input, device)?;
            validate_output_shape(&output)?;

            overlap_add_output(
                &output,
                shifted_offset,
                total_frames,
                &merge_weight,
                &mut sum_weight,
                &mut accum,
            )?;
        }
    }

    let stems: Vec<AudioBuffer> = finalize_stems(
        accum,
        &sum_weight,
        global_mean,
        global_std,
        audio.sample_rate,
        audio.channels,
    )?;

    Ok(Stems {
        drums: stems[0].clone(),
        bass: stems[1].clone(),
        other: stems[2].clone(),
        vocals: stems[3].clone(),
    })
}

fn validate_audio(audio: &AudioBuffer) -> Result<()> {
    if audio.channels != 2 {
        bail!("Demucs expects stereo audio, got {}", audio.channels);
    }

    if audio.sample_rate != MODEL_SAMPLE_RATE {
        bail!(
            "Demucs htdemucs expects 44.1 kHz audio, got {}",
            audio.sample_rate
        );
    }

    if audio.samples.is_empty() {
        bail!("Audio buffer is empty");
    }

    if !audio.samples.len().is_multiple_of(2) {
        bail!("Stereo interleaved buffer must have an even number of samples");
    }

    Ok(())
}
